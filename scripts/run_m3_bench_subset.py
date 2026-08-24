#!/usr/bin/env python3
"""Run the M3 multi-modal pipeline over a subset of M3-Bench videos.

Driver for scaling the single-video ``m3.run_video_pipeline`` flow from
1 video to N videos. For each ``video_id`` in the requested subset the
script:

1. Resolves the source video path via ``m3.config.resolve_video_path``
   (so it works whether the bench data lives under ``data/videos``,
   ``data/m3/videos``, or a relocated root).
2. Spawns ``python -m m3.run_video_pipeline`` as a subprocess with the
   full pipeline spec (``video_episodic_full`` by default), the real
   M3-Bench ``robot.json`` annotations file, and ``--set
   qa.params.video=<id>`` to filter the qa_list down to this video.
3. Collects the per-video ``m3_qa_results.jsonl`` into a merged JSONL
   at ``<work-dir-root>/all_m3_qa_results.jsonl``.
4. Optionally runs the M3 judge (``m3.eval``) over the merged JSONL
   to produce ``m3_eval_summary.json``.

Resumable: by default a video is skipped if its per-video
``m3_qa_results.jsonl`` already contains records for every question
listed in the annotations file (use ``--force`` to override).

This is the Option A (subprocess loop) implementation described in the
M3-Bench multi-video plan. The alternative — a single-process
in-process orchestrator sharing one graph store across videos — is
covered by running with ``--graph-store sql`` and is left as a future
optimization.

LLM / config wiring
-------------------

The QA stage reads its LLM provider / model / API key from the
``m3.qa_llm`` block of the YAML passed via ``--config``. The driver
does NOT override this — leave ``deploy/ogmemory.yaml`` (or your
envsubst-resolved copy) as the single source of truth.

Two supported workflows:

**Workflow A — envsubst-resolved YAML (preferred)::

    set -a && source deploy/.env && set +a
    envsubst < deploy/ogmemory.yaml > data/ogmem_test_lyj.resolved.yaml
    uv run python scripts/run_m3_bench_subset.py \\
        --config data/ogmem_test_lyj.resolved.yaml \\
        --videos bedroom_03,living_room_06 \\
        --work-dir-root data/m3_runs/m3bench_smoke

The resolved YAML has hardcoded values, no ``${VAR}`` refs. The driver
detects this and skips env injection (it's a no-op).

**Workflow B — original YAML with placeholders::

    uv run python scripts/run_m3_bench_subset.py \\
        --config deploy/ogmemory.yaml \\
        ...

If ``m3.qa_llm`` in the original YAML has unresolved ``${LLM_API_KEY}``
refs and the parent shell didn't export them, the driver copies the
hardcoded ``m3.episodic_llm`` values into ``LLM_API_KEY`` /
``LLM_BASE_URL`` / etc. before spawning the subprocess. This is a
back-compat shim — workflow A is cleaner.

To override the QA provider for one specific run without editing the
YAML, pass ``--set qa.params.llm=openai`` (or any supported provider);
this is forwarded as a stage-param override.

Usage::

    ./scripts/run_m3_bench_subset.py \\
        --videos bedroom_03,living_room_06,kitchen_01 \\
        --annotations data/m3/annotations/robot.json \\
        --work-dir-root data/m3_runs/m3bench_pilot \\
        --pipeline config/m3/pipelines/video_episodic_full.yaml \\
        --seconds 120 --judge

Notes on what already works today (no code changes needed in the core
pipeline):
- ``m3.pipeline.stages.qa_stage`` (line 400) reads
  ``stage.params["annotations"]`` and falls back to the auto-generated
  ``one_video.json`` only when the param is unset.
- ``run_react_qa`` (m3/qa.py:383) splits ``--video`` on commas and
  filters ``qa_list`` records whose ``video_id`` matches, so a single
  run only answers questions for the requested video.

If you want to switch to a real multi-video graph (Option B), pass
``--graph-store sql`` plus ``--connection-string`` so all videos share
one store. The per-process driver shape stays the same.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from m3.config import resolve_video_path  # noqa: E402

logger = logging.getLogger(__name__)


def resolve_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    """Build the env passed to each per-video subprocess.

    Workflow A — envsubst-resolved YAML (preferred)::

        set -a && source deploy/.env && set +a
        envsubst < deploy/ogmemory.yaml > data/ogmem_test_lyj.resolved.yaml
        # pass --config data/ogmem_test_lyj.resolved.yaml to the driver

    The resolved YAML has hardcoded values (no ``${VAR}`` refs).
    ``m3.config._load_yaml_config`` will read those values directly, no
    env vars needed. This function detects that and returns the parent
    env unchanged.

    Workflow B — original YAML with ``${VAR}`` placeholders::

        # deploy/ogmemory.yaml still has api_key: "${LLM_API_KEY}"
        # m3.qa_llm block was empty, m3.episodic_llm has hardcoded key.
        # The QA stage would fail with "LLM_API_KEY not set" because the
        # parent shell doesn't export it.

    To make workflow B work, copy the hardcoded ``episodic_llm`` values
    into the standard env var names that ``m3.config._expand_env_ref``
    looks up (``LLM_PROVIDER``, ``LLM_API_KEY``, ``LLM_BASE_URL``,
    ``LLM_MODEL``). The QA stage then resolves its ``${LLM_API_KEY}``
    reference against the injected env. Users who don't want this can
    run their pipeline with a resolved YAML (workflow A) and the
    injection becomes a no-op.
    """
    env = os.environ.copy()
    try:
        import yaml

        cfg_path = Path(args.config).resolve()
        if not cfg_path.exists():
            return env
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        m3 = cfg.get("m3") or {}

        # Detect workflow A: no ${VAR} placeholders anywhere in m3.qa_llm
        # or m3.episodic_llm. The resolved YAML doesn't need env help.
        def has_unresolved_refs(block: dict[str, Any]) -> bool:
            for v in block.values():
                if (
                    isinstance(v, str)
                    and v.startswith("${")
                    and v.endswith("}")
                ):
                    return True
            return False

        qa_block = m3.get("qa_llm") or {}
        ep_block = m3.get("episodic_llm") or {}
        if not has_unresolved_refs(qa_block) and not has_unresolved_refs(ep_block):
            logger.debug(
                "config %s has no ${VAR} refs in m3.qa_llm/m3.episodic_llm — "
                "skipping env injection (workflow A: envsubst-resolved).",
                cfg_path,
            )
            return env

        # Workflow B: inject qa_llm (preferred) or episodic_llm values into
        # the env so the QA stage can resolve its ${LLM_*} placeholders.
        for block_name in ("qa_llm", "episodic_llm"):
            block = m3.get(block_name) or {}
            if not isinstance(block, dict):
                continue
            mapping = {
                "provider": "LLM_PROVIDER",
                "api_key": "LLM_API_KEY",
                "base_url": "LLM_BASE_URL",
                "model": "LLM_MODEL",
            }
            injected_any = False
            for yaml_key, env_key in mapping.items():
                value = block.get(yaml_key)
                if not value or (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    continue
                env[env_key] = str(value)
                injected_any = True
            if injected_any:
                # qa_llm wins; if it's incomplete fall through to
                # episodic_llm only for fields qa_llm left unset.
                if block_name == "qa_llm":
                    break
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("could not load %s for env injection: %s", args.config, exc)
    return env


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def parse_csv(value: str | None) -> list[str]:
    """Split a comma-separated id list, stripping whitespace.

    Empty input -> empty list (lets ``--videos`` and ``--exclude`` compose).
    """
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def select_subset(
    annotations: dict[str, Any],
    include: list[str],
    exclude: list[str],
    limit: int | None,
) -> list[str]:
    """Resolve the requested subset to a concrete list of video_ids.

    Order is preserved: explicit ``--videos`` is honored first, then any
    remaining video_ids from the annotation file in JSON-iteration order
    (subject to ``--exclude`` and ``--limit``).
    """
    all_ids = list(annotations.keys())
    if include:
        missing = [v for v in include if v not in annotations]
        if missing:
            raise SystemExit(
                f"--videos references ids not in annotations: {missing}"
            )
        return include
    selected = [v for v in all_ids if v not in set(exclude)]
    if limit is not None:
        selected = selected[:limit]
    return selected


# ---------------------------------------------------------------------------
# Per-video run + resume
# ---------------------------------------------------------------------------


def load_questions_for_video(
    annotations: dict[str, Any], video_id: str
) -> list[dict[str, Any]]:
    """Return the qa_list for ``video_id`` from the loaded annotations."""
    entry = annotations.get(video_id)
    if entry is None:
        raise KeyError(f"video_id {video_id!r} not in annotations")
    return list(entry.get("qa_list") or [])


def already_done(
    work_dir_root: Path,
    video_id: str,
    expected_qids: set[str],
) -> bool:
    """Has this video already produced records for every expected question?

    Returns True if the per-video JSONL exists and contains every
    ``(video_id, question_id)`` we'd run, AND every record is non-empty
    (no ``error`` field). Empty JSONL or partial coverage -> False.
    """
    jsonl = work_dir_root / video_id / "m3_qa_results.jsonl"
    if not jsonl.exists():
        return False
    seen: set[str] = set()
    saw_error = False
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = rec.get("question_id")
        if qid:
            seen.add(str(qid))
        if rec.get("error"):
            saw_error = True
    if saw_error:
        return False
    return expected_qids.issubset(seen)


def build_pipeline_cmd(
    *,
    python: list[str],
    video_path: Path,
    video_id: str,
    annotations_path: Path,
    pipeline: Path,
    config_path: Path,
    seconds: int,
    work_dir: Path,
    graph_store: str,
    connection_string: str | None,
    account_id: str,
    enable_stages: list[str],
    disable_stages: list[str],
    set_values: list[str],
    extra_qa_params: list[str],
    force: bool,
    verbose: bool,
) -> list[str]:
    """Compose the ``python -m m3.run_video_pipeline`` argv.

    The two key overrides:
      * ``--set qa.params.annotations=<robot.json>`` so the QA stage reads
        the real qa_list instead of the empty auto-generated stub.
      * ``--set qa.params.video=<video_id>`` so the adapter only answers
        this video's questions (robot.json contains all 1276 QAs across
        100 videos).

    ``python`` is a *list* (e.g. ``["uv", "run", "python"]`` or
    ``["/path/to/.venv/bin/python"]``) so the subprocess inherits the
    same venv / env that the driver was launched from. Using the system
    ``/usr/bin/python3`` will fail because it lacks the project's
    optional deps (``typing_extensions``, ``faster_whisper``,
    ``speechbrain``, etc.).
    """
    cmd: list[str] = [
        *python,
        "-m", "m3.run_video_pipeline",
        "--video", str(video_path),
        "--video-id", video_id,
        "--seconds", str(seconds),
        "--pipeline", str(pipeline),
        "--work-dir", str(work_dir),
        "--config", str(config_path),
        "--graph-store", graph_store,
        "--account-id", account_id,
        # Tell QA where the real qa_list lives and which video to answer.
        "--set", f"qa.params.annotations={annotations_path}",
        "--set", f"qa.params.video={video_id}",
    ]
    if connection_string:
        cmd.extend(["--connection-string", connection_string])
    for stage in enable_stages:
        cmd.extend(["--enable-stage", stage])
    for stage in disable_stages:
        cmd.extend(["--disable-stage", stage])
    for sv in set_values:
        cmd.extend(["--set", sv])
    for extra in extra_qa_params:
        cmd.extend(["--set", extra])
    if force:
        cmd.append("--force")
    if verbose:
        cmd.append("--verbose")
    return cmd


def run_one_video(
    *,
    video_id: str,
    video_path: Path,
    args: argparse.Namespace,
    annotations: dict[str, Any],
    expected_qids: set[str],
) -> dict[str, Any]:
    """Run the pipeline for one video; return a summary dict.

    Returns ``{"video_id": ..., "status": "ok|skipped|failed",
    "duration_sec": ..., "jsonl": str, "returncode": int}``. Caller
    can persist these summaries for an aggregate run report.
    """
    work_dir = args.work_dir_root / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and already_done(
        args.work_dir_root, video_id, expected_qids
    ):
        logger.info("[%s] already complete — skipping (use --force to rerun)",
                    video_id)
        return {
            "video_id": video_id,
            "status": "skipped",
            "duration_sec": 0.0,
            "jsonl": str(work_dir / "m3_qa_results.jsonl"),
            "returncode": 0,
        }

    set_values = list(args.set or [])
    extra_qa_params = [
        f"qa.params.max_iterations={args.qa_max_iterations}",
        f"qa.params.timeout_seconds={args.qa_timeout}",
        f"qa.params.output_jsonl={work_dir / 'm3_qa_results.jsonl'}",
    ]
    if args.qa_top_k:
        extra_qa_params.append(f"qa.params.top_k={args.qa_top_k}")

    python_argv = shlex.split(args.python)
    cmd = build_pipeline_cmd(
        python=python_argv,
        video_path=video_path,
        video_id=video_id,
        annotations_path=args.annotations,
        pipeline=args.pipeline,
        config_path=args.config,
        seconds=args.seconds,
        work_dir=work_dir,
        graph_store=args.graph_store,
        connection_string=args.connection_string,
        account_id=args.account_id,
        enable_stages=args.enable_stage,
        disable_stages=args.disable_stage,
        set_values=set_values,
        extra_qa_params=extra_qa_params,
        force=args.force,
        verbose=args.verbose,
    )

    log_path = work_dir / "pipeline.log"
    logger.info("[%s] running: %s", video_id, " ".join(shlex.quote(c) for c in cmd))
    logger.info("[%s] streaming logs to %s", video_id, log_path)
    start = time.monotonic()

    if args.dry_run:
        return {
            "video_id": video_id,
            "status": "dry_run",
            "duration_sec": 0.0,
            "jsonl": str(work_dir / "m3_qa_results.jsonl"),
            "returncode": 0,
            "command": cmd,
        }

    sub_env = resolve_subprocess_env(args)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=sub_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        "video_id": video_id,
        "status": status,
        "duration_sec": round(duration, 2),
        "jsonl": str(work_dir / "m3_qa_results.jsonl"),
        "returncode": proc.returncode,
    }


# ---------------------------------------------------------------------------
# Aggregation + optional judge
# ---------------------------------------------------------------------------


def aggregate_jsonl(
    video_ids: Iterable[str],
    work_dir_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Concatenate per-video ``m3_qa_results.jsonl`` files into one.

    Skips missing/empty files. Returns counts for the run report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    answered = 0
    failed = 0
    per_video: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for vid in video_ids:
            jsonl = work_dir_root / vid / "m3_qa_results.jsonl"
            if not jsonl.exists():
                logger.warning("[%s] no m3_qa_results.jsonl at %s", vid, jsonl)
                continue
            count = 0
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                out.write(line + "\n")
                count += 1
                total_records += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    failed += 1
                    continue
                if rec.get("error"):
                    failed += 1
                else:
                    answered += 1
            per_video[vid] = count
    return {
        "merged_jsonl": str(out_path),
        "total_records": total_records,
        "answered": answered,
        "failed": failed,
        "per_video": per_video,
    }


def run_judge(
    *,
    annotations: Path,
    merged_jsonl: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Invoke ``python -m m3.eval`` to judge the merged JSONL.

    Returns the parsed ``m3_eval_summary.json`` if available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    python_argv = shlex.split(args.python)
    cmd = [
        *python_argv,
        "-m", "m3.eval",
        "--annotations", str(annotations),
        "--input-qa-results", str(merged_jsonl),
        "--output-dir", str(output_dir),
        "--graph-store", args.graph_store,
        "--account-id", args.account_id,
    ]
    if args.connection_string:
        cmd.extend(["--connection-string", args.connection_string])
    if args.judge_llm:
        cmd.extend(["--judge-provider", args.judge_llm])
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    if args.judge_base_url:
        cmd.extend(["--judge-base-url", args.judge_base_url])
    if args.judge_api_key:
        cmd.extend(["--judge-api-key", args.judge_api_key])
    if args.resume:
        cmd.append("--resume")
    logger.info("judge: %s", " ".join(shlex.quote(c) for c in cmd))
    log_path = output_dir / "judge.log"
    sub_env = resolve_subprocess_env(args)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=sub_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = output_dir / "m3_eval_summary.json"
    summary: dict[str, Any] = {
        "returncode": proc.returncode,
        "summary_path": str(summary_path) if summary_path.exists() else None,
    }
    if summary_path.exists():
        try:
            summary["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("could not parse %s as JSON", summary_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the M3 multi-modal pipeline over a subset of "
            "M3-Bench videos (default annotations: data/m3/annotations/robot.json)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations",
                   default="data/m3/annotations/robot.json",
                   help="Path to M3-Bench annotations JSON (e.g. robot.json).")
    p.add_argument("--videos", default=None,
                   help="Comma-separated list of video_ids to run. "
                        "Default: all (subject to --limit and --exclude).")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated list of video_ids to skip.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the subset size when --videos is not given.")
    p.add_argument("--work-dir-root", default="data/m3_runs/m3bench_subset",
                   help="Root directory; each video gets <work-dir-root>/<video_id>/.")
    p.add_argument("--videos-root", default="data/m3/videos",
                   help="Root directory passed to m3.config.resolve_video_path. "
                        "The M3-Bench robot.json declares paths like "
                        "'data/videos/robot/<id>.mp4' but on this layout the "
                        "actual files live under 'data/m3/videos/robot/' "
                        "(data/m3 is a symlink to /data3/m3); override when "
                        "running elsewhere.")
    p.add_argument("--pipeline",
                   default="config/m3/pipelines/video_episodic_full.yaml",
                   help="Pipeline YAML spec to run per video.")
    p.add_argument("--config", default="deploy/ogmemory.yaml",
                   help="Path to ogmemory.yaml (LLM/embedding defaults). "
                        "Used both as the subprocess --config and as the "
                        "source of LLM API keys for the env injection.")
    p.add_argument("--seconds", type=int, default=120,
                   help="Trim each video to its first N seconds (<=0 = full).")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="Required when --graph-store=sql.")
    p.add_argument("--account-id", default="m3bench")
    p.add_argument("--enable-stage", action="append", default=[],
                   help="Forwarded to run_video_pipeline --enable-stage (repeatable).")
    p.add_argument("--disable-stage", action="append", default=[],
                   help="Forwarded to run_video_pipeline --disable-stage (repeatable).")
    p.add_argument("--set", action="append", default=[],
                   help="Forwarded to run_video_pipeline --set (repeatable).")
    p.add_argument("--qa-max-iterations", type=int, default=6)
    p.add_argument("--qa-timeout", type=int, default=90)
    p.add_argument("--qa-top-k", type=int, default=None,
                   help="Forwarded as qa.params.top_k. The QA LLM provider / "
                        "model / API key come from m3.qa_llm in --config; "
                        "to override the provider for one run use "
                        "--set qa.params.llm=openai.")
    p.add_argument("--force", action="store_true",
                   help="Rerun videos even if their JSONL looks complete.")
    p.add_argument("--resume", action="store_true",
                   help="Forward --resume to m3.eval during the judge pass.")
    p.add_argument("--judge", action="store_true",
                   help="After all videos, invoke m3.eval over the merged JSONL.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the merged JSONL + judge output "
                        "(default: <work-dir-root>/_aggregate).")
    p.add_argument("--judge-llm", choices=("minimax", "openai", "mock"), default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--python", default="uv run python",
                   help="Python invocation used for subprocesses. Defaults to "
                        "'uv run python' so the spawned pipeline inherits the "
                        "project venv (typing_extensions, faster_whisper, etc.). "
                        "Override with a path like '/abs/path/to/.venv/bin/python' "
                        "or a different launcher. Anything before the final token "
                        "is treated as the launcher (e.g. 'uv run') and tokens "
                        "after are passed through as a single argv.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print each per-video command and exit without running.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d video entries from %s",
                len(annotations), annotations_path)

    args.annotations = annotations_path  # normalize to absolute
    args.config = Path(args.config).resolve()
    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 1
    args.work_dir_root = Path(args.work_dir_root).resolve()
    args.work_dir_root.mkdir(parents=True, exist_ok=True)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else args.work_dir_root / "_aggregate"
    )

    subset = select_subset(
        annotations,
        include=parse_csv(args.videos),
        exclude=parse_csv(args.exclude),
        limit=args.limit,
    )
    if not subset:
        logger.error("subset is empty after applying --videos/--exclude/--limit")
        return 1
    logger.info("Running subset (%d videos): %s", len(subset), subset)

    project_root = REPO_ROOT
    videos_root = (project_root / args.videos_root).resolve() \
        if not Path(args.videos_root).is_absolute() \
        else Path(args.videos_root).resolve()

    summaries: list[dict[str, Any]] = []
    for video_id in subset:
        declared = annotations[video_id].get("video_path", "")
        video_path = resolve_video_path(
            declared,
            project_root=project_root,
            videos_root=videos_root,
        )
        if video_path is None:
            logger.error("[%s] cannot resolve video_path=%r under %s",
                         video_id, declared, videos_root)
            summaries.append({
                "video_id": video_id,
                "status": "failed",
                "reason": "video_not_found",
                "declared_video_path": declared,
            })
            continue
        expected_qids = {
            str(q.get("question_id"))
            for q in load_questions_for_video(annotations, video_id)
            if q.get("question_id") is not None
        }
        try:
            summary = run_one_video(
                video_id=video_id,
                video_path=video_path,
                args=args,
                annotations=annotations,
                expected_qids=expected_qids,
            )
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("[%s] driver raised: %s", video_id, exc)
            summary = {
                "video_id": video_id,
                "status": "failed",
                "reason": "driver_exception",
                "error": str(exc),
            }
        summaries.append(summary)

    merged_jsonl = output_dir / "all_m3_qa_results.jsonl"
    agg = aggregate_jsonl(subset, args.work_dir_root, merged_jsonl)
    run_report_path = output_dir / "run_report.json"
    run_report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "annotations": str(annotations_path),
        "pipeline": str(Path(args.pipeline).resolve()),
        "seconds": args.seconds,
        "graph_store": args.graph_store,
        "subset": subset,
        "per_video": summaries,
        "aggregate": agg,
    }
    run_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Run report: %s", run_report_path)
    logger.info("Merged JSONL: %s (%d records, %d answered, %d failed)",
                merged_jsonl, agg["total_records"], agg["answered"], agg["failed"])

    if args.judge and not args.dry_run:
        judge_summary = run_judge(
            annotations=annotations_path,
            merged_jsonl=merged_jsonl,
            output_dir=output_dir,
            args=args,
        )
        report["judge"] = judge_summary
        run_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    failed = [s for s in summaries if s.get("status") == "failed"]
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())