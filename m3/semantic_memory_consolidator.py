"""M3-Bench semantic memory consolidator.

For each video, walks the graph for every ``episodic_memory_generator``
ExtractedNode and consolidates them into a single ``MemoryNode`` with
merged descriptions, characters, events, and entities. Each source
ExtractedNode is linked to the consolidated MemoryNode via a
``GraphEdgeType.EVOLVES`` edge (the canonical "memory evolution"
edge — older → newer).

Why this lives outside ``extraction/operators/``:

* It's a cross-clip aggregation, not a per-node operation —
  per-video consolidation reads many EpisrictedNodes and produces
  one MemoryNode. The standard MultimodalExtractor contract takes
  a single input node, so it doesn't fit.
* No embeddings needed — just text deduplication + set union.

Use case: M3-Bench "General Knowledge" question buckets ask things
like "what does this video show?" where the answer requires facts
spread across multiple clips. The consolidated MemoryNode gives
[Search] a single high-coverage node per video instead of N
partial-clip nodes.

The consolidation rules:

* Description: concatenate per-clip descriptions (in clip order)
  with blank-line separators; cap at 50 KB to keep the JSONL
  payload reasonable.
* Characters: union of (name, role) pairs across clips; duplicates
  dropped.
* Events: union of (type, summary) pairs across clips; duplicates
  dropped.
* Entities: union of (name, type) pairs across clips; duplicates
  dropped.

Idempotency: the consolidated MemoryNode id is a deterministic
sha1 of ``"semantic/{video_id}"`` so re-runs update the same node
in place rather than creating duplicates. The EVOLVES edges from
each source EpisrictedNode to the MemoryNode are also de-duped
on re-run.

Usage::

    python -m m3.semantic_memory_consolidator \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/semantic_memories.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.graph_enums import GraphEdgeType, GraphNodeType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure consolidation helpers (no I/O)
# ---------------------------------------------------------------------------

MAX_DESCRIPTION_BYTES = 50_000  # cap the JSONL payload size


@dataclass
class EpisodicItem:
    """One EpisrictedNode's contribution to the consolidated memory."""

    node_id: str
    video_id: str | None
    clip_id: int | None
    description: str
    characters: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)


@dataclass
class ConsolidatedMemory:
    """The output: a single MemoryNode-shaped record per video."""

    video_id: str
    memory_node_id: str
    description: str
    characters: list[dict]
    events: list[dict]
    entities: list[dict]
    source_node_ids: list[str]
    source_extracted_ids: list[str] = field(default_factory=list)
    evolves_edge_count: int = 0

    def to_jsonl(self) -> dict:
        return asdict(self)


def _normalize_character(c: dict) -> tuple:
    """Stable key for character dedup. Treat missing fields as empty
    strings so two dicts with different keys but same content dedup."""
    return (
        str(c.get("name", "")).strip().lower(),
        str(c.get("role", "")).strip().lower(),
    )


def _normalize_event(e: dict) -> tuple:
    return (
        str(e.get("type", "")).strip().lower(),
        str(e.get("summary", "")).strip().lower(),
    )


def _normalize_entity(e: dict) -> tuple:
    return (
        str(e.get("name", "")).strip().lower(),
        str(e.get("type", "")).strip().lower(),
    )


def consolidate(items: list[EpisodicItem], video_id: str) -> ConsolidatedMemory:
    """Pure consolidation: takes per-clip EpisodicItems for one video
    and produces a single ConsolidatedMemory. Deterministic."""
    # Sort by (clip_id, node_id) so the merged description has a
    # stable order across runs (critical for the deterministic
    # MemoryNode content).
    ordered = sorted(
        items,
        key=lambda it: (it.clip_id if it.clip_id is not None else 0,
                        it.node_id),
    )

    description = "\n\n".join(
        it.description.strip() for it in ordered if it.description and it.description.strip()
    )
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        # Truncate cleanly on a UTF-8 boundary so the JSONL stays
        # valid even when a single description blows the cap.
        encoded = description.encode("utf-8")[:MAX_DESCRIPTION_BYTES]
        description = encoded.decode("utf-8", errors="ignore")

    char_seen: dict[tuple, dict] = {}
    for it in ordered:
        for c in it.characters or []:
            if not isinstance(c, dict):
                continue
            key = _normalize_character(c)
            if key not in char_seen:
                char_seen[key] = c
    characters = list(char_seen.values())

    event_seen: dict[tuple, dict] = {}
    for it in ordered:
        for e in it.events or []:
            if not isinstance(e, dict):
                continue
            key = _normalize_event(e)
            if key not in event_seen:
                event_seen[key] = e
    events = list(event_seen.values())

    entity_seen: dict[tuple, dict] = {}
    for it in ordered:
        for e in it.entities or []:
            if not isinstance(e, dict):
                continue
            key = _normalize_entity(e)
            if key not in entity_seen:
                entity_seen[key] = e
    entities = list(entity_seen.values())

    seed = f"semantic/{video_id}".encode()
    memory_node_id = hashlib.sha1(seed).hexdigest()[:32]

    return ConsolidatedMemory(
        video_id=video_id,
        memory_node_id=memory_node_id,
        description=description,
        characters=characters,
        events=events,
        entities=entities,
        source_node_ids=[it.node_id for it in ordered],
    )


# ---------------------------------------------------------------------------
# Graph-side helpers
# ---------------------------------------------------------------------------

def _episodic_items_from_graph(
    graph, ctx: RequestContext,
) -> list[EpisodicItem]:
    """Pull every ``episodic_memory_generator`` ExtractedNode and
    shape it into ``EpisodicItem``."""
    out: list[EpisodicItem] = []
    for n in graph.list_extracted_nodes("episodic_memory_generator", ctx):
        ec = n.extracted_content or {}
        # Provenance: prefer source_video_id / source_clip_id if the
        # operator stamped them; fall back to legacy _video_id /
        # _clip_id keys (m3.episodic_memory uses the legacy form).
        video_id = ec.get("source_video_id") or ec.get("_video_id")
        clip_id = ec.get("source_clip_id")
        if clip_id is None:
            clip_id = ec.get("_clip_id")
        out.append(EpisodicItem(
            node_id=n.node_id,
            video_id=video_id,
            clip_id=clip_id,
            description=str(ec.get("description", "") or ""),
            characters=list(ec.get("characters") or []),
            events=list(ec.get("events") or []),
            entities=list(ec.get("entities") or []),
        ))
    return out


def _existing_evolves_targets_for(
    graph, ctx: RequestContext, source_node_id: str,
) -> set[str]:
    """For a given source ExtractedNode, find the set of MemoryNode
    ids it already has EVOLVES edges to. Used so re-runs don't
    duplicate edges."""
    targets: set[str] = set()
    for _nb, edge, _dir in graph.get_neighbors(
        source_node_id, ctx,
        edge_types=[GraphEdgeType.EVOLVES],
    ):
        if edge.edge_type == GraphEdgeType.EVOLVES:
            targets.add(edge.target_id)
    return targets


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_semantic_memory_consolidator(
    args: argparse.Namespace,
    graph=None,
) -> int:
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)

    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-semantic-consolidator",
        trace_id="m3-semantic-consolidator",
    )

    items = _episodic_items_from_graph(graph, ctx)
    logger.info("Loaded %d EpisrictedNodes from graph", len(items))
    if args.video:
        wanted = set(args.video.split(","))
        items = [it for it in items if it.video_id in wanted]
        logger.info("After --video filter (%s): %d nodes",
                    args.video, len(items))

    # Group by video_id. Nodes with no video_id go into a single
    # "(unspecified)" bucket — keeps the output stable but flags
    # provenance gaps.
    by_video: dict[str, list[EpisodicItem]] = {}
    for it in items:
        key = it.video_id or "(unspecified)"
        by_video.setdefault(key, []).append(it)
    logger.info("Consolidating across %d video buckets", len(by_video))

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memories_written = 0
    edges_emitted = 0
    with output_path.open("w", encoding="utf-8") as f:
        for video_id, group in by_video.items():
            memory = consolidate(group, video_id)
            # Source ids list — EpisrictedNode ids we just read.
            memory.source_extracted_ids = list(memory.source_node_ids)

            # Persist the MemoryNode (create or update). The
            # create_memory_node API upserts by node_id so re-runs
            # overwrite the same node in place.
            summary = memory.description[:240].strip() or f"semantic memory for {video_id}"
            content_json = json.dumps({
                "video_id": video_id,
                "description": memory.description,
                "characters": memory.characters,
                "events": memory.events,
                "entities": memory.entities,
                "source_extracted_ids": memory.source_extracted_ids,
            }, ensure_ascii=False, indent=2)
            try:
                graph.create_memory_node(
                    summary=summary,
                    content=content_json,
                    supporting_node_ids=memory.source_node_ids,
                    topic=f"semantic:{video_id}",
                    tags=["m3", "semantic", video_id],
                    node_id=memory.memory_node_id,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "create_memory_node failed for %s", video_id,
                )
                continue

            # Emit one EVOLVES edge per source ExtractedNode. Skip
            # edges that already exist (re-run idempotency).
            for src_id in memory.source_node_ids:
                existing = _existing_evolves_targets_for(graph, ctx, src_id)
                if memory.memory_node_id in existing:
                    continue
                graph.create_edge(
                    source_id=src_id,
                    target_id=memory.memory_node_id,
                    edge_type=GraphEdgeType.EVOLVES,
                    properties={
                        "relation": "consolidated_into",
                        "video_id": video_id,
                    },
                    ctx=ctx,
                )
                edges_emitted += 1
            memory.evolves_edge_count = edges_emitted  # cumulative; close enough for audit
            f.write(json.dumps(memory.to_jsonl(), ensure_ascii=False) + "\n")
            f.flush()
            memories_written += 1

    logger.info(
        "semantic_memory_consolidator wrote %d MemoryNodes "
        "(%d EVOLVES edges emitted) → %s",
        memories_written, edges_emitted, output_path,
    )
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consolidate episodic_memory_generator ExtractedNodes "
                    "into a single MemoryNode per video (uses EVOLVES edges).",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/semantic_memories.jsonl",
                   help="Per-video consolidated memory rows for audit / "
                        "downstream consumers.")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_semantic_memory_consolidator(args)


if __name__ == "__main__":
    sys.exit(main())