"""Conservative audio extraction cache receipts, scoped to one audio source.

Only a fully materialized result earns a receipt. Legacy children, missing
files, partial writes and unknown configurations are never cache hits.
Sequential execution only: node writes and cleanup are not transactional.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import UUID

FIELDS = {
    "voice_asr": (
        "backend",
        "model",
        "language",
        "beam_size",
        "no_word_timestamps",
        "no_vad_filter",
        "base_url",
        "device",
    ),
    "voice_segmentation": (
        "energy_threshold_dbfs",
        "frame_sec",
        "min_segment_sec",
        "max_segment_sec",
        "pad_sec",
    ),
    "speaker_diarization": ("backend", "model", "min_speakers", "max_speakers", "device"),
}
RECEIPT = "_audio_cache_v1"


def _id(value):
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def signature(audio, args, kind, account_id, *, dependencies=None):
    """Hash audio bytes and inference configuration, never credentials."""
    try:
        path = Path(audio.metadata["source_path"])
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        params = {key: getattr(args, key) for key in FIELDS[kind]}
        if kind == "voice_asr":
            if args.backend == "openai_http":
                params["base_url"] = (
                    args.base_url or os.environ.get("OGMEM_ASR_BASE_URL", "")
                ).rstrip("/")
                params["model"] = args.model or "whisper-1"
            elif args.backend == "faster_whisper":
                params["model"] = args.model or "large-v3-turbo"
        if kind == "speaker_diarization" and args.backend == "pyannote":
            params["model"] = args.model or "pyannote/speaker-diarization-3.1"
        return _digest(
            {
                "version": 1,
                "kind": kind,
                "account": account_id,
                "source": _id(audio.node_id),
                "metadata": audio.metadata,
                "audio_sha256": digest.hexdigest(),
                "params": params,
                **({"dependencies": dependencies} if dependencies is not None else {}),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None


def _payload_digest(node, kind):
    content = dict(node.extracted_content or {})
    if kind == "voice_asr":
        # Diarization stamps this downstream; it is not ASR output.
        content.pop("speaker_id", None)
    return _digest(content)


class AudioCache:
    def __init__(self, graph, ctx, kind):
        self.graph, self.ctx, self.kind = graph, ctx, kind
        self.by_source = {}
        for node in graph.list_extracted_nodes(kind, ctx, limit=None):
            self.by_source.setdefault(_id(node.source_raw_id), []).append(node)

    def lookup(self, audio, fingerprint, *, writeback=None):
        """Return ordered children or None; optionally verify downstream writes."""
        if fingerprint is None:
            return None
        nodes = self.by_source.get(_id(audio.node_id), [])
        by_id = {_id(n.node_id): n for n in nodes}
        for parent in nodes:
            ec = parent.extracted_content or {}
            receipt = ec.get(RECEIPT)
            if (
                parent.extraction_status != "completed"
                or not isinstance(receipt, dict)
                or receipt.get("signature") != fingerprint
                or (writeback is not None and receipt.get("writeback") != writeback)
            ):
                continue
            ids = ec.get("child_node_ids")
            hashes = receipt.get("children")
            if not isinstance(ids, list) or not isinstance(hashes, dict):
                continue
            if not all(isinstance(i, str) for i in ids):
                continue
            ids = [_id(i) for i in ids]
            if (
                len(ids) != ec.get("child_count")
                or len(set(ids)) != len(ids)
                or set(ids) != set(hashes)
            ):
                continue
            children = [by_id.get(i) for i in ids]
            if all(
                c is not None
                and c.extraction_status == "completed"
                and _payload_digest(c, self.kind) == hashes[_id(c.node_id)]
                for c in children
            ):
                return children
        return None

    def write_summary(self, output_path, *, video, attempted, cached, executed, failed):
        summary = {
            "status": "failed" if failed else "completed",
            "account_id": self.ctx.account_id,
            "video_filter": video,
            "attempted_audio": attempted,
            "cached_audio": cached,
            "executed_audio": executed,
            "failed_audio": failed,
            "completed_audio": attempted - failed,
            "cache_version": 1,
        }
        Path(output_path).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def invalidate(self, audio):
        # Invalidate before inference: an interrupted forced refresh must not
        # allow an old receipt to certify partially updated children on retry.
        for node in self.by_source.get(_id(audio.node_id), []):
            ec = dict(node.extracted_content or {})
            if RECEIPT in ec:
                ec.pop(RECEIPT)
                self.graph.mark_extraction_completed(
                    node.node_id,
                    extracted_content=ec,
                    confidence=node.confidence,
                    ctx=self.ctx,
                )

    def finish(self, audio, fingerprint, result, expected_count, *, writeback=None):
        """Validate persistence, prune obsolete outputs, then certify."""
        if result.get("error"):
            return
        materialized = result.get("materialized_children") or []
        report = result.get("materialization") or {}
        if len(materialized) != expected_count or report.get("failed"):
            raise RuntimeError("Incomplete audio materialization; cache not certified")
        parent_id = result.get("extracted_node_id")
        if not isinstance(parent_id, str):
            raise RuntimeError("Audio persistence did not return a parent ID")
        ids = [m["child_node_id"] for m in materialized]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate audio child IDs; cache not certified")
        children = [self.graph.get_extracted_node(i, self.ctx) for i in ids]
        if any(c is None or _id(c.source_raw_id) != _id(audio.node_id) for c in children):
            raise RuntimeError("Audio children missing after persistence")
        # SQL canonicalizes UUIDs; use stored IDs in both fresh and cached
        # manifests (memory stores may preserve undashed IDs).
        for row, child in zip(materialized, children):
            row["child_node_id"] = child.node_id
        keep = {_id(i) for i in [*ids, parent_id]}
        for old in self.by_source.get(_id(audio.node_id), []):
            if _id(old.node_id) not in keep:
                self.graph.delete_node(old.node_id, self.ctx)
        parent = self.graph.get_extracted_node(parent_id, self.ctx)
        if parent is None:
            raise RuntimeError("Audio parent missing after persistence")
        ec = dict(parent.extracted_content or {})
        if [_id(i) for i in ec.get("child_node_ids", [])] != [_id(i) for i in ids] or ec.get(
            "child_count"
        ) != len(ids):
            raise RuntimeError("Audio parent/children mismatch")
        if fingerprint is not None:
            ec[RECEIPT] = {
                "signature": fingerprint,
                "children": {_id(c.node_id): _payload_digest(c, self.kind) for c in children},
                **({"writeback": writeback} if writeback is not None else {}),
            }
            self.graph.mark_extraction_completed(
                parent_id,
                extracted_content=ec,
                confidence=parent.confidence,
                ctx=self.ctx,
            )
