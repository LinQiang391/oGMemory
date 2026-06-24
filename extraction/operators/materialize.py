"""Materialization — turn an operator's output items into child ``RawNode``s.

Many multimodal operators return a list of structured items alongside a
summary. ``video_frame_extractor`` returns a list of frames; ``face_detection``
on a video clip will eventually return a list of detected faces;
``object_detection`` returns a list of detected objects. The natural graph
representation is one ``RawNode`` per item (``IMAGE``, ``FACE``, ``OBJECT``,
...) connected to the source ``VIDEO``/``IMAGE`` via a ``REFERENCES`` edge,
plus a single ``ExtractedNode`` carrying the lightweight summary.

This module is the contract between an operator's output shape and the
graph's node/edge layout:

* :class:`MaterializationPolicy` — what the operator declares about its
  output (declarative, lives on ``OperatorInfo.materialization``).
* :class:`MaterializedChild` — what was actually written.
* :func:`materialize_children` — the single generic implementation that
  ``OperatorService.run_operator`` calls when the operator declares a
  policy.
* :func:`lightweight_summary` — strips the heavy per-item payload
  (e.g. base64 JPEG bytes) from the ``ExtractedNode.extracted_content``
  after children are persisted, replacing it with ``child_node_ids``.

The policy is operator-static. The bytes-writing side effect (where on
disk to save JPEG bytes, etc.) is supplied per call by the driver via the
``bytes_writer`` argument — operators remain pure, the call site controls
filesystem layout.

Why this lives at the operator layer instead of in :mod:`m3`:
prior to this refactor, ``m3/extract_frames.py`` carried ~380 lines of
frame-decode / JPEG-write / RawNode creation logic that was identical to
what every future ``m3/detect_faces.py`` would have had to reimplement.
Moving the logic here means each operator's output-to-graph mapping is
declared once, in code that ships next to the operator, and is reused
across M3-Bench and any other workload that wants the same persistence.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.graph_enums import GraphEdgeType, RawDataType

if TYPE_CHECKING:
    from core.models import RequestContext
    from service.graph_service import GraphService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterializationPolicy:
    """How an operator's ``extracted_content`` becomes child ``RawNode``s.

    Operators that produce multiple discrete items declare a policy so
    :class:`service.operator_service.OperatorService` can persist each
    item as its own ``RawNode`` with a ``REFERENCES`` edge back to the
    source. The policy is fully declarative; the only per-call dynamic
    knob is ``bytes_writer``, which is supplied to
    :func:`materialize_children` (operators stay pure).

    Field conventions:

    * ``collection_field`` is the key in ``extracted_content`` whose value
      is a list of items. Each item dict becomes one child ``RawNode``.
    * ``child_id_fn(item, source_raw_id, source_metadata)`` must be a
      pure function — the same inputs must always produce the same ID so
      re-runs are idempotent.
    * ``child_metadata_fn`` is also pure and returns the ``RawNode.metadata``
      payload for the child.
    * If ``bytes_field`` is set, items are expected to carry base64-encoded
      bytes in that key. When the caller supplies ``bytes_writer`` to
      :func:`materialize_children`, those bytes are decoded, handed to the
      writer, and the writer's return value is stored under
      ``bytes_metadata_key`` in the child's metadata.
    """

    child_raw_type: RawDataType
    collection_field: str
    child_id_fn: Callable[[dict, str, dict | None], str]
    child_metadata_fn: Callable[[dict, str, dict | None], dict]
    edge_type: GraphEdgeType = GraphEdgeType.REFERENCES
    edge_properties_fn: Callable[[dict, str, dict | None], dict] | None = None
    bytes_field: str | None = None
    bytes_metadata_key: str = "source_path"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MaterializedChild:
    """A single child ``RawNode`` produced by :func:`materialize_children`."""

    child_node_id: str
    edge_id: str | None
    metadata: dict[str, Any]
    # True when the child ``RawNode`` already existed and was reused
    # (idempotency hit). ``edge_id`` is ``None`` in that case because we
    # don't look up an existing edge.
    resumed: bool = False


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def materialize_children(
    graph: GraphService,
    ctx: RequestContext,
    *,
    source_raw_id: str,
    source_metadata: dict | None,
    extracted_content: dict,
    policy: MaterializationPolicy,
    bytes_writer: Callable[[dict, bytes], str] | None = None,
    skip_existing: bool = True,
) -> list[MaterializedChild]:
    """Materialize ``extracted_content[policy.collection_field]`` into the graph.

    For each item in the collection:

    1. If ``policy.bytes_field`` and ``bytes_writer`` are both provided,
       decode the base64 payload, call ``bytes_writer(item, raw_bytes)``,
       and record the returned path under ``policy.bytes_metadata_key`` in
       the child's metadata. The original base64 field is dropped so the
       child metadata stays light.
    2. Compute ``child_node_id`` via ``policy.child_id_fn(item, source_raw_id, source_metadata)``.
    3. If ``skip_existing`` and the node already exists, record the hit
       and continue — no rewrite.
    4. Otherwise create the child ``RawNode`` and a
       ``policy.edge_type`` edge from ``source_raw_id`` to it.

    The function never raises on a malformed item; it skips and logs.
    That keeps a single bad frame from aborting the entire clip's
    materialization, which would be much harder to recover from.
    """

    items = extracted_content.get(policy.collection_field)
    if not isinstance(items, list) or not items:
        return []

    children: list[MaterializedChild] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning(
                "materialize_children: skipping non-dict item at %s[%d]: %r",
                policy.collection_field, idx, type(item).__name__,
            )
            continue

        # 1. Decode + persist binary payload if configured. Shallow-copy
        #    so we never mutate the operator's extracted_content in place.
        item_for_meta = dict(item)
        if policy.bytes_field and bytes_writer is not None:
            b64 = item.get(policy.bytes_field)
            if isinstance(b64, str):
                try:
                    raw_bytes = base64.b64decode(b64)
                    written_path = bytes_writer(item, raw_bytes)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "materialize_children: bytes_writer failed for %s[%d]: %s",
                        policy.collection_field, idx, exc,
                    )
                    continue
                item_for_meta[policy.bytes_metadata_key] = written_path
                # Don't persist the heavy b64 in the child metadata.
                item_for_meta.pop(policy.bytes_field, None)

        # 2. Compute IDs + metadata. Child-id is the primary idempotency
        #    key — same source + same item must yield the same id.
        try:
            child_id = policy.child_id_fn(item_for_meta, source_raw_id, source_metadata)
            metadata = policy.child_metadata_fn(item_for_meta, source_raw_id, source_metadata)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "materialize_children: policy function failed at %s[%d]: %s",
                policy.collection_field, idx, exc,
            )
            continue

        # 3. Idempotency.
        if skip_existing:
            existing = graph.get_raw_node(child_id, ctx)
            if existing is not None:
                children.append(MaterializedChild(
                    child_node_id=child_id,
                    edge_id=None,
                    metadata=metadata,
                    resumed=True,
                ))
                continue

        # 4. Create child RawNode + edge.
        graph.create_raw_node(
            raw_type=policy.child_raw_type,
            content="",
            ctx=ctx,
            metadata=metadata,
            node_id=child_id,
        )
        edge_props: dict[str, Any] = {}
        if policy.edge_properties_fn is not None:
            edge_props = policy.edge_properties_fn(item_for_meta, source_raw_id, source_metadata)
        edge = graph.create_edge(
            source_id=source_raw_id,
            target_id=child_id,
            edge_type=policy.edge_type,
            ctx=ctx,
            weight=1.0,
            properties=edge_props,
        )
        children.append(MaterializedChild(
            child_node_id=child_id,
            edge_id=edge.edge_id,
            metadata=metadata,
            resumed=False,
        ))

    return children


def lightweight_summary(
    extracted_content: dict,
    policy: MaterializationPolicy,
    child_ids: list[str],
) -> dict:
    """Build the lightweight ``extracted_content`` for the parent ``ExtractedNode``.

    Strips the heavy per-item collection (e.g. base64 JPEG bytes for
    every frame) and replaces it with ``child_node_ids`` plus a count,
    so the parent ``ExtractedNode`` stays small while every per-item
    payload remains reachable via the child ``RawNode``s.

    Top-level keys other than ``policy.collection_field`` are passed
    through unchanged (e.g. ``fps_actual``, ``keyframe_count``,
    ``duration_sec`` for video frames).
    """
    summary: dict[str, Any] = {
        k: v for k, v in extracted_content.items()
        if k != policy.collection_field
    }
    summary["child_node_ids"] = list(child_ids)
    summary["child_count"] = len(child_ids)
    return summary


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

def _video_frame_child_id(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> str:
    """Deterministic IMAGE RawNode ID for a single sampled frame.

    Seed is ``"frame/{source_raw_id}/{frame_idx}"`` — ``source_raw_id``
    already uniquely identifies the VIDEO clip (it's a sha1 of
    ``clip/{video_id}/{clip_id}``), so adding ``frame_idx`` is enough to
    disambiguate every frame in the dataset.
    """
    seed = f"frame/{source_raw_id}/{item['frame_idx']}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _video_frame_metadata(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    """Build the IMAGE RawNode.metadata payload for a single frame."""
    md = {
        "frame_idx": item["frame_idx"],
        "ts_sec": item["ts_sec"],
        "keyframe": item["keyframe"],
        "width": item["width"],
        "height": item["height"],
        "source_video_node_id": source_raw_id,
    }
    if "source_path" in item:
        md["source_path"] = item["source_path"]
    return md


def _video_frame_edge_props(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    return {
        "relation": "has_frame",
        "frame_idx": item["frame_idx"],
        "ts_sec": item.get("ts_sec"),
    }


VIDEO_FRAME_MATERIALIZATION = MaterializationPolicy(
    child_raw_type=RawDataType.IMAGE,
    collection_field="frames",
    child_id_fn=_video_frame_child_id,
    child_metadata_fn=_video_frame_metadata,
    edge_type=GraphEdgeType.REFERENCES,
    edge_properties_fn=_video_frame_edge_props,
    bytes_field="image_bytes_b64",
    bytes_metadata_key="source_path",
)


__all__ = [
    "MaterializationPolicy",
    "MaterializedChild",
    "materialize_children",
    "lightweight_summary",
    "VIDEO_FRAME_MATERIALIZATION",
]
