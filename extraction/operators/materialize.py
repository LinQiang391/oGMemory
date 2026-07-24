"""Materialization — turn an operator's output items into child graph nodes.

Many multimodal operators return a list of structured items alongside a
summary. ``video_frame_extractor`` returns a list of frames; ``face_detection``
returns a list of detected faces; ``object_detection`` returns a list of
detected objects. The natural graph representation is one child node per
item, connected to the source by a typed edge, plus a single parent
``ExtractedNode`` carrying the lightweight summary.

The child node is a ``RawNode`` when the item carries *raw* data the graph
should keep around for downstream operators (e.g. a sampled frame is raw
image bytes on disk; the bytes are not a derivation, they are exactly what
came off the video). It is an ``ExtractedNode`` when the item carries
*structured output* (bbox + embedding + detector metadata for a face
detection — the operator's actual product, not raw pixels). The two cases
share enough machinery that they live in one module, but the policy must
declare which kind it materializes so the writer / edge-type / schema
defaults are right.

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
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from core.graph_enums import GraphEdgeType, RawDataType

if TYPE_CHECKING:
    from core.models import RequestContext
    from service.graph_service import GraphService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

ChildNodeKind = Literal["raw", "extracted"]


@dataclass(frozen=True)
class MaterializationPolicy:
    """How an operator's ``extracted_content`` becomes child graph nodes.

    Operators that produce multiple discrete items declare a policy so
    :class:`service.operator_service.OperatorService` can persist each
    item as its own child node with an edge back to the source. The
    policy is fully declarative; the only per-call dynamic knob is
    ``bytes_writer``, which is supplied to :func:`materialize_children`
    (operators stay pure).

    Two flavours, distinguished by ``child_node_kind``:

    * ``"raw"`` — children are ``RawNode``s. Used when each item is
      essentially unprocessed input data (e.g. a sampled frame is raw
      image bytes). ``child_raw_type`` is required. Default edge type
      is ``REFERENCES`` (image references back to its source video).
    * ``"extracted"`` — children are ``ExtractedNode``s. Used when each
      item is structured output the operator produced (e.g. a face
      detection's bbox + embedding). ``child_extractor_type`` and
      ``child_extraction_schema`` are required. Default edge type is
      ``EXTRACTED_FROM`` (the detection was extracted from the source
      image).

    Field conventions:

    * ``collection_field`` is the key in ``extracted_content`` whose value
      is a list of items. Each item dict becomes one child node.
    * ``child_id_fn(item, source_raw_id, source_metadata)`` must be a
      pure function — the same inputs must always produce the same ID so
      re-runs are idempotent.
    * ``child_payload_fn(item, source_raw_id, source_metadata)`` is also
      pure. For ``"raw"`` kind it returns the ``RawNode.metadata`` dict;
      for ``"extracted"`` kind it returns the ``ExtractedNode
      .extracted_content`` dict.
    * If ``bytes_field`` is set, items are expected to carry base64-
      encoded bytes in that key. When the caller supplies ``bytes_writer``
      to :func:`materialize_children`, those bytes are decoded, handed
      to the writer, and the writer's return value is stored under
      ``bytes_metadata_key`` in the child's payload.
    """

    collection_field: str
    child_id_fn: Callable[[dict, str, dict | None], str]
    child_payload_fn: Callable[[dict, str, dict | None], dict]
    child_node_kind: ChildNodeKind = "raw"
    # Required when child_node_kind == "raw":
    child_raw_type: RawDataType | None = None
    # Required when child_node_kind == "extracted":
    child_extractor_type: str | None = None
    child_extraction_schema: str | None = None
    # Default edge_type depends on kind; callers can override.
    edge_type: GraphEdgeType | None = None
    edge_properties_fn: Callable[[dict, str, dict | None], dict] | None = None
    bytes_field: str | None = None
    bytes_metadata_key: str = "source_path"

    def __post_init__(self) -> None:
        if self.child_node_kind not in ("raw", "extracted"):
            raise ValueError(
                f"child_node_kind must be 'raw' or 'extracted', "
                f"got {self.child_node_kind!r}",
            )
        if self.child_node_kind == "raw" and self.child_raw_type is None:
            raise ValueError(
                "MaterializationPolicy with child_node_kind='raw' requires "
                "child_raw_type to be set.",
            )
        if self.child_node_kind == "extracted":
            missing = [
                k for k in ("child_extractor_type", "child_extraction_schema")
                if getattr(self, k) is None
            ]
            if missing:
                raise ValueError(
                    f"MaterializationPolicy with child_node_kind='extracted' "
                    f"requires {', '.join(missing)} to be set.",
                )

    def effective_edge_type(self) -> GraphEdgeType:
        """Edge type with kind-aware default if not explicitly set."""
        if self.edge_type is not None:
            return self.edge_type
        if self.child_node_kind == "raw":
            return GraphEdgeType.REFERENCES
        return GraphEdgeType.EXTRACTED_FROM


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MaterializedChild:
    """A single child node produced by :func:`materialize_children`."""

    child_node_id: str
    edge_id: str | None
    payload: dict[str, Any]
    # True when the child node already existed and was reused
    # (idempotency hit). ``edge_id`` is ``None`` in that case because we
    # don't look up an existing edge.
    resumed: bool = False
    # What kind of node was materialized, for downstream consumers that
    # need to know whether to query ``get_raw_node`` or
    # ``get_extracted_node``.
    child_node_kind: ChildNodeKind = "raw"


@dataclass
class MaterializationReport:
    """Outcome of materializing a collection field.

    Per-item failures are non-fatal by design, but callers need a structured
    signal so they do not have to scrape logs to notice dropped frames/faces.
    """

    collection_field: str
    attempted: int = 0
    created: int = 0
    resumed: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_field": self.collection_field,
            "attempted": self.attempted,
            "created": self.created,
            "resumed": self.resumed,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": list(self.errors),
        }

    def add_error(self, *, index: int, reason: str, error: str | None = None) -> None:
        item = {"index": index, "reason": reason}
        if error:
            item["error"] = error
        self.errors.append(item)
        self.skipped += 1


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
) -> tuple[list[MaterializedChild], MaterializationReport]:
    """Materialize ``extracted_content[policy.collection_field]`` into the graph.

    The function skips malformed per-item failures and records them in the
    returned :class:`MaterializationReport`; a single bad frame should not
    abort the entire clip, but callers get a programmatic dropped-item count.
    """

    items = extracted_content.get(policy.collection_field)
    if not isinstance(items, list) or not items:
        return [], MaterializationReport(collection_field=policy.collection_field)

    edge_type = policy.effective_edge_type()
    children: list[MaterializedChild] = []
    report = MaterializationReport(collection_field=policy.collection_field, attempted=len(items))
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning(
                "materialize_children: skipping non-dict item at %s[%d]: %r",
                policy.collection_field, idx, type(item).__name__,
            )
            report.add_error(index=idx, reason="non_dict_item", error=type(item).__name__)
            continue

        # 1. Decode + persist binary payload if configured. Shallow-copy
        #    so we never mutate the operator's extracted_content in place.
        item_for_payload = dict(item)
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
                    report.add_error(index=idx, reason="bytes_writer_failed", error=str(exc))
                    continue
                item_for_payload[policy.bytes_metadata_key] = written_path
                # Don't persist the heavy b64 in the child payload.
                item_for_payload.pop(policy.bytes_field, None)

        # 2 & 3. Compute IDs + payload. Child-id is the primary idempotency
        # key — same source + same item must yield the same id.
        try:
            child_id = policy.child_id_fn(item_for_payload, source_raw_id, source_metadata)
            payload = policy.child_payload_fn(item_for_payload, source_raw_id, source_metadata)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "materialize_children: policy function failed at %s[%d]: %s",
                policy.collection_field, idx, exc,
            )
            report.add_error(index=idx, reason="policy_failed", error=str(exc))
            continue

        # 4. Idempotency — look up via the appropriate kind-specific API.
        if skip_existing:
            if policy.child_node_kind == "raw":
                existing = graph.get_raw_node(child_id, ctx)
            else:  # "extracted"
                existing = graph.get_extracted_node(child_id, ctx)
            if existing is not None:
                children.append(MaterializedChild(
                    child_node_id=child_id,
                    edge_id=None,
                    payload=payload,
                    resumed=True,
                    child_node_kind=policy.child_node_kind,
                ))
                report.resumed += 1
                continue

        # 5. Create child node + edge.
        edge_id: str | None = None
        if policy.child_node_kind == "raw":
            graph.create_raw_node(
                raw_type=policy.child_raw_type,  # type: ignore[arg-type]
                content="",
                ctx=ctx,
                metadata=payload,
                node_id=child_id,
            )
            # RawNode has no auto-edge — create the explicit edge from
            # source to child.
            edge_props: dict[str, Any] = {}
            if policy.edge_properties_fn is not None:
                edge_props = policy.edge_properties_fn(
                    item_for_payload, source_raw_id, source_metadata,
                )
            edge = graph.create_edge(
                source_id=source_raw_id,
                target_id=child_id,
                edge_type=edge_type,
                ctx=ctx,
                weight=1.0,
                properties=edge_props,
            )
            edge_id = edge.edge_id
        else:  # "extracted"
            graph.create_extracted_node(
                source_raw_id=source_raw_id,
                extractor_type=policy.child_extractor_type,  # type: ignore[arg-type]
                extraction_schema=policy.child_extraction_schema,  # type: ignore[arg-type]
                ctx=ctx,
                extracted_content=payload,
                node_id=child_id,
            )
            # ExtractedNode creation auto-wires an EXTRACTED_FROM edge
            # from the child back to the source. We do NOT add a second
            # edge in the opposite direction — that would create a
            # duplicated pair (source->child + child->source) and
            # confuse downstream graph queries. edge_properties_fn is
            # therefore a no-op on this path.

        children.append(MaterializedChild(
            child_node_id=child_id,
            edge_id=edge_id,
            payload=payload,
            resumed=False,
            child_node_kind=policy.child_node_kind,
        ))
        report.created += 1

    return children, report


def lightweight_summary(
    extracted_content: dict,
    policy: MaterializationPolicy,
    child_ids: list[str],
) -> dict:
    """Build the lightweight ``extracted_content`` for the parent ``ExtractedNode``.

    Strips the heavy per-item collection (e.g. base64 JPEG bytes for
    every frame) and replaces it with ``child_node_ids`` plus a count,
    so the parent ``ExtractedNode`` stays small while every per-item
    payload remains reachable via the child nodes.

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


def _video_frame_payload(
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
    child_node_kind="raw",
    child_raw_type=RawDataType.IMAGE,
    collection_field="frames",
    child_id_fn=_video_frame_child_id,
    child_payload_fn=_video_frame_payload,
    edge_type=GraphEdgeType.REFERENCES,
    edge_properties_fn=_video_frame_edge_props,
    bytes_field="image_bytes_b64",
    bytes_metadata_key="source_path",
)


# ---------------------------------------------------------------------------
# face_detection -> FACE ExtractedNodes
# ---------------------------------------------------------------------------
#
# ``FaceDetectionExtractor.extract()`` returns ``extracted_content["faces"]``
# as a list of items shaped like::
#
#     {
#       "bbox": [x1, y1, x2, y2],            # normalised to [0, 1]
#       "confidence": 0.95,
#       "metadata": {                        # only present if backend supplied it
#         "detector": "insightface" | "haar",
#         "face_id": 0,                      # per-image, 0..N
#         "embedding": [..512 floats..],     # insightface only
#       },
#     }
#
# Each face is an *extracted* artefact — bbox + embedding are derived
# structure, not raw pixels — so ``child_node_kind="extracted"`` and the
# policy creates one ``ExtractedNode`` per face, linked to the source
# IMAGE frame via ``EXTRACTED_FROM``. Embeddings are 512 floats (~2 KB)
# — small enough to live directly in ``extracted_content`` rather than
# spill to disk.
#
# Identity note: ``face_id`` is per-image (the detector assigns 0..N
# within one call). Across calls (different frames) face_id 0 means a
# different physical face, so the deterministic child id is
# ``sha1("face/{source_frame_id}/{face_id}")`` which is unique per
# (frame, face). Cross-frame identity clustering is the job of a
# separate ``face_clustering`` operator that runs later.

def _face_child_id(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> str:
    face_meta = item.get("metadata") or {}
    if "face_id" in face_meta:
        face_id = face_meta["face_id"]
    elif "face_id" in item:
        face_id = item["face_id"]
    else:
        # Last-resort: hash the bbox so two faces with identical layout
        # still collide deterministically. Should not fire for any
        # production backend.
        face_id = hashlib.sha1(
            repr(tuple(round(v, 4) for v in item["bbox"])).encode(),
        ).hexdigest()[:8]
    seed = f"face/{source_raw_id}/{face_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _face_payload(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    """Build the face ``ExtractedNode.extracted_content`` payload."""
    face_meta = item.get("metadata") or {}
    payload: dict = {
        "bbox": [float(v) for v in item["bbox"]],
        "confidence": float(item["confidence"]),
    }
    # Flatten per-detection metadata the backend supplied so it lives
    # on the same node as the bbox/confidence it came with.
    for key in ("detector", "face_id", "embedding"):
        if key in face_meta:
            payload[key] = face_meta[key]
    # Provenance context for downstream operators (face clustering,
    # episodic memory) so they can reason without a graph traversal.
    payload["source_frame_node_id"] = source_raw_id
    if source_metadata is not None:
        for src_key in ("video_id", "clip_id", "frame_idx", "ts_sec"):
            if src_key in source_metadata:
                payload[f"source_{src_key}"] = source_metadata[src_key]
    return payload


def _face_edge_props(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    face_meta = item.get("metadata") or {}
    return {
        "relation": "detected_face",
        "face_id": face_meta.get("face_id"),
        "confidence": float(item["confidence"]),
    }


FACE_DETECTION_MATERIALIZATION = MaterializationPolicy(
    child_node_kind="extracted",
    child_extractor_type="face_detection",
    child_extraction_schema="face_detection_v1",
    collection_field="faces",
    child_id_fn=_face_child_id,
    child_payload_fn=_face_payload,
    # Default edge type for kind="extracted" is EXTRACTED_FROM, which
    # is the right semantic (the face was extracted from the source
    # IMAGE frame), so we leave edge_type unset.
    edge_properties_fn=_face_edge_props,
    # No bytes_field — embeddings are stored directly in extracted_content.
)


# ---------------------------------------------------------------------------
# audio_extractor -> AUDIO RawNode (one per VIDEO)
# ---------------------------------------------------------------------------
#
# ``AudioExtractor.extract()`` returns ``extracted_content["audio"]`` as a
# one-element list shaped like::
#
#     {
#       "format": "wav",
#       "sample_rate": 16000,
#       "channels": 1,
#       "duration_sec": 30.0,
#       "audio_bytes_b64": "<base64 WAV>",
#     }
#
# Each VIDEO clip yields one AUDIO child — the policy's
# ``collection_field`` returns a 1-element list (rather than bypassing
# materialization entirely) so the bytes-writer / REFERENCES-edge /
# idempotency / lightweight-summary machinery is reused uniformly with
# the VIDEO_FRAME policy. Future segmentation (utterance chunks) is a
# per-operator change; the policy doesn't have to move.
#
# Identity: child id is ``sha1("audio/{video_id}/{clip_id}")`` derived
# from the parent VIDEO's metadata. The same formula is used in
# ``tests/integration/m3/test_episodic_memory.py``'s audio fixture, so
# the two paths agree on id space.

def _audio_child_id(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> str:
    meta = source_metadata or {}
    video_id = str(meta.get("video_id", source_raw_id))
    clip_id = meta.get("clip_id", 0)
    seed = f"audio/{video_id}/{clip_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _audio_payload(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    """Build the AUDIO RawNode.metadata payload.

    Inherits video/clip provenance from the parent VIDEO metadata; the
    audio-format fields come from the per-item dict (``format``,
    ``sample_rate``, ``channels``, ``duration_sec``) and
    ``source_path`` is filled in by the materialization bytes-writer
    path.
    """
    meta = source_metadata or {}
    md: dict[str, Any] = {
        "format": item.get("format", "wav"),
        "sample_rate": int(item.get("sample_rate", 16000)),
        "channels": int(item.get("channels", 1)),
    }
    if "duration_sec" in item:
        md["duration_sec"] = float(item["duration_sec"])
    if "source_path" in item:
        md["source_path"] = item["source_path"]
    for src_key in ("video_id", "clip_id", "start_sec"):
        if src_key in meta:
            md[src_key] = meta[src_key]
    md["source_video_node_id"] = source_raw_id
    return md


def _audio_edge_props(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    return {
        "relation": "has_audio",
        "format": item.get("format", "wav"),
    }


AUDIO_FROM_VIDEO = MaterializationPolicy(
    child_node_kind="raw",
    child_raw_type=RawDataType.AUDIO,
    collection_field="audio",
    child_id_fn=_audio_child_id,
    child_payload_fn=_audio_payload,
    edge_type=GraphEdgeType.REFERENCES,
    edge_properties_fn=_audio_edge_props,
    bytes_field="audio_bytes_b64",
    bytes_metadata_key="source_path",
)


# ---------------------------------------------------------------------------
# Voice segmentation
# ---------------------------------------------------------------------------

def _voice_segment_child_id(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> str:
    """Stable child id from the per-segment voice_id when present,
    else a sha1 of (source, start, end). The voice_id is set by the
    operator and is already deterministic across runs.
    """
    voice_id = item.get("voice_id")
    if voice_id:
        return voice_id[:32]
    src = (source_metadata or {}).get("source_path", source_raw_id)
    seed = f"voice|{src}|{item.get('start_sec')}|{item.get('end_sec')}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _voice_segment_payload(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    payload: dict = {
        "start_sec": float(item.get("start_sec", 0.0)),
        "end_sec": float(item.get("end_sec", 0.0)),
        "duration_sec": float(item.get("duration_sec", 0.0)),
        "confidence": float(item.get("confidence", 0.0)),
        "voice_id": str(item.get("voice_id", "")),
        "rms_dbfs": item.get("rms_dbfs"),
        "frame_count": item.get("frame_count"),
    }
    # Carry the per-segment voice fingerprint into the graph so the
    # downstream voice_clustering pass can do cosine-distance
    # matching without re-reading the audio file. We only store
    # the embedding when present (the operator always emits one,
    # but tests / older manifests may not).
    emb = item.get("voice_embedding")
    if isinstance(emb, list) and emb:
        payload["voice_embedding"] = [float(x) for x in emb]
    # Provenance for cross-clip clustering: pull source_video_id /
    # source_clip_id off the audio RawNode's metadata so the
    # downstream voice_clustering pass can group segments by clip
    # without a graph traversal.
    if source_metadata is not None:
        for key in ("video_id", "clip_id"):
            if key in source_metadata:
                payload[f"source_{key}"] = source_metadata[key]
    payload["source_audio_node_id"] = source_raw_id
    return payload


def _voice_segment_edge_props(
    item: dict, source_raw_id: str, source_metadata: dict | None,
) -> dict:
    return {
        "relation": "voice_segment",
        "voice_id": str(item.get("voice_id", "")),
        "start_sec": float(item.get("start_sec", 0.0)),
        "end_sec": float(item.get("end_sec", 0.0)),
    }


VOICE_SEGMENTATION_MATERIALIZATION = MaterializationPolicy(
    child_node_kind="extracted",
    child_extractor_type="voice_segmentation",
    child_extraction_schema="voice_segmentation_v1",
    collection_field="segments",
    child_id_fn=_voice_segment_child_id,
    child_payload_fn=_voice_segment_payload,
    # Each voice segment is "extracted from" its source audio frame —
    # default edge type (EXTRACTED_FROM) matches the semantic.
    edge_properties_fn=_voice_segment_edge_props,
)


__all__ = [
    "MaterializationPolicy",
    "MaterializedChild",
    "materialize_children",
    "lightweight_summary",
    "VIDEO_FRAME_MATERIALIZATION",
    "FACE_DETECTION_MATERIALIZATION",
    "AUDIO_FROM_VIDEO",
    "VOICE_SEGMENTATION_MATERIALIZATION",
    "ChildNodeKind",
]
