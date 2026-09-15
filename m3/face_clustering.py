"""M3-Bench cross-clip face identity clustering.

Detected faces (from ``face_detection`` ExtractedNodes) carry per-face
embeddings. Across clips, the same person produces different faces
that should collapse to a single identity. This module implements a
simple single-linkage cosine clustering pass that walks the graph,
clusters faces whose embeddings are within ``threshold`` distance,
and emits ``same_identity`` edges between the cluster anchor and
each member.

Why this lives outside the ``extraction/operators/`` framework:

* Face clustering is a *cross-video aggregation*, not a per-node
  operation. The standard ``MultimodalExtractor.extract(raw_node,
  ctx)`` contract takes a single input node; face clustering needs
  to look at every face in the account at once.
* There's no fan-out — the output is a set of graph edges, not a
  set of child nodes — so the ``MaterializationPolicy`` machinery
  doesn't apply.

Algorithm — single-linkage connected-components clustering:

1. Pull every face ``ExtractedNode`` with a non-empty embedding
   from the graph (account-scoped).
2. For each pair within the same source video, skip (within-clip
   faces already have the same identity context — the operator
   only clusters across clips).
3. For each cross-clip pair with cosine distance < threshold, add
   an undirected edge to a Union-Find structure.
4. After all pairs are processed, find the connected components.
   For each component, pick the highest-confidence face as the
   anchor and emit a ``same_identity`` edge from every other member
   to the anchor. Weight on the edge = 1 - max_distance (the
   closest pair's similarity within the cluster).

Edge direction: source = member, target = anchor. The graph service
uses target → source via the existing ``create_edge`` API, but the
driver uses ``graph.create_edge(source_id=member, target_id=anchor,
edge_type=GraphEdgeType.SAME_IDENTITY, ...)`` which is the canonical
``newer → anchor`` convention.

Resumable: the driver inspects existing ``same_identity`` edges in
the graph and skips cluster pairs where both members are already
linked to a common anchor, so re-runs are idempotent.

Usage::

    python -m m3.face_clustering \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --threshold 0.4 \\
        --output-jsonl data/face_clusters.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.graph_enums import GraphEdgeType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure clustering algorithm (no I/O)
# ---------------------------------------------------------------------------

@dataclass
class FaceRecord:
    """One face with its embedding + provenance for cluster output."""

    node_id: str
    embedding: list[float]
    confidence: float
    bbox: list[float]
    video_id: str | None = None
    clip_id: int | None = None
    frame_idx: int | None = None


@dataclass
class ClusterMember:
    """One row of the cluster output JSONL."""

    face_node_id: str
    anchor_node_id: str
    video_id: str | None
    clip_id: int | None
    confidence: float
    bbox: list[float]
    cluster_size: int
    cluster_score: float

    def to_jsonl(self) -> dict:
        return asdict(self)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance (1 - cosine similarity) in [0, 2]. Handles
    zero-norm vectors by returning 1.0 (max distance).
    """
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / math.sqrt(na * nb)


class _UnionFind:
    """Tiny Union-Find with path compression + union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_faces(
    faces: list[FaceRecord],
    threshold: float = 0.4,
    same_clip_distance: float = 0.0,
) -> list[list[int]]:
    """Single-linkage clustering by cosine distance.

    Returns a list of clusters, where each cluster is a list of
    indices into ``faces``. Within-clip pairs are skipped when
    ``same_clip_distance >= 1.0`` (don't cluster within the same
    video/clip frame) — the M3 default is to cluster across clips
    only, so within-clip faces are never merged.
    """
    n = len(faces)
    if n == 0:
        return []
    uf = _UnionFind(n)
    # Cross-clip pairs only. Within-clip faces are kept in their
    # own singleton cluster even if their embeddings match — the
    # operator's job is to find cross-clip identity, and within-clip
    # co-occurrence is captured by ``co_occurs_in`` (not yet wired).
    for i in range(n):
        fi = faces[i]
        for j in range(i + 1, n):
            fj = faces[j]
            if (fi.video_id is not None and fj.video_id is not None
                    and fi.video_id == fj.video_id
                    and fi.clip_id is not None and fj.clip_id is not None
                    and fi.clip_id == fj.clip_id):
                # Within-clip; skip (we may add co_occurs_in separately).
                continue
            d = _cosine_distance(fi.embedding, fj.embedding)
            if d < threshold:
                uf.union(i, j)
    # Collect components.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    return [sorted(g) for g in groups.values()]


def build_cluster_output(
    clusters: list[list[int]],
    faces: list[FaceRecord],
) -> list[ClusterMember]:
    """Turn cluster index lists into per-face rows with an anchor.

    Each cluster's anchor is the face with the highest detection
    confidence (tie-break: first by index). Every member gets a
    row pointing at the anchor.
    """
    out: list[ClusterMember] = []
    for cluster in clusters:
        if not cluster:
            continue
        # Pick the anchor.
        anchor_i = max(
            cluster,
            key=lambda i: (faces[i].confidence, -i),  # higher conf wins
        )
        anchor = faces[anchor_i]
        size = len(cluster)
        # Cluster score: best (smallest) distance between any member
        # and the anchor. 1.0 means identical; 0 means far. For a
        # single-member cluster, score = 1.0 (the face is its own
        # identity anchor with no comparison).
        score = 1.0
        if size > 1:
            best_d = min(
                _cosine_distance(anchor.embedding, faces[j].embedding)
                for j in cluster if j != anchor_i
            )
            score = max(0.0, 1.0 - best_d)
        for idx in cluster:
            face = faces[idx]
            out.append(ClusterMember(
                face_node_id=face.node_id,
                anchor_node_id=anchor.node_id,
                video_id=face.video_id,
                clip_id=face.clip_id,
                confidence=face.confidence,
                bbox=face.bbox,
                cluster_size=size,
                cluster_score=round(score, 4),
            ))
    return out


# ---------------------------------------------------------------------------
# Graph-side helpers
# ---------------------------------------------------------------------------

def _face_records_from_graph(
    graph,
    ctx: RequestContext,
) -> list[FaceRecord]:
    """Pull every face_extraction ExtractedNode with a non-empty
    embedding, account-scoped. Returns ``FaceRecord``s in a stable
    order (sorted by node_id) so re-runs produce stable cluster ids.
    """
    out: list[FaceRecord] = []
    # Use the public API: list_extracted_nodes with extractor_type filter.
    nodes = graph.list_extracted_nodes("face_detection", ctx)
    for n in nodes:
        ec = n.extracted_content or {}
        emb = ec.get("embedding")
        if not emb or not isinstance(emb, list):
            continue
        try:
            emb_list = [float(x) for x in emb]
        except (TypeError, ValueError):
            continue
        if not emb_list or not all(math.isfinite(x) for x in emb_list) or not any(emb_list):
            continue
        out.append(FaceRecord(
            node_id=n.node_id,
            embedding=emb_list,
            confidence=float(ec.get("confidence", 0.0)),
            bbox=[float(x) for x in (ec.get("bbox") or [])],
            video_id=ec.get("source_video_id") or ec.get("_video_id"),
            clip_id=ec.get("source_clip_id")
                if ec.get("source_clip_id") is not None
                else ec.get("_clip_id"),
            frame_idx=ec.get("source_frame_idx"),
        ))
    out.sort(key=lambda f: f.node_id)
    return out


def _existing_same_identity_anchors(
    graph,
    ctx: RequestContext,
) -> dict[str, str]:
    """Build a map ``member_node_id → anchor_node_id`` from existing
    ``same_identity`` edges in the graph. Used to skip cluster pairs
    that are already linked, keeping re-runs idempotent.

    Iterating every edge in the graph is O(E) but E is small
    (only emitted by this driver) so the cost is negligible
    compared to the O(N²) cosine scan.
    """
    out: dict[str, str] = {}
    # Walk every ExtractedNode's same_identity neighbours. We use
    # the public ``list_extracted_nodes`` so the query is account-
    # scoped at the store layer.
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, edge, _dir in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            if edge.edge_type != GraphEdgeType.SAME_IDENTITY:
                continue
            # Convention: source = member, target = anchor. The
            # edge's source_id is the member.
            member = edge.source_id
            anchor = edge.target_id
            out[member] = anchor
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_face_clustering(
    args: argparse.Namespace,
    graph=None,
) -> int:
    """Walk the graph, cluster faces, emit same_identity edges."""
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
        session_id="m3bench-face-clustering",
        trace_id="m3-face-clustering",
    )

    faces = _face_records_from_graph(graph, ctx)
    logger.info("Loaded %d faces with embeddings", len(faces))

    # Always write the output JSONL, even if there's nothing to
    # cluster — downstream consumers (eval harness, audit) rely on
    # the file existing.
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(faces) < 2:
        logger.info("Need at least 2 faces to cluster; exiting")
        output_path.write_text("", encoding="utf-8")
        return 0

    threshold = float(args.threshold)
    started = time.monotonic()
    clusters = cluster_faces(faces, threshold=threshold)
    cluster_records = build_cluster_output(clusters, faces)
    logger.info(
        "Single-linkage cosine clustering (threshold=%.3f) → %d clusters "
        "in %.2fs",
        threshold, len({c.anchor_node_id for c in cluster_records}),
        time.monotonic() - started,
    )

    # Emit same_identity edges, skipping pairs already linked to a
    # common anchor.
    existing = _existing_same_identity_anchors(graph, ctx)
    emitted = 0
    skipped = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        # Re-pick the anchor (must match build_cluster_output).
        anchor_i = max(
            cluster,
            key=lambda i: (faces[i].confidence, -i),
        )
        anchor_id = faces[anchor_i].node_id
        for idx in cluster:
            if idx == anchor_i:
                continue
            member_id = faces[idx].node_id
            # If already linked to this anchor, skip. Linked to a
            # different anchor is also a no-op (the previous edge
            # takes precedence so re-runs are deterministic).
            if existing.get(member_id) == anchor_id:
                skipped += 1
                continue
            if member_id in existing:
                skipped += 1
                continue
            graph.create_edge(
                source_id=member_id,
                target_id=anchor_id,
                edge_type=GraphEdgeType.SAME_IDENTITY,
                properties={"score": round(cluster_records[0].cluster_score, 4)},
                ctx=ctx,
            )
            emitted += 1
    logger.info("Emitted %d same_identity edges (%d already linked)",
                emitted, skipped)

    # Write the per-row cluster JSONL for audit / downstream consumers.
    with output_path.open("w", encoding="utf-8") as f:
        for r in cluster_records:
            f.write(json.dumps(r.to_jsonl(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d cluster rows → %s", len(cluster_records), output_path)
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster face identities across clips and emit same_identity edges.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Cosine distance threshold (smaller = stricter). Default 0.4 "
                        "works for ArcFace / InsightFace embeddings; tune for other "
                        "backends.")
    p.add_argument("--output-jsonl", default="data/face_clusters.jsonl",
                   help="Per-face cluster rows for audit / downstream consumers.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_face_clustering(args)


if __name__ == "__main__":
    sys.exit(main())
