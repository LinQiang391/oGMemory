"""Shared traversal using only the store's account-scoped read primitives."""

from collections import deque
from typing import TYPE_CHECKING

from core.errors import NodeNotFoundError
from core.graph_enums import GraphEdgeType
from core.graph_models import ExtractedNode, MemoryNode, RawNode, SubgraphResult, SupportingChain

if TYPE_CHECKING:
    from core.interfaces import GraphStore
    from core.models import RequestContext


def traverse(
    store: "GraphStore", start_node_id: str, depth: int, ctx: "RequestContext",
    edge_types: list[GraphEdgeType] | None = None,
) -> SubgraphResult:
    start = store.get_node(start_node_id, ctx)
    if start is None:
        return SubgraphResult(nodes=[], edges=[])
    start_node_id = start.node_id
    nodes = {start_node_id: start}
    edges = {}
    queue = deque([(start_node_id, 0)])
    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for node, edge, _ in store.get_neighbors(
            current_id, ctx, edge_types=edge_types, direction="outgoing",
        ):
            edges[edge.edge_id] = edge
            if node.node_id not in nodes:
                nodes[node.node_id] = node
                queue.append((node.node_id, current_depth + 1))
    return SubgraphResult(nodes=list(nodes.values()), edges=list(edges.values()))


def get_supporting_chain(
    store: "GraphStore", memory_id: str, ctx: "RequestContext",
) -> SupportingChain:
    memory = store.get_node(memory_id, ctx)
    if not isinstance(memory, MemoryNode):
        raise NodeNotFoundError(memory_id)
    visited = {memory.node_id}
    raw_nodes, extracted_nodes, edges = {}, {}, {}
    queue = deque([(memory, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth > 2:
            continue
        # SUPPORTS points from evidence to memory, not the reverse.
        neighbors = store.get_neighbors(
            current.node_id, ctx, edge_types=[GraphEdgeType.SUPPORTS], direction="incoming",
        )
        if isinstance(current, ExtractedNode):
            neighbors += store.get_neighbors(
                current.node_id, ctx,
                edge_types=[GraphEdgeType.EXTRACTED_FROM], direction="outgoing",
            )
        for node, edge, _ in neighbors:
            edges[edge.edge_id] = edge
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            if isinstance(node, RawNode):
                raw_nodes[node.node_id] = node
            elif isinstance(node, ExtractedNode):
                extracted_nodes[node.node_id] = node
            queue.append((node, depth + 1))
    return SupportingChain(
        memory=memory, raw_nodes=list(raw_nodes.values()),
        extracted_nodes=list(extracted_nodes.values()), edges=list(edges.values()),
    )
