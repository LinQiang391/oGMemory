# Graph Relation Store Schema Design

## Overview

The GraphRelationStore is an in-memory graph database implementation for Phase 2 of ContextEngine. It provides efficient multi-hop relation traversal for hierarchical memory retrieval.

## Goals

1. **Interface Compatibility**: Must implement `RelationStore` Protocol from `core/interfaces.py`
2. **Multi-hop Traversal**: Support 1-hop and 2-hop relation queries
3. **Bidirectional Navigation**: Support both OUTBOUND and INBOUND edge traversal
4. **In-Memory Storage**: Use Python built-in types (no external dependencies for Phase 2)
5. **Path Scoring**: Return related URIs with accumulated path weights

## Data Structures

### Node Table

```python
# URI -> Node metadata
nodes: dict[str, NodeMetadata]

@dataclass
class NodeMetadata:
    uri: str
    exists: bool  # False for deleted nodes (edges kept for history)
    account_id: str  # For multi-tenant isolation
```

### Edge Table

```python
# Primary index: (from_uri, to_uri, relation_type) -> edge
edges_by_source: dict[str, dict[str, dict[str, RelationEdge]]]
# Structure: from_uri -> to_uri -> relation_type -> RelationEdge

# Reverse index for inbound traversal: (to_uri, from_uri, relation_type) -> edge
edges_by_target: dict[str, dict[str, dict[str, RelationEdge]]]
# Structure: to_uri -> from_uri -> relation_type -> RelationEdge
```

### RelationEdge (from core/models.py)

```python
@dataclass
class RelationEdge:
    from_uri: str
    to_uri: str
    relation_type: str  # e.g. "related_to" | "derived_from" | "contradicts" | "SEQUENCE"
    weight: float  # 0.0-1.0
    reason: str
```

## Relation Types

| Type | Direction | Description | Use Case |
|------|-----------|-------------|----------|
| `related_to` | bidirectional | General semantic relationship | Similar concepts |
| `derived_from` | outbound | Source derivation | Event -> Entity that caused it |
| `contradicts` | bidirectional | Contradictory information | Resolving conflicts |
| `SEQUENCE` | outbound | Temporal ordering | Events in same session |
| `contains` | outbound | Hierarchical containment | Topic -> Sub-topic |
| `references` | outbound | Citation/mention | Memory -> Source document |

## Traversal Algorithm

### 1-Hop Traversal (Depth=1)

```
Query: get_related_nodes(uri, direction=OUTBOUND, depth=1)

1. Look up uri in edges_by_source
2. Collect all outbound edges
3. Return list of (to_uri, weight, relation_type, reason)
```

### 2-Hop Traversal (Depth=2)

```
Query: get_related_nodes(uri, direction=OUTBOUND, depth=2)

1. Get 1-hop neighbors (as above)
2. For each 1-hop neighbor, get their outbound edges
3. Deduplicate destination URIs
4. Calculate path_score = min(weight1, weight2) or product
5. Return list sorted by path_score descending
```

### Path Scoring

For multi-hop paths, use **minimum weight** along path:

```
path_score = min(edge_1.weight, edge_2.weight, ..., edge_n.weight)
```

This ensures low-confidence relations don't get amplified through long paths.

## Access Control

All operations must respect `RequestContext.account_id`:

```python
def _is_accessible(self, uri: str, ctx: RequestContext) -> bool:
    """Check if URI belongs to the account in context."""
    if uri not in self._nodes:
        return False
    return self._nodes[uri].account_id == ctx.account_id
```

## API Extension

Beyond the base `RelationStore` Protocol, GraphRelationStore adds:

```python
def get_related_nodes(
    self,
    uri: str,
    ctx: RequestContext,
    direction: str = "OUTBOUND",  # OUTBOUND | INBOUND | BOTH
    depth: int = 1,  # 1 or 2
    relation_types: list[str] | None = None,  # Filter by relation type
    min_weight: float = 0.0,  # Filter by minimum weight
) -> list[RelatedNode]:
    """Get related nodes via graph traversal.

    Returns:
        List of RelatedNode with uri, score, path, relation_types
    """
    ...

@dataclass
class RelatedNode:
    uri: str
    score: float  # Path score (min weight along path)
    distance: int  # 1 for 1-hop, 2 for 2-hop
    path: list[RelationEdge]  # Full path from source to target
    relation_types: list[str]  # All relation types on path
```

## Migration Path

Phase 2: In-memory implementation (this document)
Phase 3+: Consider migration to dedicated graph database:
- **Neo4j**: Cypher queries, property graph model
- **ArangoDB**: Native graph + document hybrid
- **PostgreSQL WITH RECURSIVE**: Keep in relational DB

Interface remains unchanged, internal implementation swaps.

## Performance Considerations

### In-Memory Constraints
- All nodes/edges loaded in RAM
- Suitable for ~10K-100K nodes (single account)
- Multi-tenant: One GraphRelationStore instance per account or shared with filtering

### Optimization Strategies
1. **Lazy Loading**: Load relations on-demand from AGFSRelationStore
2. **LRU Cache**: Cache frequently accessed relation sets
3. **Bloom Filter**: Fast existence check before full traversal
4. **Index Hints**: Pre-compute popular 2-hop paths

### Concurrency
- Python `threading.Lock` for write operations
- Read operations lock-free (immutable during traversal)
- Copy-on-write for edge updates

## Testing Strategy

Unit tests must cover:
1. Single edge upsert and retrieval
2. Multiple edges from same source
3. Bidirectional navigation (OUTBOUND/INBOUND)
4. 2-hop traversal with path scoring
5. Multi-tenant isolation (account filtering)
6. Edge idempotency (re-upsert same edge)
7. Deleted node handling (edges remain)
