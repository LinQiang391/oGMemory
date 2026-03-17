"""Hierarchical Searcher - expands retrieval results using relation edges.

Phase 2 enhancement: Uses graph relations to find related memories beyond
simple vector similarity. Enables knowledge graph traversal for context-aware
retrieval.

See CLAUDE.md §7 for retrieval chain rules.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.interfaces import RelationStore
from core.models import RequestContext, SeedHit


@dataclass
class ExpandedHit:
    """A retrieval result expanded with relation information.

    Combines vector similarity score with graph relation scores.
    """
    uri: str
    base_score: float  # Original vector similarity score
    relation_score: float  # Combined relation path score
    final_score: float  # Combined final score
    distance: int  # 0 = original hit, 1 = 1-hop, 2 = 2-hop
    abstract: str
    has_overview: bool
    has_content: bool
    relation_path: list[str] = field(default_factory=list)  # relation_types along path
    source_uri: Optional[str] = None  # Original hit URI that led to this expansion


class HierarchicalSearcher:
    """Expands retrieval results using graph relations.

    Takes L0 SeedHit results and expands them with related nodes
    from the RelationStore. Enables finding contextually related
    memories that may not have high vector similarity.

    Scoring strategy:
    - Original hits: final_score = base_score (unchanged)
    - 1-hop expansions: final_score = base_score * relation_weight * 0.8
    - 2-hop expansions: final_score = base_score * relation_weight * 0.6

    The decay factors (0.8, 0.6) ensure vector similarity remains primary.
    """

    def __init__(self, relation_store: RelationStore):
        """Initialize HierarchicalSearcher with a RelationStore.

        Args:
            relation_store: RelationStore implementation (GraphRelationStore or AGFSRelationStore)
        """
        self._relation_store = relation_store

    def expand(
        self,
        seed_hits: list[SeedHit],
        ctx: RequestContext,
        depth: int = 1,
        min_relation_weight: float = 0.3,
        max_expansions: int = 50,
        direction: str = "OUTBOUND",
    ) -> list[ExpandedHit]:
        """Expand seed hits with related nodes.

        Args:
            seed_hits: Original L0 retrieval results
            ctx: RequestContext for access control
            depth: Graph traversal depth (1 or 2)
            min_relation_weight: Minimum relation weight to include
            max_expansions: Maximum number of expanded results
            direction: OUTBOUND | INBOUND | BOTH

        Returns:
            List of ExpandedHit with original and expanded results.
            Sorted by final_score descending.
        """
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")

        if direction not in ("OUTBOUND", "INBOUND", "BOTH"):
            raise ValueError("direction must be OUTBOUND, INBOUND, or BOTH")

        results: dict[str, ExpandedHit] = {}

        # Process each seed hit
        for hit in seed_hits:
            # Add original hit
            results[hit.uri] = ExpandedHit(
                uri=hit.uri,
                base_score=hit.score,
                relation_score=1.0,  # No relation decay for original hits
                final_score=hit.score,
                distance=0,
                abstract=hit.abstract,
                has_overview=hit.has_overview,
                has_content=hit.has_content,
            )

            # Expand with related nodes
            try:
                from providers.relation_store.graph_relation_store import GraphRelationStore

                if isinstance(self._relation_store, GraphRelationStore):
                    related = self._relation_store.get_related_nodes(
                        uri=hit.uri,
                        ctx=ctx,
                        direction=direction,
                        depth=depth,
                        min_weight=min_relation_weight,
                    )

                    for rel_node in related:
                        # Skip if already in results (prefer original hit)
                        if rel_node.uri in results:
                            continue

                        # Calculate decay factor based on distance
                        decay_factor = 0.8 if rel_node.distance == 1 else 0.6

                        # Calculate final score
                        final_score = hit.score * rel_node.score * decay_factor

                        # Create expanded hit
                        results[rel_node.uri] = ExpandedHit(
                            uri=rel_node.uri,
                            base_score=hit.score,  # Inherit base score from source
                            relation_score=rel_node.score,
                            final_score=final_score,
                            distance=rel_node.distance,
                            abstract="",  # Abstract not available from relation store
                            has_overview=False,
                            has_content=False,
                            relation_path=rel_node.relation_types,
                            source_uri=hit.uri,
                        )

            except ImportError:
                # GraphRelationStore not available, skip expansion
                pass

        # Convert to list and sort by final_score
        expanded_results = list(results.values())
        expanded_results.sort(key=lambda h: h.final_score, reverse=True)

        # Limit results
        return expanded_results[:max_expansions]

    def expand_with_content(
        self,
        seed_hits: list[SeedHit],
        ctx: RequestContext,
        get_abstract_fn,  # Function to fetch abstract by URI
        depth: int = 1,
        min_relation_weight: float = 0.3,
        max_expansions: int = 50,
        direction: str = "OUTBOUND",
    ) -> list[ExpandedHit]:
        """Expand seed hits and fetch abstracts for related nodes.

        Args:
            seed_hits: Original L0 retrieval results
            ctx: RequestContext for access control
            get_abstract_fn: Function(uri) -> str to fetch abstract content
            depth: Graph traversal depth (1 or 2)
            min_relation_weight: Minimum relation weight to include
            max_expansions: Maximum number of expanded results
            direction: OUTBOUND | INBOUND | BOTH

        Returns:
            List of ExpandedHit with abstracts populated for all results.
        """
        # Get basic expansion
        expanded = self.expand(
            seed_hits=seed_hits,
            ctx=ctx,
            depth=depth,
            min_relation_weight=min_relation_weight,
            max_expansions=max_expansions,
            direction=direction,
        )

        # Fetch abstracts for expanded nodes (original hits already have abstracts)
        for hit in expanded:
            if hit.distance > 0 and not hit.abstract:
                try:
                    hit.abstract = get_abstract_fn(hit.uri)
                    # Try to detect if overview/content exists
                    # This is a heuristic; full check would require reading the node
                    hit.has_overview = len(hit.abstract) > 100
                    hit.has_content = len(hit.abstract) > 200
                except Exception:
                    # Abstract fetch failed, keep empty
                    pass

        return expanded

    def get_relation_path_summary(self, hit: ExpandedHit) -> str:
        """Get human-readable summary of relation path.

        Args:
            hit: ExpandedHit with relation_path

        Returns:
            String describing the relation path, e.g. "related_to → derived_from"
        """
        if not hit.relation_path:
            return ""

        return " → ".join(hit.relation_path)

    def group_by_distance(self, expanded_hits: list[ExpandedHit]) -> dict[int, list[ExpandedHit]]:
        """Group expanded hits by distance from original query.

        Args:
            expanded_hits: List of ExpandedHit results

        Returns:
            Dict mapping distance -> list of hits at that distance
        """
        groups: dict[int, list[ExpandedHit]] = {0: [], 1: [], 2: []}

        for hit in expanded_hits:
            if hit.distance in groups:
                groups[hit.distance].append(hit)

        return groups
