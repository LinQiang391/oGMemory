"""Extended tests for InMemoryVectorIndex delete operations."""


from core.models import IndexRecord
from providers.vector_index.in_memory_index import InMemoryVectorIndex


class TestInMemoryDeleteOperations:
    """Tests for delete_account_data and delete_by_owner_space methods."""

    def test_delete_account_data_removes_all_account_records(self):
        """Test deleting all records for an account."""
        index = InMemoryVectorIndex()

        # Add records for different accounts
        records = [
            IndexRecord(
                id="1", uri="ctx://acme/users/u1/memories/profile",
                level=0, text="abstract", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"category": "profile"}
            ),
            IndexRecord(
                id="2", uri="ctx://acme/users/u1/memories/preferences/p1",
                level=0, text="pref1", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"category": "preference"}
            ),
            IndexRecord(
                id="3", uri="ctx://other/users/u2/memories/profile",
                level=0, text="other", filters={"account_id": "other", "owner_space": "user:u2"},
                metadata={"category": "profile"}
            ),
        ]
        index.upsert(records)

        # Delete all acme account data
        deleted = index.delete_account_data("acme")

        assert deleted == 2
        assert index.count() == 1
        assert index.get_record("1") is None
        assert index.get_record("2") is None
        assert index.get_record("3") is not None

    def test_delete_account_data_returns_count(self):
        """Test delete_account_data returns correct count."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id=f"{i}", uri=f"ctx://acme/users/u1/memories/p{i}",
                level=0, text=f"text{i}", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={}
            )
            for i in range(5)
        ]
        index.upsert(records)

        assert index.delete_account_data("acme") == 5
        assert index.delete_account_data("nonexistent") == 0

    def test_delete_by_owner_space_removes_matching_records(self):
        """Test deleting by account_id + owner_space."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id="1", uri="ctx://acme/users/u1/memories/profile",
                level=0, text="u1 profile", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"category": "profile"}
            ),
            IndexRecord(
                id="2", uri="ctx://acme/users/u2/memories/profile",
                level=0, text="u2 profile", filters={"account_id": "acme", "owner_space": "user:u2"},
                metadata={"category": "profile"}
            ),
            IndexRecord(
                id="3", uri="ctx://acme/agents/a1/skills/generated/s1",
                level=0, text="agent skill", filters={"account_id": "acme", "owner_space": "agent:a1"},
                metadata={"category": "skill"}
            ),
        ]
        index.upsert(records)

        # Delete only user:u1 records
        deleted = index.delete_by_owner_space("acme", "user:u1")

        assert deleted == 1
        assert index.count() == 2
        assert index.get_record("1") is None
        assert index.get_record("2") is not None
        assert index.get_record("3") is not None

    def test_delete_by_owner_space_scoped_to_account(self):
        """Test delete_by_owner_space requires both account_id and owner_space match."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id="1", uri="ctx://acme/users/u1/memories/profile",
                level=0, text="acme u1", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={}
            ),
            IndexRecord(
                id="2", uri="ctx://other/users/u1/memories/profile",
                level=0, text="other u1", filters={"account_id": "other", "owner_space": "user:u1"},
                metadata={}
            ),
        ]
        index.upsert(records)

        # Only delete acme's user:u1, not other's user:u1
        deleted = index.delete_by_owner_space("acme", "user:u1")

        assert deleted == 1
        assert index.get_record("1") is None
        assert index.get_record("2") is not None


class TestInMemorySearchByVector:
    """Tests for search_by_vector method."""

    def test_search_by_vector_filters_by_account(self):
        """Test search_by_vector enforces account_id filter."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id="1", uri="ctx://acme/users/u1/memories/profile",
                level=0, text="profile", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"category": "profile"}
            ),
            IndexRecord(
                id="2", uri="ctx://other/users/u2/memories/profile",
                level=0, text="profile", filters={"account_id": "other", "owner_space": "user:u2"},
                metadata={"category": "profile"}
            ),
        ]
        index.upsert(records)

        query_vector = index._mock_vector("profile")
        results = index.search_by_vector(
            query_vector=query_vector,
            filters={"account_id": "acme"},
            top_k=10
        )

        assert len(results) == 1
        assert results[0].uri == "ctx://acme/users/u1/memories/profile"

    def test_search_by_vector_returns_top_k(self):
        """Test search_by_vector respects top_k limit."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id=str(i), uri=f"ctx://acme/users/u1/memories/p{i}",
                level=0, text=f"text{i}", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={}
            )
            for i in range(10)
        ]
        index.upsert(records)

        query_vector = index._mock_vector("text")
        results = index.search_by_vector(
            query_vector=query_vector,
            filters={"account_id": "acme"},
            top_k=5
        )

        assert len(results) == 5

    def test_search_by_vector_preserves_metadata(self):
        """Test search_by_vector carries record metadata into SeedHit."""
        index = InMemoryVectorIndex()
        index.upsert([
            IndexRecord(
                id="1",
                uri="ctx://acme/users/u1/memories/events/pride",
                level=2,
                text="Caroline attended pride.",
                filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={
                    "context_type": "MEMORY",
                    "category": "event",
                    "when": "2023-06-26",
                    "who": ["Caroline"],
                },
            )
        ])

        results = index.search_by_vector(
            query_vector=index._mock_vector("Caroline pride"),
            filters={"account_id": "acme"},
            top_k=1,
        )

        assert results[0].metadata["when"] == "2023-06-26"
        assert results[0].metadata["who"] == ["Caroline"]

    def test_search_children_filters_by_parent_uri(self):
        """Test search_children only returns direct children of parent_uri."""
        index = InMemoryVectorIndex()

        records = [
            IndexRecord(
                id="1", uri="ctx://acme/users/u1/memories/preferences/p1",
                level=0, text="pref1", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"parent_uri": "ctx://acme/users/u1/memories/preferences/", "category": "preference"}
            ),
            IndexRecord(
                id="2", uri="ctx://acme/users/u1/memories/preferences/p2",
                level=0, text="pref2", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"parent_uri": "ctx://acme/users/u1/memories/preferences/", "category": "preference"}
            ),
            IndexRecord(
                id="3", uri="ctx://acme/users/u1/memories/entities/e1",
                level=0, text="entity", filters={"account_id": "acme", "owner_space": "user:u1"},
                metadata={"parent_uri": "ctx://acme/users/u1/memories/entities/", "category": "entity"}
            ),
        ]
        index.upsert(records)

        query_vector = index._mock_vector("pref")
        results = index.search_children(
            parent_uri="ctx://acme/users/u1/memories/preferences/",
            query_vector=query_vector,
            filters={"account_id": "acme"},
            top_k=10
        )

        assert len(results) == 2
        uris = {r.uri for r in results}
        assert uris == {
            "ctx://acme/users/u1/memories/preferences/p1",
            "ctx://acme/users/u1/memories/preferences/p2",
        }

    def test_count_supports_tenant_filters(self):
        index = InMemoryVectorIndex()
        index.upsert([
            IndexRecord(id="1", uri="ctx://a/1", level=0, text="one",
                        filters={"account_id": "a", "owner_space": "user:u1"}, metadata={}),
            IndexRecord(id="2", uri="ctx://a/2", level=0, text="two",
                        filters={"account_id": "a", "owner_space": "user:u2"}, metadata={}),
            IndexRecord(id="3", uri="ctx://b/1", level=0, text="three",
                        filters={"account_id": "b", "owner_space": "user:u1"}, metadata={}),
        ])

        assert index.count({"account_id": "a"}) == 2
        assert index.count({"account_id": "a", "owner_space": ["user:u2"]}) == 1
