"""Contract test for IndexRecord.filters — HIGHEST PRIORITY.

Phase 0 frozen — CRITICAL invariant: IndexRecord.filters MUST contain
account_id + owner_space. Missing either field causes silent cross-tenant
data leakage (no runtime error, just wrong results).

This contract test MUST pass before any VectorIndex implementation is used.
"""

import pytest

from core.models import IndexRecord
from core.errors import ValidationError


class TestIndexRecordFiltersMandatory:
    """Verify IndexRecord.filters contains account_id and owner_space.

    This is the MOST IMPORTANT contract test in Phase 0.
    Missing filters cause silent cross-tenant leakage.
    """

    def test_index_record_filters_must_contain_account_id(self):
        """IndexRecord.filters MUST contain 'account_id' key."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/users/alice/memories/profile",
            level=0,
            text="Abstract text",
            filters={"account_id": "acme", "owner_space": "user_space_alice"},
            metadata={},
        )
        assert "account_id" in record.filters
        assert record.filters["account_id"] == "acme"

    def test_index_record_filters_must_contain_owner_space(self):
        """IndexRecord.filters MUST contain 'owner_space' key."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/users/alice/memories/profile",
            level=0,
            text="Abstract text",
            filters={"account_id": "acme", "owner_space": "user_space_alice"},
            metadata={},
        )
        assert "owner_space" in record.filters
        assert record.filters["owner_space"] == "user_space_alice"

    def test_index_record_filters_missing_account_id_raises_error(self):
        """IndexRecord without account_id in filters should raise ValidationError.

        This enforces the multi-tenant isolation invariant.
        """
        with pytest.raises(ValidationError) as exc_info:
            self._validate_index_record(
                IndexRecord(
                    id="abc123",
                    uri="ctx://acme/users/alice/memories/profile",
                    level=0,
                    text="Abstract",
                    # Missing account_id!
                    filters={"owner_space": "user_space_alice"},
                    metadata={},
                )
            )
        assert "account_id" in str(exc_info.value).lower()

    def test_index_record_filters_missing_owner_space_raises_error(self):
        """IndexRecord without owner_space in filters should raise ValidationError.

        This enforces the user/agent namespace isolation invariant.
        """
        with pytest.raises(ValidationError) as exc_info:
            self._validate_index_record(
                IndexRecord(
                    id="abc123",
                    uri="ctx://acme/users/alice/memories/profile",
                    level=0,
                    text="Abstract",
                    # Missing owner_space!
                    filters={"account_id": "acme"},
                    metadata={},
                )
            )
        assert "owner_space" in str(exc_info.value).lower()

    def test_index_record_filters_missing_both_raises_error(self):
        """IndexRecord with neither field should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            self._validate_index_record(
                IndexRecord(
                    id="abc123",
                    uri="ctx://acme/users/alice/memories/profile",
                    level=0,
                    text="Abstract",
                    # Missing both!
                    filters={},
                    metadata={},
                )
            )
        assert "account_id" in str(exc_info.value).lower() or "owner_space" in str(exc_info.value).lower()

    def test_index_record_filters_all_three_levels(self):
        """All 3 IndexRecords from one node must have the same filters."""
        uri = "ctx://acme/users/alice/memories/profile"
        filters = {"account_id": "acme", "owner_space": "user_space_alice"}

        l0 = IndexRecord(
            id=IndexRecord.generate_id(uri, 0),
            uri=uri,
            level=0,
            text="abstract",
            filters=filters.copy(),
            metadata={"level": "abstract"},
        )
        l1 = IndexRecord(
            id=IndexRecord.generate_id(uri, 1),
            uri=uri,
            level=1,
            text="overview",
            filters=filters.copy(),
            metadata={"level": "overview"},
        )
        l2 = IndexRecord(
            id=IndexRecord.generate_id(uri, 2),
            uri=uri,
            level=2,
            text="content",
            filters=filters.copy(),
            metadata={"level": "content"},
        )

        for record in [l0, l1, l2]:
            self._validate_index_record(record)
            assert record.filters["account_id"] == "acme"
            assert record.filters["owner_space"] == "user_space_alice"

    def test_index_record_filters_user_scope(self):
        """User-scoped nodes should have user_space_* owner_space."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/users/alice/memories/preferences/coffee",
            level=0,
            text="Likes dark roast",
            filters={"account_id": "acme", "owner_space": "user_space_alice"},
            metadata={"category": "preference"},
        )
        self._validate_index_record(record)
        assert record.filters["owner_space"].startswith("user_space_")

    def test_index_record_filters_agent_scope(self):
        """Agent-scoped nodes should have agent_space_* owner_space."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/agents/bob/skills/code_review",
            level=0,
            text="Code review skill",
            filters={"account_id": "acme", "owner_space": "agent_space_bob"},
            metadata={"category": "skill"},
        )
        self._validate_index_record(record)
        assert record.filters["owner_space"].startswith("agent_space_")

    def test_index_record_filters_can_contain_additional_fields(self):
        """filters may contain additional fields beyond the required two."""
        record = IndexRecord(
            id="abc123",
            uri="ctx://acme/users/alice/memories/profile",
            level=0,
            text="Abstract",
            filters={
                "account_id": "acme",
                "owner_space": "user_space_alice",
                "category": "profile",
                "context_type": "MEMORY",
            },
            metadata={},
        )
        self._validate_index_record(record)
        assert record.filters["category"] == "profile"
        assert record.filters["context_type"] == "MEMORY"

    def test_index_record_filters_values_non_empty(self):
        """account_id and owner_space values must be non-empty strings."""
        # Empty account_id
        with pytest.raises(ValidationError):
            self._validate_index_record(
                IndexRecord(
                    id="abc123",
                    uri="ctx://acme/users/alice/memories/profile",
                    level=0,
                    text="Abstract",
                    filters={"account_id": "", "owner_space": "user_space_alice"},
                    metadata={},
                )
            )

        # Empty owner_space
        with pytest.raises(ValidationError):
            self._validate_index_record(
                IndexRecord(
                    id="abc123",
                    uri="ctx://acme/users/alice/memories/profile",
                    level=0,
                    text="Abstract",
                    filters={"account_id": "acme", "owner_space": ""},
                    metadata={},
                )
            )

    def test_index_record_filters_isolation_guarantee(self):
        """
        Verify filters guarantee isolation: same URI, different accounts
        produce different IndexRecords that won't leak across searches.
        """
        uri = "ctx://acme/users/alice/memories/profile"

        record_acme = IndexRecord(
            id=IndexRecord.generate_id(uri, 0),
            uri=uri,
            level=0,
            text="Abstract",
            filters={"account_id": "acme", "owner_space": "user_space_alice"},
            metadata={},
        )

        record_other = IndexRecord(
            id=IndexRecord.generate_id(uri, 0),
            uri=uri,
            level=0,
            text="Abstract",
            filters={"account_id": "other-corp", "owner_space": "user_space_alice"},
            metadata={},
        )

        # Same ID, same URI, same text...
        assert record_acme.id == record_other.id
        assert record_acme.uri == record_other.uri

        # But different filters ensure isolation in vector search
        assert record_acme.filters["account_id"] == "acme"
        assert record_other.filters["account_id"] == "other-corp"
        assert record_acme.filters["account_id"] != record_other.filters["account_id"]

    # Helper method for validation

    def _validate_index_record(self, record: IndexRecord) -> None:
        """Validate IndexRecord.filters contains required fields.

        This mimics what IndexRecordBuilder and VectorIndex implementations
        should do before upserting records.

        Raises:
            ValidationError: If account_id or owner_space is missing.
        """
        required_fields = ["account_id", "owner_space"]
        missing = [f for f in required_fields if f not in record.filters]

        if missing:
            raise ValidationError(
                field="filters",
                reason=f"Missing required field(s): {', '.join(missing)}",
            )

        # Also check non-empty
        for field in required_fields:
            if not record.filters.get(field):
                raise ValidationError(
                    field="filters",
                    reason=f"Field '{field}' must be non-empty",
                )


class TestIndexRecordFiltersIntegration:
    """Integration-style tests showing how filters prevent leakage."""

    def test_vector_search_with_account_isolation(self):
        """
        Demonstrate how account_id in filters prevents cross-tenant leakage.

        When VectorIndex.search() filters by query.account_id, records
        from other accounts are excluded even if they're semantically similar.
        """
        # Imagine we're searching as user in account "acme"
        query_account_id = "acme"

        # These records have similar content but different accounts
        records = [
            IndexRecord(
                id="1",
                uri="ctx://acme/users/alice/memories/profile",
                level=0,
                text="Software engineer who loves coffee",
                filters={"account_id": "acme", "owner_space": "user_space_alice"},
                metadata={},
            ),
            IndexRecord(
                id="2",
                uri="ctx://other-corp/users/bob/memories/profile",
                level=0,
                text="Software engineer who loves coffee",  # Same text!
                filters={"account_id": "other-corp", "owner_space": "user_space_bob"},
                metadata={},
            ),
        ]

        # VectorIndex would filter like this:
        matching = [r for r in records if r.filters["account_id"] == query_account_id]

        assert len(matching) == 1
        assert matching[0].filters["account_id"] == "acme"
        # The "other-corp" record is excluded → no leakage

    def test_vector_search_with_owner_space_isolation(self):
        """
        Demonstrate how owner_space in filters prevents user-agent leakage.

        Even within same account, user and agent memories are isolated.
        """
        query_filters = {"account_id": "acme", "owner_space": "user_space_alice"}

        records = [
            IndexRecord(
                id="1",
                uri="ctx://acme/users/alice/memories/profile",
                level=0,
                text="Alice's profile",
                filters={"account_id": "acme", "owner_space": "user_space_alice"},
                metadata={},
            ),
            IndexRecord(
                id="2",
                uri="ctx://acme/agents/bot/memories/patterns/qna_style",
                level=0,
                text="Bot's Q&A patterns",
                filters={"account_id": "acme", "owner_space": "agent_space_bot"},
                metadata={},
            ),
        ]

        matching = [
            r
            for r in records
            if r.filters["account_id"] == query_filters["account_id"]
            and r.filters.get("owner_space") == query_filters["owner_space"]
        ]

        assert len(matching) == 1
        assert matching[0].filters["owner_space"] == "user_space_alice"
        # Agent memory is excluded → no leakage


class TestIndexRecordFiltersEndToEnd:
    """End-to-end integration tests with IndexRecordBuilder and InMemoryVectorIndex.

    Phase 1 supplement — verifies that ALL IndexRecord upsert paths produce
    records with valid account_id + owner_space filters.

    This is the ONLY interception point for silent cross-tenant leakage.
    """

    def test_index_record_builder_produces_valid_filters(self):
        """
        Verify IndexRecordBuilder.build_index_records() produces records
        with mandatory account_id + owner_space filters.

        This tests the PRIMARY upsert path for ContextNode → IndexRecord.
        """
        from core.models import ContextNode
        from index.index_record_builder import build_index_records

        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space_alice",
            abstract="Software engineer profile",
            overview="## Profile\n- Role: SWE\n- Focus: AI agents",
            content="Full profile content here...",
        )

        records = build_index_records(node)

        # All records must have account_id + owner_space
        for record in records:
            assert "account_id" in record.filters, (
                f"Record level {record.level} missing account_id filter. "
                "This causes SILENT cross-tenant leakage!"
            )
            assert "owner_space" in record.filters, (
                f"Record level {record.level} missing owner_space filter. "
                "This causes SILENT cross-tenant leakage!"
            )
            assert record.filters["account_id"] == "acme"
            assert record.filters["owner_space"] == "user_space_alice"

    def test_build_single_record_produces_valid_filters(self):
        """
        Verify IndexRecordBuilder.build_single_record() produces records
        with mandatory account_id + owner_space filters.

        This tests the SECONDARY upsert path used by OutboxWorker.
        """
        from index.index_record_builder import build_single_record

        record = build_single_record(
            uri="ctx://bigco/agents/bot-007/skills/debugging",
            level=0,
            text="Debugging skill",
            account_id="bigco",
            owner_space="agent_space_bot-007",
            category="skill",
            context_type="SKILL",
        )

        assert "account_id" in record.filters
        assert "owner_space" in record.filters
        assert record.filters["account_id"] == "bigco"
        assert record.filters["owner_space"] == "agent_space_bot-007"

    def test_in_memory_index_enforces_account_isolation(self):
        """
        End-to-end test: InMemoryVectorIndex.search() filters by account_id.

        Verifies that records from other accounts are NOT returned in search results,
        even if they have similar content.
        """
        from core.models import TypedQuery
        from providers.vector_index.in_memory_index import InMemoryVectorIndex

        index = InMemoryVectorIndex()

        # Upsert records from two different accounts with similar content
        index.upsert([
            IndexRecord(
                id="acme-record",
                uri="ctx://acme/users/alice/memories/profile",
                level=0,
                text="Software engineer who loves coffee",
                filters={"account_id": "acme", "owner_space": "user_space_alice"},
                metadata={"category": "profile", "context_type": "MEMORY"},
            ),
            IndexRecord(
                id="othercorp-record",
                uri="ctx://othercorp/users/bob/memories/profile",
                level=0,
                text="Software engineer who loves coffee",  # Same text!
                filters={"account_id": "othercorp", "owner_space": "user_space_bob"},
                metadata={"category": "profile", "context_type": "MEMORY"},
            ),
        ])

        # Search as acme user - use exact text match for mock vector similarity
        query = TypedQuery(
            text="Software engineer who loves coffee",  # Exact match for test
            context_type="MEMORY",
            categories=[],
            account_id="acme",  # Searching as acme
            top_k=10,
        )

        results = index.search(query)

        # Should ONLY return acme record, not othercorp
        assert len(results) == 1
        assert results[0].uri == "ctx://acme/users/alice/memories/profile"
        # othercorp record is excluded → no leakage

    def test_in_memory_index_enforces_owner_space_isolation(self):
        """
        End-to-end test: InMemoryVectorIndex.search() filters by owner_space.

        Verifies that user memories are isolated from agent memories within
        the same account.
        """
        from core.models import TypedQuery
        from providers.vector_index.in_memory_index import InMemoryVectorIndex

        index = InMemoryVectorIndex()

        # Upsert records from same account, different owner spaces
        index.upsert([
            IndexRecord(
                id="user-record",
                uri="ctx://acme/users/alice/memories/profile",
                level=0,
                text="Alice's preferences",
                filters={"account_id": "acme", "owner_space": "user_space_alice"},
                metadata={"category": "preference", "context_type": "MEMORY"},
            ),
            IndexRecord(
                id="agent-record",
                uri="ctx://acme/agents/bot/memories/patterns/qna",
                level=0,
                text="Bot's Q&A patterns",
                filters={"account_id": "acme", "owner_space": "agent_space_bot"},
                metadata={"category": "pattern", "context_type": "SKILL"},
            ),
        ])

        # Search with owner_space filter - use exact text for similarity
        query = TypedQuery(
            text="Alice's preferences",  # Exact match for test
            context_type="MEMORY",
            categories=[],
            account_id="acme",
            owner_space="user_space_alice",
            top_k=10,
        )

        results = index.search(query)

        # Should ONLY return user record
        assert len(results) == 1
        assert results[0].uri == "ctx://acme/users/alice/memories/profile"
        # Agent record is excluded → no leakage

    def test_end_to_end_context_node_to_search(self):
        """
        Full end-to-end test: ContextNode → IndexRecordBuilder → InMemoryVectorIndex → Search

        This simulates the complete pipeline:
        1. Create a ContextNode
        2. Use IndexRecordBuilder to expand into IndexRecords
        3. Upsert to InMemoryVectorIndex
        4. Search and verify account_id isolation

        Missing filters at any step would cause silent leakage.
        """
        from core.models import ContextNode, TypedQuery
        from index.index_record_builder import build_index_records
        from providers.vector_index.in_memory_index import InMemoryVectorIndex

        index = InMemoryVectorIndex()

        # Create ContextNodes for two different accounts
        node_acme = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space_alice",
            abstract="Software engineer at Acme Corp",
            overview="## Profile\n- 5 years experience\n- Specializes in AI agents",
            content="Full detailed profile...",
        )

        node_othercorp = ContextNode(
            uri="ctx://othercorp/users/bob/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space_bob",
            abstract="Software engineer at OtherCorp Inc",
            overview="## Profile\n- 3 years experience\n- Specializes in web dev",
            content="Full detailed profile...",
        )

        # Expand to IndexRecords
        records_acme = build_index_records(node_acme)
        records_othercorp = build_index_records(node_othercorp)

        # Verify all records have valid filters
        all_records = records_acme + records_othercorp
        for record in all_records:
            assert "account_id" in record.filters
            assert "owner_space" in record.filters

        # Upsert to index
        index.upsert(all_records)

        # Search as acme user - use exact text for mock vector similarity
        query = TypedQuery(
            text="Software engineer at Acme Corp",  # Exact match
            context_type="MEMORY",
            categories=["profile"],
            account_id="acme",
            top_k=10,
        )

        results = index.search(query)

        # Should ONLY return acme results
        for hit in results:
            assert "acme" in hit.uri or hit.uri.startswith("ctx://acme/")
            # No othercorp results → no leakage

    def test_filter_validation_catches_missing_account_id(self):
        """
        Verify IndexRecordBuilder._validate_filters() catches missing account_id.

        This is the last line of defense before upsert to VectorIndex.
        """
        from core.models import ContextNode
        from index.index_record_builder import build_index_records, FilterValidationError
        import pytest

        # This should NOT raise - filters are present
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=0,
            owner_space="user_space_alice",
            abstract="Profile",
            overview="Overview",
            content="Content",
        )

        records = build_index_records(node)  # Should not raise
        assert len(records) == 3

        # But if we manually create a record without filters...
        from index.index_record_builder import _validate_filters
        bad_record = IndexRecord(
            id="bad",
            uri="ctx://test/users/u1/memories/test",
            level=0,
            text="test",
            filters={},  # Missing both account_id AND owner_space
            metadata={},
        )

        with pytest.raises(FilterValidationError) as exc:
            _validate_filters([bad_record])

        assert exc.value.missing_keys == ("account_id", "owner_space")

    def test_multiple_accounts_isolated_in_search(self):
        """
        Stress test: multiple accounts with same content remain isolated.

        Simulates a realistic scenario where multiple tenants have similar
        data (e.g., "software engineer" profiles). Search must NEVER
        return results from other accounts.
        """
        from core.models import ContextNode, TypedQuery
        from index.index_record_builder import build_index_records
        from providers.vector_index.in_memory_index import InMemoryVectorIndex

        index = InMemoryVectorIndex()

        # Create 5 accounts with identical profile structure
        accounts = ["acme", "globex", "initech", "umbrella", "cyberdyne"]

        for account_id in accounts:
            node = ContextNode(
                uri=f"ctx://{account_id}/users/user/memories/profile",
                context_type="MEMORY",
                category="profile",
                level=0,
                owner_space=f"user_space_user",
                abstract="Software engineer profile",  # Identical text
                overview="## Overview\n- Software engineer\n- 5 years experience",
                content="Full profile content",
            )

            records = build_index_records(node)
            index.upsert(records)

        # Search from each account - should only get own results
        for account_id in accounts:
            # Use exact text match for mock vector similarity
            query = TypedQuery(
                text="Software engineer profile",  # Exact match
                context_type="MEMORY",
                categories=["profile"],
                account_id=account_id,
                top_k=10,
            )

            results = index.search(query)

            # All results must belong to the searching account
            for hit in results:
                assert account_id in hit.uri, (
                    f"Account {account_id} received result from {hit.uri}. "
                    "CROSS-TENANT LEAKAGE!"
                )
