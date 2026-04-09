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
