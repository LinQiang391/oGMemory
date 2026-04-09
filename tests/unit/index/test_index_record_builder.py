"""Unit tests for IndexRecordBuilder.

Tests verify the critical invariant: all IndexRecords MUST contain
account_id + owner_space in filters to prevent cross-tenant leakage.
"""

import pytest

from core.models import ContextNode
from index.index_record_builder import (
    build_index_records,
    build_single_record,
    _extract_account_id,
    _validate_filters,
    FilterValidationError,
)


class TestExtractAccountId:
    """Tests for _extract_account_id utility."""

    def test_extract_from_user_memory_uri(self):
        uri = "ctx://acme/users/alice/memories/profile"
        account = _extract_account_id(uri)
        assert account == "acme"

    def test_extract_from_agent_memory_uri(self):
        uri = "ctx://acme/agents/bob/memories/cases/case_123"
        account = _extract_account_id(uri)
        assert account == "acme"

    def test_extract_from_skill_uri(self):
        uri = "ctx://acme/agents/bob/skills/code_review"
        account = _extract_account_id(uri)
        assert account == "acme"

    def test_invalid_uri_raises_error(self):
        with pytest.raises(ValueError, match="Invalid URI format"):
            _extract_account_id("not-a-uri")

    def test_missing_account_raises_error(self):
        with pytest.raises(ValueError, match="Invalid URI format"):
            _extract_account_id("ctx:///")


class TestValidateFilters:
    """Tests for _validate_filters utility."""

    def test_all_records_with_valid_filters_pass(self):
        from core.models import IndexRecord

        records = [
            IndexRecord(
                id="abc123",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test", "owner_space": "user:u1"},
            ),
        ]
        # Should not raise
        _validate_filters(records)

    def test_record_missing_account_id_fails(self):
        from core.models import IndexRecord

        records = [
            IndexRecord(
                id="abc123",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"owner_space": "user:u1"},  # Missing account_id
            ),
        ]

        with pytest.raises(FilterValidationError) as exc:
            _validate_filters(records)

        assert "account_id" in exc.value.missing_keys

    def test_record_missing_owner_space_fails(self):
        from core.models import IndexRecord

        records = [
            IndexRecord(
                id="abc123",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={"account_id": "test"},  # Missing owner_space
            ),
        ]

        with pytest.raises(FilterValidationError) as exc:
            _validate_filters(records)

        assert "owner_space" in exc.value.missing_keys

    def test_record_missing_both_keys_fails(self):
        from core.models import IndexRecord

        records = [
            IndexRecord(
                id="abc123",
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                filters={},  # Missing both
            ),
        ]

        with pytest.raises(FilterValidationError) as exc:
            _validate_filters(records)

        assert exc.value.missing_keys == ("account_id", "owner_space")


class TestBuildIndexRecords:
    """Tests for build_index_records function."""

    def test_returns_three_records(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)

        assert len(records) == 3

    def test_level_0_abstract_record(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)
        level_0 = next(r for r in records if r.level == 0)

        assert level_0.text == "User profile"
        assert level_0.metadata["level"] == "abstract"

    def test_level_1_overview_record(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)
        level_1 = next(r for r in records if r.level == 1)

        assert level_1.text == "Detailed overview"
        assert level_1.metadata["level"] == "overview"

    def test_level_2_content_record(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)
        level_2 = next(r for r in records if r.level == 2)

        assert level_2.text == "Full content"
        assert level_2.metadata["level"] == "content"

    def test_all_records_contain_account_id_filter(self):
        """CRITICAL: This is the cross-tenant leakage prevention test."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)

        for record in records:
            assert "account_id" in record.filters, (
                f"Record level {record.level} missing account_id in filters. "
                "This causes SILENT cross-tenant leakage!"
            )
            assert record.filters["account_id"] == "acme"

    def test_all_records_contain_owner_space_filter(self):
        """CRITICAL: This is the cross-tenant leakage prevention test."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)

        for record in records:
            assert "owner_space" in record.filters, (
                f"Record level {record.level} missing owner_space in filters. "
                "This causes SILENT cross-tenant leakage!"
            )
            assert record.filters["owner_space"] == "user:alice"

    def test_all_records_contain_category_and_context_type(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)

        for record in records:
            assert record.filters["category"] == "profile"
            assert record.filters["context_type"] == "MEMORY"

    def test_missing_overview_skips_level_1(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="",  # Empty
            content="Full content",
        )

        records = build_index_records(node)

        # Should only have level 0 and level 2
        assert len(records) == 2
        levels = {r.level for r in records}
        assert levels == {0, 2}

    def test_missing_content_skips_level_2(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="",  # Empty
        )

        records = build_index_records(node)

        # Should only have level 0 and level 1
        assert len(records) == 2
        levels = {r.level for r in records}
        assert levels == {0, 1}

    def test_id_is_stable(self):
        """Same URI and level should produce same ID."""
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records1 = build_index_records(node)
        records2 = build_index_records(node)

        for r1, r2 in zip(records1, records2):
            assert r1.id == r2.id

    def test_different_levels_different_ids(self):
        node = ContextNode(
            uri="ctx://acme/users/alice/memories/profile",
            context_type="MEMORY",
            category="profile",
            level=3,
            owner_space="user:alice",
            abstract="User profile",
            overview="Detailed overview",
            content="Full content",
        )

        records = build_index_records(node)
        ids = {r.id for r in records}

        assert len(ids) == 3  # All IDs should be unique

    def test_agent_uri_extract_account_correctly(self):
        node = ContextNode(
            uri="ctx://bigco/agents/agent-007/memories/cases/case_123",
            context_type="MEMORY",
            category="case",
            level=5,
            owner_space="agent:agent-007",
            abstract="Agent case",
            overview="Case overview",
            content="Case content",
        )

        records = build_index_records(node)

        for record in records:
            assert record.filters["account_id"] == "bigco"


class TestBuildSingleRecord:
    """Tests for build_single_record utility function."""

    def test_builds_valid_record(self):
        record = build_single_record(
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="abstract",
            account_id="test",
            owner_space="user:u1",
            category="profile",
            context_type="MEMORY",
        )

        assert record.uri == "ctx://test/users/u1/memories/profile"
        assert record.level == 0
        assert record.text == "abstract"

    def test_record_has_mandatory_filters(self):
        record = build_single_record(
            uri="ctx://test/users/u1/memories/profile",
            level=0,
            text="abstract",
            account_id="test",
            owner_space="user:u1",
            category="profile",
            context_type="MEMORY",
        )

        assert record.filters["account_id"] == "test"
        assert record.filters["owner_space"] == "user:u1"

    def test_empty_account_id_raises_error(self):
        with pytest.raises(FilterValidationError) as exc:
            build_single_record(
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                account_id="",  # Empty
                owner_space="user:u1",
                category="profile",
                context_type="MEMORY",
            )

        assert "account_id" in exc.value.missing_keys

    def test_empty_owner_space_raises_error(self):
        with pytest.raises(FilterValidationError) as exc:
            build_single_record(
                uri="ctx://test/users/u1/memories/profile",
                level=0,
                text="abstract",
                account_id="test",
                owner_space="",  # Empty
                category="profile",
                context_type="MEMORY",
            )

        assert "owner_space" in exc.value.missing_keys
