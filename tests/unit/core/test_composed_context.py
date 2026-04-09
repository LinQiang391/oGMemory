"""Unit tests for ComposedContext dataclass and conversion."""

import pytest
from core.models import ComposedContext


class TestComposedContext:
    """Test ComposedContext dataclass."""

    def test_assemble_result_creation(self):
        """Test creating an ComposedContext with all fields."""
        result = ComposedContext(
            identity_context="User profile and preferences",
            episodic_context="Archive history",
            session_context="Active task and constraints",
            task_context="Current task description",
            retrieved_evidence="Working set search results",
            open_loops=["Task 1 incomplete", "Waiting for user input"],
            uncertainties=["User intent unclear", "Missing requirements"],
            estimated_tokens=5000,
            budget_used_by_slot={"identity": 1000, "session": 2000, "retrieved": 2000},
            messages=[{"role": "user", "content": "Hello"}],
            archive_count=3,
            archive_included=True,
            system_prompt_suffix="Session state suffix",
        )

        assert result.identity_context == "User profile and preferences"
        assert result.episodic_context == "Archive history"
        assert result.session_context == "Active task and constraints"
        assert result.task_context == "Current task description"
        assert result.retrieved_evidence == "Working set search results"
        assert len(result.open_loops) == 2
        assert len(result.uncertainties) == 2
        assert result.estimated_tokens == 5000
        assert result.budget_used_by_slot["identity"] == 1000
        assert result.archive_count == 3
        assert result.archive_included is True

    def test_assemble_result_defaults(self):
        """Test ComposedContext with default values."""
        result = ComposedContext()

        assert result.identity_context == ""
        assert result.episodic_context == ""
        assert result.session_context == ""
        assert result.task_context == ""
        assert result.retrieved_evidence == ""
        assert result.open_loops == []
        assert result.uncertainties == []
        assert result.estimated_tokens == 0
        assert result.budget_used_by_slot == {}
        assert result.messages == []
        assert result.archive_count == 0
        assert result.archive_included is False
        assert result.system_prompt_suffix == ""
        assert result.stats == {}

    def test_assemble_result_semantic_slots(self):
        """Test that semantic slots represent the cognitive contract."""
        result = ComposedContext(
            identity_context="Layer 1a: Profile",
            episodic_context="Layer 1b: Archives",
            session_context="Layer 2: Session state",
            retrieved_evidence="Layer 3: Working set",
            open_loops=["Loop 1", "Loop 2"],
            uncertainties=["Uncertainty 1"],
        )

        # Layer 1: Stable identity
        assert "Layer 1a" in result.identity_context
        # Layer 1: Episodic history
        assert "Layer 1b" in result.episodic_context
        # Layer 2: Session state
        assert "Layer 2" in result.session_context
        # Layer 3: Retrieved evidence
        assert "Layer 3" in result.retrieved_evidence
        # Structured state
        assert len(result.open_loops) == 2
        assert len(result.uncertainties) == 1

    def test_assemble_result_budget_metadata(self):
        """Test budget metadata tracking by slot."""
        result = ComposedContext(
            estimated_tokens=10000,
            budget_used_by_slot={
                "identity": 2000,
                "episodic": 1000,
                "session": 3000,
                "retrieved": 4000,
            },
        )

        assert result.estimated_tokens == 10000
        assert result.budget_used_by_slot["identity"] == 2000
        assert result.budget_used_by_slot["episodic"] == 1000
        assert result.budget_used_by_slot["session"] == 3000
        assert result.budget_used_by_slot["retrieved"] == 4000
        # Total should match
        total_budget_used = sum(result.budget_used_by_slot.values())
        assert total_budget_used == 10000

    def test_assemble_result_legacy_compat(self):
        """Test legacy compatibility fields."""
        result = ComposedContext(
            messages=[{"role": "user", "content": "test"}],
            archive_count=5,
            archive_included=True,
            system_prompt_suffix="Session suffix",
            stats={"key": "value"},
        )

        assert len(result.messages) == 1
        assert result.archive_count == 5
        assert result.archive_included is True
        assert result.system_prompt_suffix == "Session suffix"
        assert result.stats["key"] == "value"


class TestComposedContextConversion:
    """Test ComposedContext to dict conversion (via MemoryService)."""

    def test_dict_conversion_structure(self):
        """Test that ComposedContext can be converted to dict with all fields."""
        result = ComposedContext(
            identity_context="Profile",
            episodic_context="Archives",
            session_context="Session",
            retrieved_evidence="Working set",
            open_loops=["Loop 1"],
            uncertainties=["Uncertainty 1"],
            estimated_tokens=5000,
            budget_used_by_slot={"identity": 1000},
            messages=[{"role": "user"}],
            archive_count=2,
            archive_included=True,
            system_prompt_suffix="Suffix",
        )

        # Convert to dict (simulating what MemoryService._assemble_result_to_dict does)
        dict_result = {
            "messages": result.messages,
            "systemPromptAddition": result.identity_context + result.episodic_context,
            "systemPromptSuffix": result.system_prompt_suffix,
            "memoryUserMessage": result.retrieved_evidence,
            "estimatedTokens": result.estimated_tokens,
            "archiveCount": result.archive_count,
            "archiveIncluded": result.archive_included,
            "stats": result.stats,
            "identityContext": result.identity_context,
            "episodicContext": result.episodic_context,
            "sessionContext": result.session_context,
            "taskContext": result.task_context,
            "retrievedEvidence": result.retrieved_evidence,
            "openLoops": result.open_loops,
            "uncertainties": result.uncertainties,
            "budgetUsedBySlot": result.budget_used_by_slot,
        }

        # Verify all fields are present
        assert "messages" in dict_result
        assert "systemPromptAddition" in dict_result
        assert "systemPromptSuffix" in dict_result
        assert "memoryUserMessage" in dict_result
        assert "estimatedTokens" in dict_result
        assert "archiveCount" in dict_result
        assert "archiveIncluded" in dict_result
        assert "identityContext" in dict_result
        assert "episodicContext" in dict_result
        assert "sessionContext" in dict_result
        assert "retrievedEvidence" in dict_result
        assert "openLoops" in dict_result
        assert "uncertainties" in dict_result
        assert "budgetUsedBySlot" in dict_result

        # Verify values
        assert dict_result["identityContext"] == "Profile"
        assert dict_result["episodicContext"] == "Archives"
        assert dict_result["sessionContext"] == "Session"
        assert dict_result["retrievedEvidence"] == "Working set"
        assert len(dict_result["openLoops"]) == 1
        assert len(dict_result["uncertainties"]) == 1
