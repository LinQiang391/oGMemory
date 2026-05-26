"""Tests for merge_policies and archive_builder provenance handling.

TODO (missing coverage for merge_policies.py):
- ProfilePolicy: content replacement on merge, abstract/overview/content fields
- AggregateTopicPolicy: overview_append / content_append merged_fields on merge,
  create action when no existing node, similar-slug routing
- SkillToolPolicy: skill_merge merged_fields, usage_count increment, create with usage_count=1
- AppendOnlyPolicy: always create, unique event/case URI generation
- Edge cases: optimistic locking (expected_version), empty candidate fields
"""

import pytest
from unittest.mock import Mock

from commit.merge_policies import (
    _merge_provenance_ids,
    ProfilePolicy,
    AggregateTopicPolicy,
    SkillToolPolicy,
)
from commit.archive_builder import ArchiveBuilder
from core.models import CandidateMemory, RequestContext, WritePlan


# --- Helpers ---

def _make_ctx():
    return RequestContext(
        account_id="test", user_id="test", agent_id="test",
        session_id="test", trace_id="test",
    )


def _make_candidate(category="profile", provenance_ids=None):
    return CandidateMemory(
        category=category, routing_key="test_key",
        abstract="test abstract", overview="test overview",
        content="test content", confidence=0.9, owner_scope="user",
        provenance_ids=provenance_ids or [],
    )


def _mock_fs(existing_provenance_ids=None):
    fs = Mock()
    node = Mock()
    node.metadata = {"version": 1}
    if existing_provenance_ids is not None:
        node.metadata["provenance_ids"] = existing_provenance_ids
    node.overview = "existing overview"
    node.content = "existing content"
    node.abstract = "existing abstract"
    fs.exists.return_value = True
    fs.read_node.return_value = node
    return fs


def _mock_uri_resolver():
    resolver = Mock()
    resolver.resolve.return_value = "ctx://test/path"
    return resolver


# --- _merge_provenance_ids ---

class TestMergeProvenanceIds:
    def test_both_empty(self):
        assert _merge_provenance_ids(None, None) == []

    def test_existing_only(self):
        assert _merge_provenance_ids(["prov:1:archive:a1:"], None) == ["prov:1:archive:a1:"]

    def test_incoming_only(self):
        assert _merge_provenance_ids(None, ["prov:1:archive:b1:"]) == ["prov:1:archive:b1:"]

    def test_merge_preserves_order(self):
        result = _merge_provenance_ids(
            ["prov:1:archive:a1:", "prov:1:archive:a2:"],
            ["prov:1:archive:b1:"],
        )
        assert result == ["prov:1:archive:a1:", "prov:1:archive:a2:", "prov:1:archive:b1:"]

    def test_dedup_preserves_first_occurrence(self):
        dup = "prov:1:archive:a1:"
        result = _merge_provenance_ids(
            [dup, "prov:1:archive:a2:"],
            ["prov:1:archive:b1:", dup],
        )
        assert result == [dup, "prov:1:archive:a2:", "prov:1:archive:b1:"]

    def test_all_duplicates(self):
        dup = "prov:1:archive:a1:"
        assert _merge_provenance_ids([dup, dup], [dup]) == [dup]


# --- Policy provenance behavior ---

class TestProfilePolicyProvenance:
    def test_merge_combines_provenance(self):
        fs = _mock_fs(existing_provenance_ids=["prov:1:archive:old1:"])
        policy = ProfilePolicy(fs, _mock_uri_resolver())
        candidate = _make_candidate(provenance_ids=["prov:1:archive:new1:"])
        plan = policy.plan(candidate, _make_ctx())
        assert plan.merged_fields["provenance_ids"] == [
            "prov:1:archive:old1:", "prov:1:archive:new1:",
        ]

    def test_create_has_no_merged_provenance(self):
        fs = _mock_fs()
        fs.exists.return_value = False
        policy = ProfilePolicy(fs, _mock_uri_resolver())
        candidate = _make_candidate(provenance_ids=["prov:1:archive:new1:"])
        plan = policy.plan(candidate, _make_ctx())
        assert "provenance_ids" not in plan.merged_fields


class TestAggregateTopicPolicyProvenance:
    def test_merge_combines_provenance(self):
        fs = _mock_fs(existing_provenance_ids=["prov:1:archive:a:", "prov:1:archive:b:"])
        policy = AggregateTopicPolicy(fs, _mock_uri_resolver())
        candidate = _make_candidate(category="preference", provenance_ids=["prov:1:archive:c:"])
        plan = policy.plan(candidate, _make_ctx())
        assert plan.merged_fields["provenance_ids"] == [
            "prov:1:archive:a:", "prov:1:archive:b:", "prov:1:archive:c:",
        ]


class TestSkillToolPolicyProvenance:
    def test_merge_combines_provenance(self):
        fs = _mock_fs(existing_provenance_ids=["prov:1:archive:old:"])
        fs.read_node.return_value.metadata["usage_count"] = 3
        policy = SkillToolPolicy(fs, _mock_uri_resolver())
        candidate = _make_candidate(category="skill", provenance_ids=["prov:1:archive:new:"])
        plan = policy.plan(candidate, _make_ctx())
        assert plan.merged_fields["provenance_ids"] == [
            "prov:1:archive:old:", "prov:1:archive:new:",
        ]


# --- ArchiveBuilder provenance wiring ---

class TestArchiveBuilderProvenance:
    def test_prefers_merged_fields_over_candidate(self):
        llm = Mock()
        builder = ArchiveBuilder(llm)
        candidate = _make_candidate(provenance_ids=["prov:1:archive:from_candidate:"])
        plan = WritePlan(
            action="merge", target_uri="ctx://test/path",
            merged_fields={"provenance_ids": ["prov:1:archive:m1:", "prov:1:archive:m2:"]},
            relation_edges=[],
        )
        node = builder.build(candidate, plan, _make_ctx())
        assert node.metadata["provenance_ids"] == ["prov:1:archive:m1:", "prov:1:archive:m2:"]

    def test_falls_back_to_candidate_on_create(self):
        llm = Mock()
        builder = ArchiveBuilder(llm)
        candidate = _make_candidate(provenance_ids=["prov:1:archive:from_candidate:"])
        plan = WritePlan(
            action="create", target_uri="ctx://test/path",
            merged_fields={}, relation_edges=[],
        )
        node = builder.build(candidate, plan, _make_ctx())
        assert node.metadata["provenance_ids"] == ["prov:1:archive:from_candidate:"]

    def test_no_provenance_when_both_empty(self):
        llm = Mock()
        builder = ArchiveBuilder(llm)
        candidate = _make_candidate(provenance_ids=[])
        plan = WritePlan(
            action="create", target_uri="ctx://test/path",
            merged_fields={}, relation_edges=[],
        )
        node = builder.build(candidate, plan, _make_ctx())
        assert "provenance_ids" not in node.metadata
