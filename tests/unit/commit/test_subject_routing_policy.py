"""Test subject-based profile routing policy."""

import pytest
from unittest.mock import Mock

from core.models import RequestContext, CandidateMemory
from commit.merge_policies import SubjectRoutingProfilePolicy


def create_mock_fs():
    """Create a mock ContextFS for testing."""
    mock_fs = Mock()
    mock_fs.exists.return_value = False
    mock_fs.read_node.return_value = None
    return mock_fs


def create_test_context(user_id: str = "alice") -> RequestContext:
    """Create a test RequestContext."""
    return RequestContext(
        account_id="test-acct",
        user_id=user_id,
        agent_id="agent-1",
        session_id="sess-1",
        trace_id="trace-1",
    )


class TestSubjectRoutingProfilePolicy:
    """Test SubjectRoutingProfilePolicy subject resolution and routing."""

    # -- Self / None subjects → profile path --

    def test_subject_self_uses_user_id(self):
        """subject='self' should resolve to profile path under ctx.user_id."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="User's name",
            overview="## Name\n- Alice",
            content="Alice",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/profile/name"
        assert plan.action == "create"

    def test_subject_none_uses_current_user(self):
        """subject=None should use profile path under ctx.user_id (backward compatible)."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="User's name",
            overview="## Name\n- Alice",
            content="Alice",
            confidence=0.95,
            subject=None,
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/profile/name"

    def test_subject_case_insensitive_self(self):
        """subject='SELF' should also resolve to profile path under ctx.user_id."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="User's name",
            overview="## Name\n- Alice",
            content="Alice",
            confidence=0.95,
            subject="SELF",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/profile/name"

    # -- Named subjects → entity path under current user --

    def test_subject_named_routes_to_entity(self):
        """subject='Caroline' should route to entity under current user's namespace."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="age",
            abstract="Caroline is 28",
            overview="## Age\n- 28",
            content="Caroline is 28 years old",
            confidence=0.95,
            subject="Caroline",
        )

        plan = policy.plan(candidate, ctx)

        # Named other person → entity path, NOT virtual user path
        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"

    def test_subject_normalization_handles_spaces(self):
        """Subject names with spaces should be normalized to underscores in entity slug."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Mary Jane's profile",
            overview="## Name\n- Mary Jane",
            content="Mary Jane",
            confidence=0.95,
            subject="Mary Jane",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/mary_jane"

    def test_subject_normalization_multiple_spaces(self):
        """Multiple consecutive spaces should become underscores in entity slug."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Test profile",
            overview="## Name\n- Test",
            content="Test",
            confidence=0.95,
            subject="Mary  Jane  Smith",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/mary__jane__smith"

    def test_subject_mixed_case_name_normalized_to_lowercase(self):
        """Mixed case names should be normalized to lowercase in entity slug."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Test profile",
            overview="## Name\n- Test",
            content="Test",
            confidence=0.95,
            subject="CaRoLiNe",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"

    def test_subject_with_hyphens_preserved(self):
        """Subject names with hyphens should preserve hyphens in entity slug."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Test profile",
            overview="## Name\n- Test",
            content="Test",
            confidence=0.95,
            subject="Anne-Marie",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/anne-marie"

    def test_subject_with_spaces_and_hyphens(self):
        """Subject with both spaces and hyphens should normalize correctly."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Test profile",
            overview="## Name\n- Test",
            content="Test",
            confidence=0.95,
            subject="Anne Marie Smith-Jones",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/anne_marie_smith-jones"

    # -- Multiple subjects route to different entity URIs --

    def test_multiple_subjects_route_to_different_entity_uris(self):
        """Different named subjects should route to different entity URIs."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        caroline_candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="age",
            abstract="Caroline is 28",
            overview="## Age\n- 28",
            content="Caroline is 28 years old",
            confidence=0.95,
            subject="Caroline",
        )

        melanie_candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="location",
            abstract="Melanie lives in Portland",
            overview="## Location\n- Portland",
            content="Melanie lives in Portland.",
            confidence=0.95,
            subject="Melanie",
        )

        caroline_plan = policy.plan(caroline_candidate, ctx)
        melanie_plan = policy.plan(melanie_candidate, ctx)

        assert caroline_plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"
        assert melanie_plan.target_uri == "ctx://test-acct/users/alice/memories/entities/melanie"
        assert caroline_plan.target_uri != melanie_plan.target_uri

    # -- Same-named subject merges across routing_keys --

    def test_same_subject_different_routing_keys_merge_into_same_entity(self):
        """All attributes for one person should merge into the same entity node."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        age_candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="age",
            abstract="Caroline is 28",
            overview="## Age\n- 28",
            content="Caroline is 28 years old",
            confidence=0.95,
            subject="Caroline",
        )

        occupation_candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="occupation",
            abstract="Caroline is a teacher",
            overview="## Occupation\n- Teacher",
            content="Caroline works as a teacher",
            confidence=0.95,
            subject="Caroline",
        )

        age_plan = policy.plan(age_candidate, ctx)
        occupation_plan = policy.plan(occupation_candidate, ctx)

        # Both route to the same entity node regardless of routing_key
        assert age_plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"
        assert occupation_plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"

    # -- Error cases --

    def test_subject_empty_string_raises_error(self):
        """Empty string subject should raise ValueError (no fallback)."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)

        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="User's name",
            overview="Name",
            content="Alice",
            confidence=0.95,
            subject="",
        )

        with pytest.raises(ValueError) as exc_info:
            policy.plan(candidate, ctx)

        assert "empty string" in str(exc_info.value).lower()

    # -- Merge vs create actions --

    def test_existing_self_profile_returns_merge_action(self):
        """Existing self profile should return action='merge'."""
        mock_fs = create_mock_fs()
        mock_fs.exists.return_value = True

        existing_node = Mock()
        existing_node.metadata = {"version": 1}
        mock_fs.read_node.return_value = existing_node

        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="Updated name",
            overview="## Name\n- Alice Updated",
            content="Alice Updated",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.action == "merge"
        assert plan.target_uri == "ctx://test-acct/users/alice/memories/profile/name"
        assert plan.merged_fields is not None
        assert "abstract" in plan.merged_fields
        assert "overview" in plan.merged_fields
        assert "content" in plan.merged_fields

    def test_existing_entity_returns_merge_action(self):
        """Existing entity for a named subject should return action='merge'."""
        mock_fs = create_mock_fs()
        mock_fs.exists.return_value = True

        existing_node = Mock()
        existing_node.metadata = {"version": 2}
        mock_fs.read_node.return_value = existing_node

        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="age",
            abstract="Caroline is 29 now",
            overview="## Age\n- 29",
            content="Caroline is 29 years old",
            confidence=0.95,
            subject="Caroline",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.action == "merge"
        assert plan.target_uri == "ctx://test-acct/users/alice/memories/entities/caroline"
        # Named entity merge uses append semantics
        assert "content_append" in plan.merged_fields
        assert "overview_append" in plan.merged_fields
        assert "existing_content" in plan.merged_fields

    def test_nonexistent_returns_create_action(self):
        """Nonexistent node should return action='create'."""
        mock_fs = create_mock_fs()
        mock_fs.exists.return_value = False

        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="name",
            abstract="User's name",
            overview="## Name\n- Alice",
            content="Alice",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.action == "create"
        assert plan.merged_fields == {}

    # -- Entity merge uses append semantics --

    def test_entity_merge_uses_append_not_replace(self):
        """Named entity merge should use content_append/overview_append, not replace."""
        mock_fs = create_mock_fs()
        mock_fs.exists.return_value = True

        existing_node = Mock()
        existing_node.metadata = {"version": 1}
        existing_node.abstract = "Caroline is a person."
        existing_node.overview = "## Occupation\n- Teacher"
        existing_node.content = "Caroline works as a teacher."
        mock_fs.read_node.return_value = existing_node

        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="age",
            abstract="Caroline is 28",
            overview="## Age\n- 28",
            content="Caroline is 28 years old.",
            confidence=0.95,
            subject="Caroline",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.action == "merge"
        # Append fields must be present
        assert plan.merged_fields["content_append"] == "Caroline is 28 years old."
        assert plan.merged_fields["existing_content"] == "Caroline works as a teacher."
        assert plan.merged_fields["overview_append"] == "## Age\n- 28"
        # No replace-style keys (would cause overwrite)
        assert "content" not in plan.merged_fields
        assert "overview" not in plan.merged_fields

    def test_self_profile_merge_still_uses_replace(self):
        """Self-profile merge should still use replace semantics."""
        mock_fs = create_mock_fs()
        mock_fs.exists.return_value = True

        existing_node = Mock()
        existing_node.metadata = {"version": 1}
        mock_fs.read_node.return_value = existing_node

        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="occupation",
            abstract="I am a developer",
            overview="## Occupation\n- Developer",
            content="I am a developer.",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert plan.action == "merge"
        # Replace-style keys for self-profile
        assert "abstract" in plan.merged_fields
        assert "overview" in plan.merged_fields
        assert "content" in plan.merged_fields
        assert "content_append" not in plan.merged_fields

    # -- Misattributed self-subject detection --

    def test_misattributed_self_caroline_redirects_to_entity(self):
        """subject='self' with content about Caroline → redirect to entities/caroline."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="eval-1")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="gender",
            abstract="Caroline is a transgender woman",
            overview="## Gender\n- Transgender woman",
            content="Caroline is a transgender woman.",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        # Should redirect to entity, not profile
        assert "entities/caroline" in plan.target_uri
        assert "profile" not in plan.target_uri

    def test_misattributed_self_melanie_redirects_to_entity(self):
        """subject='self' with content 'Melanie loves pottery' → redirect to entities/melanie."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="u-alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="hobbies",
            abstract="Melanie loves pottery",
            overview="## Hobbies\n- Pottery",
            content="Melanie loves pottery and painting.",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert "entities/melanie" in plan.target_uri
        assert "profile" not in plan.target_uri

    def test_self_subject_with_i_am_stays_as_profile(self):
        """subject='self' with 'I am...' content should stay as profile."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="occupation",
            abstract="I am a developer",
            overview="## Occupation\n- Developer",
            content="I am a developer.",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert "profile/occupation" in plan.target_uri

    def test_self_subject_with_user_generic_stays_as_profile(self):
        """subject='self' with 'User is a developer' should stay as profile (generic word)."""
        mock_fs = create_mock_fs()
        policy = SubjectRoutingProfilePolicy(mock_fs)
        ctx = create_test_context(user_id="alice")

        candidate = CandidateMemory(
            category="profile",
            owner_scope="user",
            routing_key="occupation",
            abstract="User is a developer",
            overview="## Occupation\n- Developer",
            content="User is a developer.",
            confidence=0.95,
            subject="self",
        )

        plan = policy.plan(candidate, ctx)

        assert "profile/occupation" in plan.target_uri
