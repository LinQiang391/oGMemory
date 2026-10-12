import json

from skills.skill_creator import creation_prompt, revision_decision_prompt, revision_prompt


def _brief() -> dict:
    return {
        "schema_version": 1,
        "mode": "create",
        "capability_key": "edit-workbook",
        "observed_tools": ["spreadsheet"],
        "implementation_contract": {
            "language": "python", "declared_inputs": ["source"], "owned_by": "ogmemory",
        },
        "evidence": [{"summary": "Edit a workbook", "skill_facts": {"inputs": []}}],
    }


def test_creation_prompt_accepts_only_structured_brief_and_denies_code_ownership() -> None:
    prompt = creation_prompt(_brief())

    assert "SKILL_CREATION_BRIEF_DATA" in prompt
    assert '"capability_key": "edit-workbook"' in prompt
    assert "Return an empty implementation_template" in prompt
    assert "already capable agent" in prompt
    assert "Do not author generic host-agent execution policy" in prompt
    assert "minimal evidence-supported scope" in prompt
    assert "one self-contained SKILL.md" in prompt
    assert "companion scripts" in prompt
    assert "BUNDLED_SKILL_CREATOR_GUIDE" not in prompt
    assert "raw trace" not in prompt.casefold()


def test_revision_decision_is_separate_from_rewriting() -> None:
    prompt = revision_decision_prompt(
        {"name": "edit-workbook", "completion_checks": ["Verify output."]},
        {"mode": "failed_execution", "observation": "Agent ended early."},
        failed=True,
    )

    assert "Do not rewrite the Skill" in prompt
    assert "Agent noncompliance is not a Skill defect" in prompt
    assert "needs_revision=true" in prompt


def test_revision_prompt_contains_only_approved_brief() -> None:
    brief = _brief() | {"mode": "revise", "defect": {"required_change": "Add one check."}}
    prompt = revision_prompt(brief)

    assert "SKILL_REVISION_BRIEF_DATA" in prompt
    assert json.dumps(brief, ensure_ascii=False, sort_keys=True) in prompt
    assert "Apply only the approved minimal change" in prompt
    assert "BUNDLED_SKILL_CREATOR_GUIDE" not in prompt


def test_worked_example_rules_are_absent_when_disabled() -> None:
    creation = creation_prompt(_brief(), include_worked_examples=False)
    revision = revision_prompt(_brief(), include_worked_examples=False)

    assert "worked example" not in creation.casefold()
    assert "examples" not in creation.casefold()
    assert "worked example" not in revision.casefold()
    assert "examples" not in revision.casefold()
