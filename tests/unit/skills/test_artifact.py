import pytest

from skills.artifact import (
    MAX_SKILL_NAME_LENGTH,
    normalize_skill_name,
    validate_skill_name,
)


@pytest.mark.parametrize(
    "value",
    [
        "a" * (MAX_SKILL_NAME_LENGTH - 1) + "--suffix",
        "a" * (MAX_SKILL_NAME_LENGTH - 1) + " + suffix",
        "Mixed CASE / punctuation / capability",
        "---already---separated---",
        12345,
        None,
    ],
)
def test_every_nonempty_normalized_skill_name_is_valid(value):
    normalized = normalize_skill_name(value)
    assert normalized
    assert validate_skill_name(normalized) == normalized
    assert len(normalized) <= MAX_SKILL_NAME_LENGTH


@pytest.mark.parametrize("value", ["", "---", " "])
def test_normalize_skill_name_can_return_empty_for_content_without_identifiers(value):
    assert normalize_skill_name(value) == ""
