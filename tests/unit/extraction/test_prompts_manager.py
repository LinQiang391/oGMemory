import logging

import pytest

from extraction.prompts.manager import PromptManager


def test_load_yaml_error_includes_template_path_and_line(tmp_path):
    template_path = tmp_path / "broken.yaml"
    template_path.write_text("system_prompt: [unterminated\n", encoding="utf-8")

    mgr = PromptManager(template_dir=tmp_path)

    with pytest.raises(ValueError) as exc:
        mgr.load("broken")

    message = str(exc.value)
    assert str(template_path) in message
    assert "line" in message.lower()


def test_load_validates_required_sections(tmp_path):
    template_path = tmp_path / "incomplete.yaml"
    template_path.write_text("system_prompt: hello\n", encoding="utf-8")

    mgr = PromptManager(template_dir=tmp_path)

    with pytest.raises(ValueError) as exc:
        mgr.load("incomplete")

    assert str(template_path) in str(exc.value)
    assert "output_instruction" in str(exc.value)


def test_render_missing_template_logs_path_and_returns_empty(tmp_path, caplog):
    mgr = PromptManager(template_dir=tmp_path)

    with caplog.at_level(logging.WARNING):
        rendered = mgr.render("missing", "system_prompt")

    assert rendered == ""
    assert str(tmp_path / "missing.yaml") in caplog.text


def test_default_template_dir_is_used():
    mgr = PromptManager()

    assert mgr.has_template("extraction") is True


def test_load_rejects_non_mapping_yaml(tmp_path):
    template_path = tmp_path / "list_template.yaml"
    template_path.write_text("- not\n- mapping\n", encoding="utf-8")

    mgr = PromptManager(template_dir=tmp_path)

    with pytest.raises(ValueError) as exc:
        mgr.load("list_template")

    assert str(template_path) in str(exc.value)
    assert "YAML mapping" in str(exc.value)


def test_render_existing_section_and_uses_cache(tmp_path):
    template_path = tmp_path / "custom.yaml"
    template_path.write_text(
        """
system_prompt: "Hello {{ name }}"
output_instruction: "Return JSON"
optional: "Value {{ value }}"
""",
        encoding="utf-8",
    )
    mgr = PromptManager(template_dir=tmp_path)

    assert mgr.render("custom", "system_prompt", name="Ada") == "Hello Ada"
    template_path.write_text(
        """
system_prompt: "Changed"
output_instruction: "Return JSON"
""",
        encoding="utf-8",
    )
    assert mgr.render("custom", "system_prompt", name="Ada") == "Hello Ada"
    assert mgr.render("custom", "missing") == ""


def test_render_returns_raw_text_on_template_error(tmp_path, caplog):
    template_path = tmp_path / "bad_render.yaml"
    template_path.write_text(
        """
system_prompt: "{{ broken"
output_instruction: "Return JSON"
""",
        encoding="utf-8",
    )
    mgr = PromptManager(template_dir=tmp_path)

    with caplog.at_level(logging.ERROR):
        rendered = mgr.render("bad_render", "system_prompt")

    assert rendered == "{{ broken"
    assert "Template render error" in caplog.text


def test_has_template_false_for_missing_template(tmp_path):
    mgr = PromptManager(template_dir=tmp_path)

    assert mgr.has_template("missing") is False
