"""Prompt template manager for extraction prompts.

Loads Jinja2 YAML templates, renders them with variables, and provides
fallback to ensure robustness when template files are missing or corrupted.
"""

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateError

from core.logging_config import get_logger

logger = get_logger(__name__)

# Default template directory (sibling of this file)
_DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates"
_REQUIRED_SECTIONS = ("system_prompt", "output_instruction")


class PromptManager:
    """Load and render extraction prompt templates.

    Templates are YAML files with sections like system_prompt, examples,
    conversation_header, output_instruction. Each section can contain
    Jinja2 variables for dynamic rendering.

    Usage:
        mgr = PromptManager()
        prompt = mgr.render("extraction", "system_prompt",
                            session_summary="Previously extracted...")
        has = mgr.has_template("extraction")
    """

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize PromptManager.

        Args:
            template_dir: Path to templates directory.
                         Defaults to extraction/prompts/templates/
        """
        if template_dir is None:
            template_dir = _DEFAULT_TEMPLATE_DIR
        self._template_dir = Path(template_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            autoescape=False,  # We're generating prompts, not HTML
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._cache: dict[str, dict] = {}

    def load(self, template_name: str) -> dict:
        """Load and cache a YAML template file.

        Args:
            template_name: Template name without .yaml extension

        Returns:
            Dict parsed from YAML file

        Raises:
            FileNotFoundError: If template file does not exist
        """
        if template_name not in self._cache:
            path = self._template_dir / f"{template_name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Template not found: {path}")
            try:
                with open(path, encoding="utf-8") as f:
                    template_data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                location = ""
                if mark is not None:
                    location = f" line {mark.line + 1}, column {mark.column + 1}"
                raise ValueError(f"Failed to parse template {path}{location}: {exc}") from exc
            if not isinstance(template_data, dict):
                raise ValueError(f"Template {path} must be a YAML mapping")
            missing = [key for key in _REQUIRED_SECTIONS if not template_data.get(key)]
            if missing:
                raise ValueError(
                    f"Template {path} missing required section(s): {', '.join(missing)}"
                )
            self._cache[template_name] = template_data
            logger.debug("Loaded template: %s", template_name)
        return self._cache[template_name]

    def render(
        self,
        template_name: str,
        section: str,
        **variables: Any,
    ) -> str:
        """Render a specific section of a template with Jinja2.

        Args:
            template_name: Template name without .yaml extension
            section: Key within the YAML file (e.g. "system_prompt")
            **variables: Jinja2 template variables

        Returns:
            Rendered string, or empty string if section not found
        """
        try:
            template_data = self.load(template_name)
        except FileNotFoundError as exc:
            logger.warning("%s, returning empty", exc)
            return ""

        raw_text = template_data.get(section, "")
        if not raw_text:
            return ""

        try:
            tmpl = self._env.from_string(str(raw_text))
            return tmpl.render(**variables)
        except TemplateError as e:
            logger.error("Template render error in %s/%s: %s", template_name, section, e)
            # Return raw text as fallback
            return str(raw_text)

    def has_template(self, template_name: str) -> bool:
        """Check if a template file exists."""
        return (self._template_dir / f"{template_name}.yaml").exists()
