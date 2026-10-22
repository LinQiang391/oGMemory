from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_scene_imports_from_built_wheel(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(wheel)!r}); import scene; "
            "from scene.decision_contracts import PreflightOutput; "
            "from scene.decision_evidence import SceneEvidenceBuilder; "
            "from scene.decision_calls import CallBudget, SceneModelCaller; "
            "from scene.decision_preflight import ScenePreflight; "
            "from scene.decision_prompts import QUALITY_PROMPT; "
            "assert SceneEvidenceBuilder().limits.max_input_nodes == 128; "
            "assert CallBudget(3).remaining == 3; "
            "assert ScenePreflight(SceneModelCaller(None)).config.max_react_rounds == 2; "
            "assert PreflightOutput(status='READY').status == 'READY'; "
            "assert QUALITY_PROMPT.endswith('PROPOSAL:\\n'); print(scene.__name__)",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "scene"
