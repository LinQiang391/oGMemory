"""Tests for M3 config loading from ogmemory.yaml."""

from __future__ import annotations

from pathlib import Path

from m3.config import load_m3_runtime_config


def test_load_m3_runtime_config_reads_m3_block_without_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg_path = tmp_path / "ogmemory.yaml"
    cfg_path.write_text(
        """
llm:
  provider: openai
  api_key: top-key
  base_url: https://llm.example/v1
  model: gpt-test
embedding:
  provider: mock
  model: top-embedding
vector_db:
  dimension: 12
m3:
  episodic_llm:
    provider: minimax
    api_key: episodic-key
    base_url: https://api.minimaxi.com/v1
    model: MiniMax-M3
    temperature: 0.2
    max_tokens: 777
  qa_llm:
    provider: mock
    model: qa-model
  embedding:
    provider: mock
    model: m3-embedding
    dimension: 8
""".strip(),
        encoding="utf-8",
    )

    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.provider == "minimax"
    assert cfg.episodic_llm.api_key == "episodic-key"
    assert cfg.episodic_llm.model == "MiniMax-M3"
    assert cfg.episodic_llm.max_tokens == 777
    assert cfg.qa_llm.provider == "mock"
    assert cfg.qa_llm.model == "qa-model"
    assert cfg.embedding_provider == "mock"
    assert cfg.embedding_model == "m3-embedding"
    assert cfg.embedding_dimension == 8


def test_load_m3_runtime_config_env_placeholders_require_export(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg_path = tmp_path / "ogmemory.yaml"
    cfg_path.write_text(
        """
m3:
  episodic_llm:
    provider: minimax
    api_key: "${MINIMAX_API_KEY}"
    model: MiniMax-M3
""".strip(),
        encoding="utf-8",
    )
    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.api_key is None

    monkeypatch.setenv("MINIMAX_API_KEY", "exported-key")
    cfg = load_m3_runtime_config(str(cfg_path))
    assert cfg.episodic_llm.api_key == "exported-key"
