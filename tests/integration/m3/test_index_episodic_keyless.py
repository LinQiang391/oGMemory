"""Tests that m3.index_episodic.build_embedder works with keyless local servers."""

from __future__ import annotations

from pathlib import Path

from m3.index_episodic import build_embedder


def test_build_embedder_uses_placeholder_key_when_yaml_key_is_empty(tmp_path: Path):
    cfg = tmp_path / "ogmemory.yaml"
    cfg.write_text(
        """
m3:
  embedding:
    provider: openai
    model: BAAI/bge-large-zh-v1.5
    base_url: http://127.0.0.1:8345/v1/
    api_key: ""
    dimension: 1024
""".strip(),
        encoding="utf-8",
    )
    from providers.embedder import get_openai_embedder
    OpenAIEmbedder, _ = get_openai_embedder()
    emb = build_embedder(
        None,
        model=None,
        base_url=None,
        api_key=None,
        dimension=None,
        config_path=str(cfg),
    )
    assert isinstance(emb, OpenAIEmbedder)
    assert emb._api_key == "no-key-required"
    assert emb._base_url == "http://127.0.0.1:8345/v1/"
    assert emb._dimension == 1024
