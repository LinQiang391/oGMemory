"""Lifecycle coverage for request-local candidate code retrieval."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from nltk.stem import PorterStemmer

from core.models import RequestContext
from retrieval.code_retrieval import ActiveCodeSearchRequest, CodeRetrievalService
from service.code_workspace_sync import ResolvedCodeWorkspace

_MOCK_STOPS = frozenset(
    "a an the and or but is are was were be been being have has had do does did "
    "will would could should may might shall can to of in for on with at by from "
    "as into through during before after above below between out off over under "
    "again further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very it its he she "
    "they them we you i me my his her our your this that these those what which who"
    .split()
)

_stemmer = PorterStemmer()


def _mock_tokenize(text: str) -> list[str]:
    """Simple regex tokenization + Porter stemming + stop filtering."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(t) for t in tokens if t not in _MOCK_STOPS]


@pytest.fixture(autouse=True)
def _patch_nltk():
    """Avoid network-dependent NLTK data downloads in lifecycle tests."""
    with patch("retrieval.bm25_index._ensure_nltk"), \
         patch("retrieval.bm25_index._tokenize", _mock_tokenize):
        yield


def test_candidate_local_rebuilds_from_current_files_without_retaining_old_evidence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "tracked.py"
    source.write_text(
        "def tracked_token():\n    return 'first revision'\n",
        encoding="utf-8",
    )

    class Resolver:
        def resolve(self, *, ctx, root_slugs):
            assert ctx.account_id == "acct"
            assert root_slugs == ()
            return ResolvedCodeWorkspace(
                root_id="root-1",
                root_slug="repo",
                root_fingerprint="f" * 64,
                physical_root=root,
                max_files=300,
                language_hints={},
            )

    service = CodeRetrievalService(
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_mode="candidate_local",
            code_retrieval_route_enabled=lambda route: route != "semantic",
        ),
        workspace_resolver=Resolver(),
    )
    ctx = RequestContext("acct", "user", "agent", "session", "trace")

    first = service.search_active(ActiveCodeSearchRequest("tracked_token"), ctx)

    assert first.evidence
    first_uris = {item.uri for item in first.evidence}
    assert service._cache == {}

    source.write_text(
        "def tracked_token():\n    return 'second revision'\n",
        encoding="utf-8",
    )
    second = service.search_active(ActiveCodeSearchRequest("tracked_token"), ctx)

    assert second.evidence
    second_uris = {item.uri for item in second.evidence}
    assert first_uris.isdisjoint(second_uris)
    assert service._cache == {}

    source.unlink()
    third = service.search_active(ActiveCodeSearchRequest("tracked_token"), ctx)

    assert third.status == "zero_result"
    assert not third.evidence
    assert service._cache == {}
