from types import SimpleNamespace

import pytest

from providers.lexical_index import factory


def _config(provider="opengauss_native", options=None):
    return SimpleNamespace(
        retrieval_lexical_provider=provider,
        retrieval_lexical_options=options or {},
        sql_connection_string="host=sql dbname=ogmemory",
        opengauss_connection_string="host=gauss dbname=ogmemory",
    )


def test_factory_routes_open_gauss_connection_and_options(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    factory.create_lexical_index(
        _config(
            options={
                "table_name": "lexical_custom",
                "bm25_topk": 25,
                "max_query_chars": 4096,
            }
        )
    )

    assert constructed["connection_string"] == "host=gauss dbname=ogmemory"
    assert constructed["table_name"] == "lexical_custom"
    assert constructed["bm25_topk"] == 25
    assert constructed["max_query_chars"] == 4096


def test_factory_allows_explicit_lexical_connection(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    factory.create_lexical_index(
        _config(options={"connection_string": "host=override dbname=ogmemory"})
    )

    assert constructed["connection_string"] == "host=override dbname=ogmemory"


def test_factory_falls_back_to_sql_connection(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    config = _config()
    config.opengauss_connection_string = None
    factory.create_lexical_index(config)

    assert constructed["connection_string"] == "host=sql dbname=ogmemory"


def test_factory_rejects_unrelated_provider():
    with pytest.raises(ValueError, match="only 'opengauss_native'"):
        factory.create_lexical_index(_config(provider="memory"))


def test_factory_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown openGauss lexical provider options"):
        factory.create_lexical_index(_config(options={"typo": True}))
