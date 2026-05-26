import inspect

from extraction.tools import Extractor


def test_extractor_no_longer_keeps_unreachable_span_structuring_code():
    source = inspect.getsource(Extractor)

    assert "def _structure_span(" not in source
    assert "Lazy mode: use ReAct loop" not in source
    assert "return self._structure_span_eager(span, prompt, ctx, focused)" not in source
