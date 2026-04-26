"""LangSmith adapter checks for the LLM contextual evaluator.

Mirrors `test_eval_evaluators_ir.py` in shape: the focus is the wrapper
contract (which fields are read from `run` and `example`, what feedback
shape is emitted, when to skip) — not prompt content. The LLM call is
stubbed via a fake `ChatOpenAI` injected into the lazy import path.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import pytest

from cosmere_rag.eval.evaluators_llm import (
    DEFAULT_LLM_METRICS,
    _Verdict,
    llm_contextual_evaluator,
)


class _FakeStructured:
    def __init__(self, scripted: list[_Verdict]) -> None:
        self._scripted = scripted
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> _Verdict:
        self.calls.append(messages)
        return self._scripted[len(self.calls) - 1]


class _FakeChatOpenAI:
    last_init: dict | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init = kwargs
        self._structured: _FakeStructured | None = None

    def with_structured_output(self, schema):  # noqa: ANN001
        assert schema is _Verdict
        self._structured = _FakeStructured(_FakeChatOpenAI._scripted)
        _FakeChatOpenAI.last_structured = self._structured
        return self._structured


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeChatOpenAI]]:
    """Replace `langchain_openai.ChatOpenAI` with the scripted fake.

    Tests assign `_FakeChatOpenAI._scripted` to the verdict sequence the
    judge should return on successive `.invoke()` calls.
    """
    module = ModuleType("langchain_openai")
    module.ChatOpenAI = _FakeChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    _FakeChatOpenAI.last_init = None
    _FakeChatOpenAI._scripted = []  # type: ignore[attr-defined]
    yield _FakeChatOpenAI


def _run(retrieved: list[str]) -> SimpleNamespace:
    return SimpleNamespace(outputs={"retrieved_texts": retrieved})


def _example(
    query: str = "who is Kelsier", expected: str = "a Mistborn crew leader"
) -> SimpleNamespace:
    return SimpleNamespace(
        inputs={"query": query},
        outputs={"expected_answer": expected},
    )


def test_returns_three_results_with_correct_keys(fake_llm) -> None:
    fake_llm._scripted = [
        _Verdict(score=0.9, reason="on-topic"),
        _Verdict(score=0.8, reason="relevant first"),
        _Verdict(score=0.7, reason="claims supported"),
    ]
    out = llm_contextual_evaluator()(_run(["chunk-a", "chunk-b"]), _example())
    assert out is not None
    keys = [r["key"] for r in out["results"]]
    assert keys == list(DEFAULT_LLM_METRICS)
    assert out["results"][0]["score"] == pytest.approx(0.9)
    assert out["results"][0]["comment"] == "on-topic"


def test_subset_of_metrics_param(fake_llm) -> None:
    fake_llm._scripted = [_Verdict(score=0.5, reason="meh")]
    out = llm_contextual_evaluator(metrics=("contextual_relevancy",))(
        _run(["c"]), _example()
    )
    assert out == {
        "results": [
            {"key": "contextual_relevancy", "score": 0.5, "comment": "meh"}
        ]
    }


def test_skips_when_no_query(fake_llm) -> None:
    example = SimpleNamespace(inputs={}, outputs={"expected_answer": "x"})
    assert llm_contextual_evaluator()(_run(["c"]), example) is None


def test_skips_when_no_expected_answer(fake_llm) -> None:
    example = SimpleNamespace(inputs={"query": "q"}, outputs={})
    assert llm_contextual_evaluator()(_run(["c"]), example) is None


def test_skips_when_no_retrieved_texts(fake_llm) -> None:
    assert llm_contextual_evaluator()(_run([]), _example()) is None


def test_handles_missing_outputs_dict(fake_llm) -> None:
    run = SimpleNamespace(outputs=None)
    example = SimpleNamespace(inputs=None, outputs=None)
    assert llm_contextual_evaluator()(run, example) is None


def test_unknown_metric_raises_at_factory_time() -> None:
    with pytest.raises(ValueError, match="unknown LLM metrics"):
        llm_contextual_evaluator(metrics=("contextual_relevancy", "bogus"))


def test_uses_judge_model_kwarg(fake_llm) -> None:
    fake_llm._scripted = [_Verdict(score=1.0, reason="r")] * 3
    llm_contextual_evaluator(judge_model="gpt-test-model")(
        _run(["c"]), _example()
    )
    assert fake_llm.last_init == {"model": "gpt-test-model", "temperature": 0}
