"""Tests for the LangChain / LangGraph integration.

langchain-core is not a test dependency, so these tests do two things:

1. Test the pure usage-extraction helper (``_extract_llm_usage``) directly
   against fake LangChain response shapes — no langchain install needed.
2. Test the callback behavior by monkeypatching the module's ``_HAS_LANGCHAIN``
   flag to ``True`` so ``LangChainBudgetCallback`` instantiates against its
   built-in stub ``BaseCallbackHandler``.

The fakes reproduce the response shapes LangChain actually emits:
- legacy ``LLMResult.llm_output["token_usage"]`` (completion models / older),
- modern chat models with ``message.usage_metadata`` (input_tokens/output_tokens),
- Anthropic-style ``message.response_metadata["usage"]``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from agentbudget import BudgetExhausted
from agentbudget.integrations.langchain import (
    LangChainBudgetCallback,
    _extract_llm_usage,
)


# ---------------------------------------------------------------------------
# Fake LangChain response objects
# ---------------------------------------------------------------------------

class FakeMessage:
    """Stand-in for langchain_core AIMessage."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class FakeChatGeneration:
    """Stand-in for ChatGeneration (has a `.message`)."""

    def __init__(self, message, text=""):
        self.text = text
        self.message = message


class FakeTextGeneration:
    """Stand-in for a plain Generation (no `.message`)."""

    def __init__(self, text=""):
        self.text = text


class FakeLLMResult:
    """Stand-in for langchain_core LLMResult."""

    def __init__(self, generations=None, llm_output=None):
        self.generations = generations or []
        self.llm_output = llm_output


def legacy_result(model="gpt-4o", prompt_tokens=1000, completion_tokens=500):
    """Old-style LLMResult: usage lives in llm_output['token_usage']."""
    return FakeLLMResult(
        generations=[[FakeTextGeneration(text="hello")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model,
        },
    )


def chat_usage_metadata_result(model="gpt-4o-mini", input_tokens=1000, output_tokens=500):
    """Modern chat model: usage lives in message.usage_metadata, model in response_metadata."""
    msg = FakeMessage(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model, "finish_reason": "stop"},
    )
    return FakeLLMResult(generations=[[FakeChatGeneration(message=msg)]], llm_output={})


def anthropic_response_metadata_result(model="claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500):
    """Anthropic-style: no usage_metadata, usage + model in response_metadata."""
    msg = FakeMessage(
        usage_metadata=None,
        response_metadata={
            "model": model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    return FakeLLMResult(generations=[[FakeChatGeneration(message=msg)]], llm_output=None)


# ---------------------------------------------------------------------------
# _extract_llm_usage — pure function, no langchain needed
# ---------------------------------------------------------------------------

class TestExtractLLMUsage:
    def test_legacy_llm_output_shape(self):
        model, inp, out = _extract_llm_usage(legacy_result("gpt-4o", 1200, 300))
        assert model == "gpt-4o"
        assert inp == 1200
        assert out == 300

    def test_modern_usage_metadata_shape(self):
        model, inp, out = _extract_llm_usage(
            chat_usage_metadata_result("gpt-4o-mini", 800, 200)
        )
        assert model == "gpt-4o-mini"
        assert inp == 800
        assert out == 200

    def test_anthropic_response_metadata_shape(self):
        model, inp, out = _extract_llm_usage(
            anthropic_response_metadata_result("claude-3-5-sonnet-20241022", 500, 100)
        )
        assert model == "claude-3-5-sonnet-20241022"
        assert inp == 500
        assert out == 100

    def test_aggregates_multiple_generations(self):
        m1 = FakeMessage(usage_metadata={"input_tokens": 100, "output_tokens": 50},
                         response_metadata={"model_name": "gpt-4o"})
        m2 = FakeMessage(usage_metadata={"input_tokens": 200, "output_tokens": 70},
                         response_metadata={"model_name": "gpt-4o"})
        result = FakeLLMResult(
            generations=[[FakeChatGeneration(m1)], [FakeChatGeneration(m2)]],
            llm_output={},
        )
        model, inp, out = _extract_llm_usage(result)
        assert model == "gpt-4o"
        assert inp == 300
        assert out == 120

    def test_empty_response_returns_none(self):
        assert _extract_llm_usage(FakeLLMResult()) == (None, None, None)

    def test_no_message_generation_returns_none(self):
        result = FakeLLMResult(generations=[[FakeTextGeneration("x")]], llm_output={})
        assert _extract_llm_usage(result) == (None, None, None)


# ---------------------------------------------------------------------------
# LangChainBudgetCallback behavior (requires _HAS_LANGCHAIN monkeypatched)
# ---------------------------------------------------------------------------

@pytest.fixture
def lc(monkeypatch):
    """Enable LangChainBudgetCallback against its stub base handler."""
    import agentbudget.integrations.langchain as lc_mod

    monkeypatch.setattr(lc_mod, "_HAS_LANGCHAIN", True)
    return lc_mod


class TestCallbackLLMTracking:
    def test_records_cost_from_usage_metadata(self, lc):
        """The core #26 bug: chat-model usage_metadata must be tracked, not silently $0."""
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        cb.on_llm_end(chat_usage_metadata_result("gpt-4o-mini", 1_000_000, 1_000_000))
        # gpt-4o-mini: $0.15/1M in + $0.60/1M out -> 0.15 + 0.60 = 0.75
        assert cb.session.spent == pytest.approx(0.75)

    def test_records_cost_from_legacy_shape(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        cb.on_llm_end(legacy_result("gpt-4o", 100_000, 100_000))
        # gpt-4o: $2.5/1M in + $10/1M out -> 0.25 + 1.0 = 1.25
        assert cb.session.spent == pytest.approx(1.25)

    def test_records_cost_from_anthropic_shape(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$50.00")
        cb.on_llm_end(anthropic_response_metadata_result("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000))
        # claude-3-5-sonnet: $3/1M in + $15/1M out -> 18.0
        assert cb.session.spent == pytest.approx(18.0)

    def test_unknown_model_records_nothing_and_warns(self, lc, caplog):
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        with caplog.at_level("WARNING"):
            cb.on_llm_end(chat_usage_metadata_result("totally-made-up-model", 1000, 500))
        assert cb.session.spent == 0.0
        assert any("pricing" in r.message.lower() for r in caplog.records)

    def test_missing_usage_records_nothing_and_warns(self, lc, caplog):
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        with caplog.at_level("WARNING"):
            cb.on_llm_end(FakeLLMResult())
        assert cb.session.spent == 0.0
        assert len(caplog.records) >= 1

    def test_budget_enforcement_raises(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$0.10")
        with pytest.raises(BudgetExhausted):
            # claude-3-5-sonnet 1M/1M = $18 >> $0.10
            cb.on_llm_end(anthropic_response_metadata_result("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000))


class TestCallbackToolTracking:
    def _start_end(self, cb, name, run_id):
        cb.on_tool_start({"name": name}, "input", run_id=run_id)
        cb.on_tool_end("output", run_id=run_id)

    def test_no_cost_map_tracks_nothing(self, lc):
        """Backward compatible: without configured costs, tools record nothing."""
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        rid = uuid4()
        self._start_end(cb, "search", rid)
        assert cb.session.spent == 0.0

    def test_tool_costs_are_tracked(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$5.00", tool_costs={"search": 0.25})
        rid = uuid4()
        self._start_end(cb, "search", rid)
        assert cb.session.spent == pytest.approx(0.25)
        by_tool = cb.get_report()["breakdown"]["tools"]["by_tool"]
        assert by_tool["search"] == pytest.approx(0.25)

    def test_default_tool_cost_applies(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$5.00", default_tool_cost=0.05)
        rid = uuid4()
        self._start_end(cb, "unmapped_tool", rid)
        assert cb.session.spent == pytest.approx(0.05)

    def test_tool_budget_enforcement(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$0.10", tool_costs={"expensive": 0.20})
        with pytest.raises(BudgetExhausted):
            self._start_end(cb, "expensive", uuid4())


class TestCallbackLifecycle:
    def test_context_manager_finalizes_session(self, lc):
        with lc.LangChainBudgetCallback(budget="$5.00") as cb:
            cb.on_llm_end(chat_usage_metadata_result("gpt-4o-mini", 1000, 500))
        report = cb.get_report()
        assert report["duration_seconds"] is not None

    def test_close_is_idempotent(self, lc):
        cb = lc.LangChainBudgetCallback(budget="$5.00")
        cb.close()
        cb.close()  # must not raise

    def test_on_hard_limit_fires_via_context_manager(self, lc):
        fired = []
        try:
            with lc.LangChainBudgetCallback(
                budget="$0.10", on_hard_limit=lambda r: fired.append(r)
            ) as cb:
                cb.on_llm_end(anthropic_response_metadata_result(
                    "claude-3-5-sonnet-20241022", 1_000_000, 1_000_000
                ))
        except BudgetExhausted:
            pass
        assert len(fired) == 1

    def test_on_soft_limit_forwarded(self, lc):
        warnings = []
        cb = lc.LangChainBudgetCallback(
            budget="$1.00", on_soft_limit=lambda r: warnings.append(r), tool_costs={"t": 0.95}
        )
        rid = uuid4()
        cb.on_tool_start({"name": "t"}, "in", run_id=rid)
        cb.on_tool_end("out", run_id=rid)
        assert len(warnings) == 1


class TestImportGuardUnchanged:
    def test_still_raises_without_langchain(self):
        """The default (no monkeypatch) path must still raise ImportError."""
        with pytest.raises(ImportError, match="langchain-core"):
            LangChainBudgetCallback(budget="$5.00")
