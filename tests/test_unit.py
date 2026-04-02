"""
Comprehensive unit tests for the AgentBudget Python SDK.

No network calls, no API keys required. All LLM responses are mocked
with realistic objects that match the actual OpenAI / Anthropic wire
formats (usage attributes, model names, etc.).

Run with:
    pytest tests/test_unit.py -v
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

import agentbudget
from agentbudget import (
    AgentBudget,
    AgentBudgetError,
    AsyncBudgetSession,
    BudgetExhausted,
    BudgetSession,
    InvalidBudget,
    LoopDetected,
    register_model,
    register_models,
)
from agentbudget.budget import parse_budget
from agentbudget.circuit_breaker import CircuitBreaker, LoopDetectorConfig
from agentbudget.ledger import Ledger
from agentbudget.pricing import (
    MODEL_PRICING,
    _custom_pricing,
    calculate_llm_cost,
    get_model_pricing,
)
from agentbudget.types import CostEvent, CostType


# ---------------------------------------------------------------------------
# Realistic mock response objects
# ---------------------------------------------------------------------------


class _FakeOpenAIUsage:
    """Mirrors openai.types.CompletionUsage fields."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeOpenAIChoice:
    """Mirrors openai.types.chat.ChatCompletionMessage."""

    def __init__(self, content: str = "Hello!") -> None:
        self.finish_reason = "stop"
        self.index = 0
        self.message = MagicMock(content=content, role="assistant")


class _FakeOpenAIResponse:
    """Realistic OpenAI ChatCompletion response object."""

    def __init__(
        self,
        model: str = "gpt-4o",
        prompt_tokens: int = 1_000,
        completion_tokens: int = 500,
        content: str = "Hello!",
    ) -> None:
        self.id = "chatcmpl-abc123"
        self.object = "chat.completion"
        self.created = 1_700_000_000
        self.model = model
        self.choices = [_FakeOpenAIChoice(content)]
        self.usage = _FakeOpenAIUsage(prompt_tokens, completion_tokens)


class _FakeAnthropicUsage:
    """Mirrors anthropic.types.Usage fields."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    """Realistic Anthropic Messages.create response object."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        input_tokens: int = 1_000,
        output_tokens: int = 500,
    ) -> None:
        self.id = "msg_01XFDUDYJgAACzvnptvVoYEL"
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.stop_reason = "end_turn"
        self.content = [MagicMock(type="text", text="Hello!")]
        self.usage = _FakeAnthropicUsage(input_tokens, output_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(budget: float = 5.0, **kwargs) -> BudgetSession:
    """Create a bare BudgetSession directly from a Ledger (no AgentBudget)."""
    ledger = Ledger(budget=budget)
    return BudgetSession(ledger=ledger, **kwargs)


def _make_budget_session(
    max_spend: Any = "$5.00", **budget_kwargs
) -> tuple[AgentBudget, BudgetSession]:
    budget = AgentBudget(max_spend=max_spend, **budget_kwargs)
    return budget, budget.session()


# ===========================================================================
# 1. Budget Parsing — parse_budget() / AgentBudget constructor
# ===========================================================================


class TestBudgetParsing:
    """parse_budget() must accept multiple input formats."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("$5.00", 5.0),
            ("$5", 5.0),
            ("5.00", 5.0),
            ("5", 5.0),
            (5.0, 5.0),
            (5, 5.0),
            ("$0.01", 0.01),
            ("  $10.50  ", 10.50),
            (100, 100.0),
            (0.001, 0.001),
        ],
    )
    def test_valid_formats(self, value, expected):
        result = parse_budget(value)
        assert abs(result - expected) < 1e-12

    @pytest.mark.parametrize(
        "value",
        [
            0,
            0.0,
            -1,
            -0.01,
            "$0",
            "0.00",
            "-5",
            "$-3.00",
            "abc",
            "$abc",
            "",
        ],
    )
    def test_invalid_raises_invalid_budget(self, value):
        with pytest.raises(InvalidBudget):
            parse_budget(value)

    def test_double_dollar_parses_as_valid(self):
        """
        "$$5" strips the leading '$', leaving "$5" which is a valid budget.
        This documents the parser's lstrip('$') behavior — it's not an error.
        """
        result = parse_budget("$$5")
        assert result == pytest.approx(5.0)

    def test_invalid_budget_is_agent_budget_error(self):
        with pytest.raises(AgentBudgetError):
            parse_budget("bad")

    def test_invalid_budget_stores_value(self):
        exc = None
        try:
            parse_budget("$bad")
        except InvalidBudget as e:
            exc = e
        assert exc is not None
        assert "$bad" in exc.value

    def test_agent_budget_dollar_string(self):
        budget = AgentBudget(max_spend="$3.50")
        assert budget.max_spend == 3.50

    def test_agent_budget_numeric_float(self):
        budget = AgentBudget(max_spend=2.5)
        assert budget.max_spend == 2.5

    def test_agent_budget_int(self):
        budget = AgentBudget(max_spend=10)
        assert budget.max_spend == 10.0

    def test_agent_budget_bare_string_float(self):
        budget = AgentBudget(max_spend="7.25")
        assert budget.max_spend == 7.25


# ===========================================================================
# 2. Finalization Reserve
# ===========================================================================


class TestFinalizationReserve:
    """finalization_reserve shrinks the effective enforcement budget."""

    def test_reserve_reduces_session_budget(self):
        # 5% reserve on $1.00 → session budget = $0.95
        budget = AgentBudget(max_spend=1.0, finalization_reserve=0.05)
        session = budget.session()
        # remaining should start at $0.95
        assert abs(session.remaining - 0.95) < 1e-10

    def test_max_spend_reflects_full_budget(self):
        budget = AgentBudget(max_spend=10.0, finalization_reserve=0.10)
        assert budget.max_spend == 10.0  # full, not reduced

    def test_zero_reserve_no_effect(self):
        budget = AgentBudget(max_spend=5.0, finalization_reserve=0.0)
        session = budget.session()
        assert abs(session.remaining - 5.0) < 1e-10

    def test_reserve_enforced_before_full_budget(self):
        # session budget is $0.90, so spending $0.91 should raise
        budget = AgentBudget(max_spend=1.0, finalization_reserve=0.10)
        with pytest.raises(BudgetExhausted):
            with budget.session() as session:
                session.track("x", cost=0.91)

    def test_invalid_reserve_raises_value_error(self):
        with pytest.raises(ValueError):
            AgentBudget(max_spend=5.0, finalization_reserve=1.0)

    def test_invalid_negative_reserve_raises_value_error(self):
        with pytest.raises(ValueError):
            AgentBudget(max_spend=5.0, finalization_reserve=-0.1)


# ===========================================================================
# 3. Ledger — core bookkeeping
# ===========================================================================


class TestLedger:
    def test_initial_state(self):
        ledger = Ledger(budget=5.0)
        assert ledger.budget == 5.0
        assert ledger.spent == 0.0
        assert ledger.remaining == 5.0

    def test_record_event_updates_spent(self):
        ledger = Ledger(budget=5.0)
        event = CostEvent(cost=1.0, cost_type=CostType.TOOL)
        ledger.record(event)
        assert ledger.spent == 1.0
        assert ledger.remaining == 4.0

    def test_record_multiple_events(self):
        ledger = Ledger(budget=5.0)
        for cost in [0.10, 0.25, 0.50]:
            ledger.record(CostEvent(cost=cost, cost_type=CostType.TOOL))
        assert abs(ledger.spent - 0.85) < 1e-12

    def test_record_raises_budget_exhausted(self):
        ledger = Ledger(budget=0.50)
        ledger.record(CostEvent(cost=0.40, cost_type=CostType.TOOL))
        with pytest.raises(BudgetExhausted) as exc_info:
            ledger.record(CostEvent(cost=0.20, cost_type=CostType.TOOL))
        assert exc_info.value.budget == 0.50

    def test_would_exceed_true(self):
        ledger = Ledger(budget=1.0)
        ledger.record(CostEvent(cost=0.80, cost_type=CostType.TOOL))
        assert ledger.would_exceed(0.21) is True

    def test_would_exceed_false(self):
        ledger = Ledger(budget=1.0)
        ledger.record(CostEvent(cost=0.80, cost_type=CostType.TOOL))
        assert ledger.would_exceed(0.20) is False  # exactly at limit, not over

    def test_would_exceed_exact_boundary(self):
        ledger = Ledger(budget=1.0)
        ledger.record(CostEvent(cost=0.80, cost_type=CostType.TOOL))
        # 0.80 + 0.20 == 1.0, not > 1.0
        assert ledger.would_exceed(0.20) is False

    def test_breakdown_llm(self):
        ledger = Ledger(budget=5.0)
        ledger.record(
            CostEvent(cost=0.50, cost_type=CostType.LLM, model="gpt-4o")
        )
        bd = ledger.breakdown()
        assert bd["llm"]["calls"] == 1
        assert abs(bd["llm"]["total"] - 0.50) < 1e-10
        assert "gpt-4o" in bd["llm"]["by_model"]

    def test_breakdown_tools(self):
        ledger = Ledger(budget=5.0)
        ledger.record(CostEvent(cost=0.10, cost_type=CostType.TOOL, tool_name="serp"))
        ledger.record(CostEvent(cost=0.20, cost_type=CostType.TOOL, tool_name="scrape"))
        bd = ledger.breakdown()
        assert bd["tools"]["calls"] == 2
        assert abs(bd["tools"]["total"] - 0.30) < 1e-10
        assert bd["tools"]["by_tool"]["serp"] == 0.10
        assert bd["tools"]["by_tool"]["scrape"] == 0.20

    def test_thread_safety(self):
        """Concurrent records must not corrupt the total."""
        ledger = Ledger(budget=1000.0)
        errors: list[Exception] = []

        def record_many():
            try:
                for _ in range(50):
                    ledger.record(CostEvent(cost=0.01, cost_type=CostType.TOOL))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert abs(ledger.spent - 5.0) < 1e-10  # 10 * 50 * 0.01


# ===========================================================================
# 4. BudgetSession — wrap() / track() / would_exceed()
# ===========================================================================


class TestBudgetSessionWrap:
    """session.wrap() extracts usage from response objects."""

    def test_wrap_openai_response_records_cost(self):
        session = _make_session(budget=5.0)
        response = _FakeOpenAIResponse("gpt-4o", prompt_tokens=1_000, completion_tokens=500)
        with session:
            result = session.wrap(response)
        assert result is response
        # gpt-4o: $2.50/1M input, $10/1M output
        expected = (1_000 * 2.50 / 1_000_000) + (500 * 10.0 / 1_000_000)
        assert abs(session.spent - expected) < 1e-12

    def test_wrap_anthropic_response_records_cost(self):
        session = _make_session(budget=5.0)
        response = _FakeAnthropicResponse(
            "claude-3-5-sonnet-20241022", input_tokens=1_000, output_tokens=500
        )
        with session:
            session.wrap(response)
        # $3/1M input, $15/1M output
        expected = (1_000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        assert abs(session.spent - expected) < 1e-12

    def test_wrap_unknown_model_skips_cost(self):
        session = _make_session(budget=5.0)
        response = _FakeOpenAIResponse("nonexistent-model-v99")
        with session:
            session.wrap(response)
        assert session.spent == 0.0

    def test_wrap_no_usage_attribute(self):
        session = _make_session(budget=5.0)
        with session:
            session.wrap("just a string")
        assert session.spent == 0.0

    def test_wrap_passes_through_response_unchanged(self):
        session = _make_session(budget=5.0)
        response = _FakeOpenAIResponse("gpt-4o-mini")
        with session:
            result = session.wrap(response)
        assert result is response

    def test_wrap_gpt4o_mini_cost(self):
        session = _make_session(budget=5.0)
        response = _FakeOpenAIResponse("gpt-4o-mini", prompt_tokens=10_000, completion_tokens=2_000)
        with session:
            session.wrap(response)
        # $0.15/1M input, $0.60/1M output
        expected = (10_000 * 0.15 / 1_000_000) + (2_000 * 0.60 / 1_000_000)
        assert abs(session.spent - expected) < 1e-12

    def test_wrap_accumulates_multiple_calls(self):
        session = _make_session(budget=5.0)
        r1 = _FakeOpenAIResponse("gpt-4o-mini", prompt_tokens=1_000, completion_tokens=200)
        r2 = _FakeOpenAIResponse("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
        with session:
            session.wrap(r1)
            session.wrap(r2)
        cost1 = (1_000 * 0.15 / 1_000_000) + (200 * 0.60 / 1_000_000)
        cost2 = (500 * 0.15 / 1_000_000) + (100 * 0.60 / 1_000_000)
        assert abs(session.spent - (cost1 + cost2)) < 1e-12


class TestBudgetSessionTrack:
    """session.track() for tool / external API costs."""

    def test_track_returns_result(self):
        session = _make_session()
        with session:
            data = {"key": "value"}
            result = session.track(data, cost=0.05, tool_name="my_api")
        assert result is data

    def test_track_records_cost(self):
        session = _make_session()
        with session:
            session.track("x", cost=0.10, tool_name="serp")
        assert abs(session.spent - 0.10) < 1e-12

    def test_track_with_metadata(self):
        session = _make_session()
        meta = {"query": "hello world", "hits": 10}
        with session:
            session.track("result", cost=0.01, tool_name="search", metadata=meta)
        report = session.report()
        event = report["events"][0]
        assert event["metadata"] == meta

    def test_track_accumulates(self):
        session = _make_session()
        with session:
            session.track(None, cost=0.10)
            session.track(None, cost=0.20)
            session.track(None, cost=0.30)
        assert abs(session.spent - 0.60) < 1e-12

    def test_track_raises_budget_exhausted_when_over(self):
        session = _make_session(budget=0.15)
        with pytest.raises(BudgetExhausted):
            with session:
                session.track(None, cost=0.10)
                session.track(None, cost=0.10)  # total $0.20 > $0.15


# ===========================================================================
# 5. Hard limit raises BudgetExhausted
# ===========================================================================


class TestHardLimit:
    def test_hard_limit_raises_on_wrap(self):
        budget = AgentBudget(max_spend=0.001)  # $0.001
        with pytest.raises(BudgetExhausted):
            with budget.session() as session:
                # gpt-4o at 1000 tokens is way over $0.001
                resp = _FakeOpenAIResponse("gpt-4o", prompt_tokens=1_000, completion_tokens=500)
                session.wrap(resp)

    def test_hard_limit_raises_on_track(self):
        budget = AgentBudget(max_spend=0.05)
        with pytest.raises(BudgetExhausted):
            with budget.session() as session:
                session.track(None, cost=0.06)

    def test_budget_exhausted_exception_fields(self):
        budget = AgentBudget(max_spend=0.10)
        exc: Optional[BudgetExhausted] = None
        try:
            with budget.session() as session:
                session.track(None, cost=0.15)
        except BudgetExhausted as e:
            exc = e
        assert exc is not None
        assert exc.budget == pytest.approx(0.10, abs=1e-10)
        assert exc.spent == pytest.approx(0.15, abs=1e-10)
        assert "Budget exhausted" in str(exc)

    def test_budget_exhausted_is_agent_budget_error(self):
        with pytest.raises(AgentBudgetError):
            with AgentBudget(max_spend=0.01).session() as session:
                session.track(None, cost=100.0)

    def test_on_hard_limit_callback_fires(self):
        reports: list[dict] = []
        budget = AgentBudget(
            max_spend=0.05,
            on_hard_limit=lambda r: reports.append(r),
        )
        try:
            with budget.session() as session:
                session.track(None, cost=0.10)
        except BudgetExhausted:
            pass
        assert len(reports) == 1
        assert reports[0]["terminated_by"] == "budget_exhausted"

    def test_on_hard_limit_callback_fires_exactly_once(self):
        calls: list[int] = []
        budget = AgentBudget(
            max_spend=0.05,
            on_hard_limit=lambda _: calls.append(1),
        )
        try:
            with budget.session() as session:
                session.track(None, cost=0.10)
        except BudgetExhausted:
            pass
        assert len(calls) == 1


# ===========================================================================
# 6. Soft limit callback fires once at threshold
# ===========================================================================


class TestSoftLimit:
    def test_soft_limit_callback_fires(self):
        warnings: list[dict] = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.80,
            on_soft_limit=lambda r: warnings.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.85)  # 85% > 80% threshold
        assert len(warnings) == 1

    def test_soft_limit_callback_fires_at_threshold(self):
        warnings: list[dict] = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.50,
            on_soft_limit=lambda r: warnings.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.50)  # exactly 50%
        assert len(warnings) == 1

    def test_soft_limit_fires_only_once(self):
        """Even with many calls that each exceed the threshold, callback fires once."""
        warnings: list[dict] = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.30,
            on_soft_limit=lambda r: warnings.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.10)  # 10% — below threshold
            session.track(None, cost=0.10)  # 20% — below threshold
            session.track(None, cost=0.10)  # 30% — threshold, fires
            session.track(None, cost=0.10)  # 40% — already fired, skip
            session.track(None, cost=0.10)  # 50% — already fired, skip
        assert len(warnings) == 1

    def test_soft_limit_callback_receives_report(self):
        warnings: list[dict] = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.90,
            on_soft_limit=lambda r: warnings.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.95)
        report = warnings[0]
        assert "session_id" in report
        assert "total_spent" in report
        assert "remaining" in report
        assert "budget" in report

    def test_no_callback_below_threshold(self):
        warnings: list[dict] = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.90,
            on_soft_limit=lambda r: warnings.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.50)  # only 50%, below 90%
        assert len(warnings) == 0


# ===========================================================================
# 7. Loop detection raises LoopDetected
# ===========================================================================


class TestLoopDetection:
    def test_loop_raises_after_max_calls(self):
        budget = AgentBudget(max_spend="$50.00", max_repeated_calls=5)
        with pytest.raises(LoopDetected):
            with budget.session() as session:
                for _ in range(10):
                    session.track(None, cost=0.01, tool_name="loopy_tool")

    def test_loop_detected_stores_key(self):
        budget = AgentBudget(max_spend="$50.00", max_repeated_calls=3)
        exc: Optional[LoopDetected] = None
        try:
            with budget.session() as session:
                for _ in range(5):
                    session.track(None, cost=0.01, tool_name="bad_tool")
        except LoopDetected as e:
            exc = e
        assert exc is not None
        assert exc.key == "bad_tool"
        assert "bad_tool" in str(exc)

    def test_loop_detected_terminates_session(self):
        budget = AgentBudget(max_spend="$50.00", max_repeated_calls=3)
        try:
            with budget.session() as session:
                for _ in range(5):
                    session.track(None, cost=0.01, tool_name="stuck")
        except LoopDetected:
            pass
        assert session.report()["terminated_by"] == "loop_detected"

    def test_different_tools_do_not_loop(self):
        """Different tool names should not trigger each other's loop counter."""
        budget = AgentBudget(max_spend="$50.00", max_repeated_calls=5)
        with budget.session() as session:
            for i in range(20):
                session.track(None, cost=0.01, tool_name=f"tool_{i % 6}")
        # Should not raise — 6 different tool names, each called < 5 times in window

    def test_on_loop_detected_callback_fires(self):
        reports: list[dict] = []
        budget = AgentBudget(
            max_spend="$50.00",
            max_repeated_calls=3,
            on_loop_detected=lambda r: reports.append(r),
        )
        try:
            with budget.session() as session:
                for _ in range(5):
                    session.track(None, cost=0.01, tool_name="stuck")
        except LoopDetected:
            pass
        assert len(reports) >= 1

    def test_loop_on_llm_model_repeated_calls(self):
        """Repeated calls to the same LLM model should also trigger loop detection."""
        budget = AgentBudget(max_spend="$100.00", max_repeated_calls=4)
        with pytest.raises(LoopDetected):
            with budget.session() as session:
                for _ in range(6):
                    resp = _FakeOpenAIResponse(
                        "gpt-4o-mini", prompt_tokens=100, completion_tokens=50
                    )
                    session.wrap(resp)

    def test_loop_detected_is_exception(self):
        exc = LoopDetected("test_key")
        assert isinstance(exc, Exception)
        assert exc.key == "test_key"


# ===========================================================================
# 8. would_exceed() pre-flight check
# ===========================================================================


class TestWouldExceed:
    def test_would_exceed_true_when_over(self):
        session = _make_session(budget=1.0)
        with session:
            session.track(None, cost=0.90)
            assert session.would_exceed(0.15) is True

    def test_would_exceed_false_when_under(self):
        session = _make_session(budget=1.0)
        with session:
            session.track(None, cost=0.70)
            assert session.would_exceed(0.20) is False

    def test_would_exceed_exact_boundary_not_exceeded(self):
        session = _make_session(budget=1.0)
        with session:
            session.track(None, cost=0.80)
            # 0.80 + 0.20 == 1.0 — not strictly greater than, so False
            assert session.would_exceed(0.20) is False

    def test_would_exceed_does_not_record_cost(self):
        session = _make_session(budget=1.0)
        with session:
            session.track(None, cost=0.50)
            session.would_exceed(0.99)  # pre-flight, should not record
        assert abs(session.spent - 0.50) < 1e-12

    def test_would_exceed_fresh_session(self):
        session = _make_session(budget=1.0)
        with session:
            # Nothing spent yet
            assert session.would_exceed(0.50) is False
            assert session.would_exceed(1.01) is True


# ===========================================================================
# 9. Child sessions cap to parent remaining
# ===========================================================================


class TestChildSessions:
    def test_child_capped_at_max_spend(self):
        session = _make_session(budget=5.0)
        with session:
            child = session.child_session(max_spend=1.0)
            # Child budget should be min(1.0, remaining=5.0) = 1.0
            assert child.remaining == pytest.approx(1.0, abs=1e-10)

    def test_child_capped_at_parent_remaining(self):
        session = _make_session(budget=2.0)
        with session:
            session.track(None, cost=1.80)  # parent has $0.20 left
            child = session.child_session(max_spend=1.0)
            # Child budget = min(1.0, 0.20) = 0.20
            assert child.remaining == pytest.approx(0.20, abs=1e-10)

    def test_child_spend_rolls_up_to_parent(self):
        session = _make_session(budget=5.0)
        with session:
            child = session.child_session(max_spend=1.0)
            with child:
                child.track(None, cost=0.50, tool_name="sub_task")
        # Parent should now reflect child's $0.50 rolled up
        assert abs(session.spent - 0.50) < 1e-10

    def test_child_context_manager(self):
        session = _make_session(budget=5.0)
        with session:
            child = session.child_session(max_spend=2.0)
            with child:
                child.track(None, cost=0.75)
            assert abs(session.spent - 0.75) < 1e-10

    def test_child_cannot_exceed_its_own_budget(self):
        session = _make_session(budget=10.0)
        with pytest.raises(BudgetExhausted):
            with session:
                child = session.child_session(max_spend=0.50)
                with child:
                    child.track(None, cost=0.60)  # exceeds child budget

    def test_child_spend_does_not_double_count(self):
        """Child spend charges parent exactly once via __exit__."""
        session = _make_session(budget=5.0)
        with session:
            child = session.child_session(max_spend=1.0)
            with child:
                child.track(None, cost=0.30, tool_name="sub")
                child.track(None, cost=0.20, tool_name="sub")
        assert abs(session.spent - 0.50) < 1e-10

    def test_nested_child_sessions(self):
        """Multiple sibling children should each roll up independently."""
        session = _make_session(budget=5.0)
        with session:
            child_a = session.child_session(max_spend=1.0)
            child_b = session.child_session(max_spend=1.0)
            with child_a:
                child_a.track(None, cost=0.40)
            with child_b:
                child_b.track(None, cost=0.30)
        assert abs(session.spent - 0.70) < 1e-10


# ===========================================================================
# 10. report() structure validation
# ===========================================================================


class TestReportStructure:
    def test_report_has_required_keys(self):
        session = _make_session(budget=5.0, session_id="sess_test001")
        with session:
            session.track("x", cost=0.25, tool_name="api")
        r = session.report()
        required_keys = {
            "session_id",
            "budget",
            "total_spent",
            "remaining",
            "breakdown",
            "duration_seconds",
            "terminated_by",
            "events",
        }
        assert required_keys.issubset(r.keys())

    def test_report_session_id_matches(self):
        session = _make_session(budget=5.0, session_id="sess_abc123")
        with session:
            pass
        assert session.report()["session_id"] == "sess_abc123"

    def test_report_budget_correct(self):
        session = _make_session(budget=3.75)
        with session:
            pass
        assert session.report()["budget"] == 3.75

    def test_report_total_spent_and_remaining_sum_to_budget(self):
        session = _make_session(budget=5.0)
        with session:
            session.track(None, cost=1.23)
        r = session.report()
        assert abs(r["total_spent"] + r["remaining"] - 5.0) < 1e-4

    def test_report_duration_positive(self):
        session = _make_session(budget=5.0)
        with session:
            time.sleep(0.01)
        r = session.report()
        assert r["duration_seconds"] is not None
        assert r["duration_seconds"] >= 0.0

    def test_report_terminated_by_none_on_success(self):
        session = _make_session(budget=5.0)
        with session:
            session.track(None, cost=0.50)
        assert session.report()["terminated_by"] is None

    def test_report_terminated_by_budget_exhausted(self):
        session = _make_session(budget=0.10)
        try:
            with session:
                session.track(None, cost=0.20)
        except BudgetExhausted:
            pass
        assert session.report()["terminated_by"] == "budget_exhausted"

    def test_report_events_list(self):
        session = _make_session(budget=5.0)
        with session:
            session.track("a", cost=0.10, tool_name="t1")
            session.track("b", cost=0.20, tool_name="t2")
        events = session.report()["events"]
        assert len(events) == 2
        assert all("cost" in e for e in events)
        assert all("cost_type" in e for e in events)
        assert all("timestamp" in e for e in events)

    def test_report_breakdown_structure(self):
        session = _make_session(budget=5.0)
        with session:
            resp = _FakeOpenAIResponse("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
            session.wrap(resp)
            session.track(None, cost=0.05, tool_name="search")
        r = session.report()["breakdown"]
        assert "llm" in r
        assert "tools" in r
        assert r["llm"]["calls"] == 1
        assert r["tools"]["calls"] == 1
        assert "by_model" in r["llm"]
        assert "by_tool" in r["tools"]

    def test_report_is_json_serializable(self):
        import json
        session = _make_session(budget=5.0)
        with session:
            session.track(None, cost=0.50)
        import json
        json_str = json.dumps(session.report())
        parsed = json.loads(json_str)
        assert "session_id" in parsed

    def test_auto_generated_session_id(self):
        session = _make_session(budget=5.0)
        with session:
            pass
        sid = session.report()["session_id"]
        assert sid.startswith("sess_")
        assert len(sid) > 5


# ===========================================================================
# 11. Custom model registration and pricing
# ===========================================================================


class TestCustomModelRegistration:
    def setup_method(self):
        """Clear custom pricing before each test."""
        _custom_pricing.clear()

    def teardown_method(self):
        """Clean up after each test."""
        _custom_pricing.clear()

    def test_register_model_basic(self):
        register_model("gpt-9000", input_price_per_million=5.0, output_price_per_million=20.0)
        pricing = get_model_pricing("gpt-9000")
        assert pricing is not None
        assert abs(pricing[0] - 5.0 / 1_000_000) < 1e-15
        assert abs(pricing[1] - 20.0 / 1_000_000) < 1e-15

    def test_register_model_overrides_builtin(self):
        """Custom pricing should take precedence over built-in."""
        # Override gpt-4o pricing
        register_model("gpt-4o", input_price_per_million=1.0, output_price_per_million=2.0)
        pricing = get_model_pricing("gpt-4o")
        assert pricing[0] == pytest.approx(1.0 / 1_000_000)
        assert pricing[1] == pytest.approx(2.0 / 1_000_000)

    def test_register_models_bulk(self):
        register_models({
            "my-model-a": (1.0, 3.0),
            "my-model-b": (2.0, 6.0),
        })
        pa = get_model_pricing("my-model-a")
        pb = get_model_pricing("my-model-b")
        assert pa[0] == pytest.approx(1.0 / 1_000_000)
        assert pb[1] == pytest.approx(6.0 / 1_000_000)

    def test_custom_model_used_in_session(self):
        register_model("acme-llm-v1", input_price_per_million=10.0, output_price_per_million=30.0)
        session = _make_session(budget=5.0)
        response = _FakeOpenAIResponse("acme-llm-v1", prompt_tokens=1_000, completion_tokens=500)
        with session:
            session.wrap(response)
        expected = (1_000 * 10.0 / 1_000_000) + (500 * 30.0 / 1_000_000)
        assert abs(session.spent - expected) < 1e-12

    def test_register_model_negative_price_raises(self):
        with pytest.raises(ValueError):
            register_model("bad-model", input_price_per_million=-1.0, output_price_per_million=5.0)

    def test_register_model_zero_price_allowed(self):
        """Zero-cost models (e.g. local models) are valid."""
        register_model("local-llm", input_price_per_million=0.0, output_price_per_million=0.0)
        pricing = get_model_pricing("local-llm")
        assert pricing == (0.0, 0.0)

    def test_unknown_model_returns_none(self):
        pricing = get_model_pricing("does-not-exist-xyz")
        assert pricing is None

    def test_calculate_llm_cost_unknown_model_returns_none(self):
        result = calculate_llm_cost("does-not-exist", 1000, 500)
        assert result is None

    def test_fuzzy_match_dated_variant(self):
        """'gpt-4o-2025-01-01' should fuzzy-match to 'gpt-4o'."""
        pricing = get_model_pricing("gpt-4o-2025-01-01")
        assert pricing is not None
        assert pricing == MODEL_PRICING["gpt-4o"]

    def test_openrouter_prefix_stripped(self):
        """'openai/gpt-4o' should resolve to gpt-4o pricing."""
        pricing = get_model_pricing("openai/gpt-4o")
        assert pricing is not None
        assert pricing == MODEL_PRICING["gpt-4o"]

    def test_openrouter_custom_model(self):
        """'myorg/my-model' resolves after prefix strip + custom lookup."""
        register_model("my-model", input_price_per_million=3.0, output_price_per_million=9.0)
        pricing = get_model_pricing("myorg/my-model")
        assert pricing is not None
        assert pricing[0] == pytest.approx(3.0 / 1_000_000)


# ===========================================================================
# 12. track() for tool costs
# ===========================================================================


class TestTrackMethod:
    def test_track_pass_through(self):
        session = _make_session()
        with session:
            sentinel = object()
            result = session.track(sentinel, cost=0.01)
        assert result is sentinel

    def test_track_tool_name_appears_in_breakdown(self):
        session = _make_session()
        with session:
            session.track(None, cost=0.05, tool_name="web_search")
        bd = session.report()["breakdown"]["tools"]
        assert "web_search" in bd["by_tool"]
        assert bd["by_tool"]["web_search"] == pytest.approx(0.05)

    def test_track_without_tool_name(self):
        """tool_name=None should still record cost."""
        session = _make_session()
        with session:
            session.track(None, cost=0.03)
        assert session.spent == pytest.approx(0.03)

    def test_track_zero_cost(self):
        """Zero cost should record without error."""
        session = _make_session()
        with session:
            session.track(None, cost=0.0)
        assert session.spent == 0.0

    def test_track_event_in_report(self):
        session = _make_session()
        with session:
            session.track("result", cost=0.07, tool_name="pricing_api")
        events = session.report()["events"]
        assert len(events) == 1
        e = events[0]
        assert e["cost_type"] == "tool"
        assert e["tool_name"] == "pricing_api"
        assert e["cost"] == pytest.approx(0.07)


# ===========================================================================
# 13. track_tool() decorator
# ===========================================================================


class TestTrackToolDecorator:
    def test_sync_decorator_basic(self):
        session = _make_session()
        with session:
            @session.track_tool(cost=0.02, tool_name="search")
            def search(query: str) -> list:
                return [query, "result"]

            result = search("hello")
        assert result == ["hello", "result"]
        assert session.spent == pytest.approx(0.02)

    def test_sync_decorator_uses_function_name_by_default(self):
        session = _make_session()
        with session:
            @session.track_tool(cost=0.01)
            def my_api_call():
                return "data"

            my_api_call()
        bd = session.report()["breakdown"]["tools"]["by_tool"]
        assert "my_api_call" in bd

    def test_sync_decorator_tracks_each_call(self):
        session = _make_session()
        with session:
            @session.track_tool(cost=0.10, tool_name="t")
            def work():
                return "ok"

            work()
            work()
            work()
        assert session.spent == pytest.approx(0.30)

    def test_sync_decorator_preserves_return_value(self):
        session = _make_session()
        with session:
            @session.track_tool(cost=0.05)
            def compute(x: int, y: int) -> int:
                return x + y

            assert compute(3, 4) == 7

    def test_sync_decorator_preserves_function_metadata(self):
        session = _make_session()

        @session.track_tool(cost=0.01, tool_name="named_fn")
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_sync_decorator_raises_budget_exhausted(self):
        session = _make_session(budget=0.05)
        with pytest.raises(BudgetExhausted):
            with session:
                @session.track_tool(cost=0.03, tool_name="expensive")
                def call():
                    return "ok"

                call()
                call()  # pushes over $0.05


# ===========================================================================
# 14. Context manager (__enter__ / __exit__)
# ===========================================================================


class TestContextManager:
    def test_enter_returns_session(self):
        session = _make_session(budget=5.0)
        result = session.__enter__()
        assert result is session
        session.__exit__(None, None, None)

    def test_with_statement_basic(self):
        session = _make_session(budget=5.0)
        with session as s:
            assert s is session

    def test_duration_recorded_on_exit(self):
        session = _make_session(budget=5.0)
        with session:
            time.sleep(0.01)
        r = session.report()
        assert r["duration_seconds"] is not None
        assert r["duration_seconds"] >= 0.01

    def test_exit_without_error_sets_end_time(self):
        session = _make_session(budget=5.0)
        with session:
            pass
        assert session._end_time is not None

    def test_exception_propagates_through_context_manager(self):
        session = _make_session(budget=5.0)
        with pytest.raises(RuntimeError):
            with session:
                raise RuntimeError("boom")

    def test_budget_exhausted_terminates_session(self):
        session = _make_session(budget=0.01)
        with pytest.raises(BudgetExhausted):
            with session:
                session.track(None, cost=0.99)
        assert session.report()["terminated_by"] == "budget_exhausted"

    def test_nested_context_managers(self):
        """Multiple independent sessions can be open simultaneously."""
        s1 = _make_session(budget=5.0, session_id="sess_a")
        s2 = _make_session(budget=10.0, session_id="sess_b")
        with s1:
            with s2:
                s1.track(None, cost=0.10)
                s2.track(None, cost=0.20)
        assert s1.spent == pytest.approx(0.10)
        assert s2.spent == pytest.approx(0.20)


# ===========================================================================
# 15. AsyncBudgetSession
# ===========================================================================


class TestAsyncBudgetSession:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        ledger = Ledger(budget=5.0)
        async with AsyncBudgetSession(ledger) as session:
            assert session.spent == 0.0

    @pytest.mark.asyncio
    async def test_async_wrap(self):
        ledger = Ledger(budget=5.0)
        async with AsyncBudgetSession(ledger) as session:
            resp = _FakeOpenAIResponse("gpt-4o", prompt_tokens=1_000, completion_tokens=500)
            result = session.wrap(resp)
        assert result is resp
        assert session.spent > 0

    @pytest.mark.asyncio
    async def test_async_track(self):
        ledger = Ledger(budget=5.0)
        async with AsyncBudgetSession(ledger) as session:
            session.track("data", cost=0.05, tool_name="my_tool")
        assert session.spent == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_async_track_tool_decorator_sync_func(self):
        ledger = Ledger(budget=5.0)
        async with AsyncBudgetSession(ledger) as session:
            @session.track_tool(cost=0.02, tool_name="sync_fn")
            def sync_work():
                return "done"

            result = sync_work()
        assert result == "done"
        assert session.spent == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_async_track_tool_decorator_async_func(self):
        ledger = Ledger(budget=5.0)
        async with AsyncBudgetSession(ledger) as session:
            @session.track_tool(cost=0.03, tool_name="async_fn")
            async def async_work():
                return "async_done"

            result = await async_work()
        assert result == "async_done"
        assert session.spent == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_wrap_async_coroutine(self):
        import asyncio
        ledger = Ledger(budget=5.0)
        resp = _FakeOpenAIResponse("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)

        async def fake_coroutine():
            return resp

        async with AsyncBudgetSession(ledger) as session:
            result = await session.wrap_async(fake_coroutine())
        assert result is resp
        assert session.spent > 0


# ===========================================================================
# 16. Global API (init / teardown / spent / remaining / report / track)
# ===========================================================================


class TestGlobalAPI:
    def setup_method(self):
        """Ensure clean state before each test."""
        agentbudget.teardown()

    def teardown_method(self):
        """Clean up after each test."""
        agentbudget.teardown()

    def test_init_returns_session(self):
        session = agentbudget.init(budget="$5.00")
        assert isinstance(session, BudgetSession)

    def test_spent_returns_zero_before_init(self):
        # teardown has been called, no session active
        assert agentbudget.spent() == 0.0

    def test_remaining_returns_zero_before_init(self):
        assert agentbudget.remaining() == 0.0

    def test_report_returns_none_before_init(self):
        assert agentbudget.report() is None

    def test_track_raises_when_not_initialized(self):
        with pytest.raises(RuntimeError, match="init"):
            agentbudget.track(cost=0.01)

    def test_init_then_track(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(result="x", cost=0.10, tool_name="test_api")
        assert agentbudget.spent() == pytest.approx(0.10)

    def test_remaining_decreases_after_track(self):
        agentbudget.init(budget="$1.00")
        agentbudget.track(cost=0.25)
        assert agentbudget.remaining() == pytest.approx(0.75)

    def test_teardown_returns_report(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=0.50)
        r = agentbudget.teardown()
        assert r is not None
        assert r["total_spent"] == pytest.approx(0.50)

    def test_teardown_resets_state(self):
        agentbudget.init(budget="$5.00")
        agentbudget.teardown()
        assert agentbudget.spent() == 0.0
        assert agentbudget.get_session() is None

    def test_double_init_replaces_session(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=0.10)
        # Second init should tear down first and start fresh
        agentbudget.init(budget="$10.00")
        assert agentbudget.spent() == 0.0

    def test_report_structure_from_global(self):
        agentbudget.init(budget="$5.00")
        agentbudget.track(cost=0.30, tool_name="test")
        r = agentbudget.report()
        assert r is not None
        assert "session_id" in r
        assert "budget" in r

    def test_get_session_returns_active_session(self):
        session = agentbudget.init(budget="$2.00")
        assert agentbudget.get_session() is session

    def test_init_with_custom_session_id(self):
        agentbudget.init(budget="$5.00", session_id="sess_custom_001")
        r = agentbudget.report()
        assert r["session_id"] == "sess_custom_001"


# ===========================================================================
# 17. AgentBudget constructor edge cases
# ===========================================================================


class TestAgentBudgetConstructor:
    def test_session_has_correct_budget(self):
        budget = AgentBudget(max_spend="$3.00")
        session = budget.session()
        assert session.remaining == pytest.approx(3.0)

    def test_session_id_injected(self):
        budget = AgentBudget(max_spend="$5.00")
        session = budget.session(session_id="sess_injected")
        assert session.session_id == "sess_injected"

    def test_soft_limit_custom_fraction(self):
        warns = []
        budget = AgentBudget(
            max_spend="$1.00",
            soft_limit=0.5,
            on_soft_limit=lambda r: warns.append(r),
        )
        with budget.session() as session:
            session.track(None, cost=0.51)
        assert len(warns) == 1

    def test_multiple_sessions_are_independent(self):
        budget = AgentBudget(max_spend="$1.00")
        s1 = budget.session()
        s2 = budget.session()
        with s1:
            s1.track(None, cost=0.30)
        with s2:
            s2.track(None, cost=0.50)
        assert s1.spent == pytest.approx(0.30)
        assert s2.spent == pytest.approx(0.50)

    def test_async_session_returned(self):
        budget = AgentBudget(max_spend="$5.00")
        async_session = budget.async_session()
        assert isinstance(async_session, AsyncBudgetSession)

    def test_max_spend_property(self):
        budget = AgentBudget(max_spend="$7.50")
        assert budget.max_spend == 7.50


# ===========================================================================
# 18. Pricing calculations
# ===========================================================================


class TestPricingCalculations:
    def test_gpt4o_cost(self):
        cost = calculate_llm_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(2.50, abs=1e-6)

    def test_gpt4o_output_cost(self):
        cost = calculate_llm_cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(10.00, abs=1e-6)

    def test_gpt4o_mini_cost(self):
        cost = calculate_llm_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.75, abs=1e-6)

    def test_claude_sonnet_cost(self):
        cost = calculate_llm_cost(
            "claude-3-5-sonnet-20241022", input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert cost == pytest.approx(18.0, abs=1e-6)

    def test_unknown_model_returns_none(self):
        assert calculate_llm_cost("future-model-x", 1000, 500) is None

    def test_zero_tokens_zero_cost(self):
        cost = calculate_llm_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == pytest.approx(0.0)

    def test_builtin_models_are_present(self):
        essential_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
            "gemini-2.5-pro", "mistral-large-latest",
        ]
        for model in essential_models:
            assert get_model_pricing(model) is not None, f"Missing pricing for {model}"


# ===========================================================================
# 19. CostEvent serialization
# ===========================================================================


class TestCostEvent:
    def test_to_dict_llm_event(self):
        event = CostEvent(
            cost=0.025,
            cost_type=CostType.LLM,
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        d = event.to_dict()
        assert d["cost"] == 0.025
        assert d["cost_type"] == "llm"
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 1000
        assert d["output_tokens"] == 500
        assert "timestamp" in d

    def test_to_dict_tool_event(self):
        event = CostEvent(
            cost=0.01,
            cost_type=CostType.TOOL,
            tool_name="web_search",
            metadata={"query": "AI news"},
        )
        d = event.to_dict()
        assert d["cost_type"] == "tool"
        assert d["tool_name"] == "web_search"
        assert d["metadata"]["query"] == "AI news"
        assert "model" not in d  # None fields should be omitted

    def test_optional_none_fields_omitted(self):
        event = CostEvent(cost=0.01, cost_type=CostType.TOOL)
        d = event.to_dict()
        assert "model" not in d
        assert "tool_name" not in d
        assert "metadata" not in d


# ===========================================================================
# 20. CircuitBreaker unit tests
# ===========================================================================


class TestCircuitBreaker:
    def test_no_warning_below_soft_limit(self):
        cb = CircuitBreaker(soft_limit_fraction=0.9)
        result = cb.check_budget(spent=0.50, budget=1.0)  # 50%
        assert result is None

    def test_warning_at_soft_limit(self):
        cb = CircuitBreaker(soft_limit_fraction=0.8)
        result = cb.check_budget(spent=0.85, budget=1.0)  # 85% >= 80%
        assert result is not None
        assert isinstance(result, str)

    def test_soft_limit_fires_once(self):
        cb = CircuitBreaker(soft_limit_fraction=0.5)
        first = cb.check_budget(spent=0.60, budget=1.0)
        second = cb.check_budget(spent=0.70, budget=1.0)
        assert first is not None
        assert second is None

    def test_soft_limit_triggered_flag(self):
        cb = CircuitBreaker(soft_limit_fraction=0.5)
        assert cb.soft_limit_triggered is False
        cb.check_budget(spent=0.60, budget=1.0)
        assert cb.soft_limit_triggered is True

    def test_loop_detection_triggers(self):
        config = LoopDetectorConfig(max_repeated_calls=3, time_window_seconds=60.0)
        cb = CircuitBreaker(loop_config=config)
        results = [cb.check_loop("tool") for _ in range(5)]
        # Third call is the one that returns True
        assert results[2] is True

    def test_different_keys_independent(self):
        config = LoopDetectorConfig(max_repeated_calls=3, time_window_seconds=60.0)
        cb = CircuitBreaker(loop_config=config)
        for _ in range(2):
            cb.check_loop("tool_a")
        # tool_b calls don't affect tool_a's counter
        assert cb.check_loop("tool_b") is False
