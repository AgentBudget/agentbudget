"""Tests for BudgetSession."""

import pytest

from agentbudget.exceptions import BudgetExhausted
from agentbudget.ledger import Ledger
from agentbudget.session import BudgetSession


class FakeUsage:
    """OpenAI-style usage object."""
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeAnthropicUsage:
    """Anthropic-style usage object."""
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, model, prompt_tokens, completion_tokens):
        self.model = model
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class FakeAnthropicResponse:
    def __init__(self, model, input_tokens, output_tokens):
        self.model = model
        self.usage = FakeAnthropicUsage(input_tokens, output_tokens)


def test_session_context_manager():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        assert session.session_id.startswith("sess_")
        assert session.spent == 0.0
        assert session.remaining == 5.0


def test_wrap_records_llm_cost():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        response = FakeResponse("gpt-4o", prompt_tokens=1000, completion_tokens=500)
        result = session.wrap(response)
        assert result is response
        assert session.spent > 0


def test_wrap_returns_response():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        response = FakeResponse("gpt-4o", prompt_tokens=100, completion_tokens=50)
        result = session.wrap(response)
        assert result is response


def test_track_records_tool_cost():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        data = {"results": [1, 2, 3]}
        result = session.track(data, cost=0.01, tool_name="serp_api")
        assert result is data
        assert session.spent == 0.01


def test_track_multiple_tools():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        session.track("a", cost=0.01, tool_name="api_a")
        session.track("b", cost=0.02, tool_name="api_b")
        session.track("c", cost=0.03, tool_name="api_c")
        assert abs(session.spent - 0.06) < 1e-10


def test_budget_exhausted_during_track():
    ledger = Ledger(budget=0.05)
    with pytest.raises(BudgetExhausted):
        with BudgetSession(ledger) as session:
            session.track("a", cost=0.03)
            session.track("b", cost=0.03)  # exceeds 0.05


def test_wrap_unknown_model(caplog):
    ledger = Ledger(budget=5.0)
    with caplog.at_level("WARNING", logger="agentbudget.session"):
        with BudgetSession(ledger) as session:
            response = FakeResponse("unknown-model-xyz", prompt_tokens=100, completion_tokens=50)
            result = session.wrap(response)
            assert result is response
            assert session.spent == 0.0  # unknown model, no cost recorded
    assert "no pricing found for model 'unknown-model-xyz'" in caplog.text


def test_wrap_no_usage(caplog):
    ledger = Ledger(budget=5.0)
    with caplog.at_level("WARNING", logger="agentbudget.session"):
        with BudgetSession(ledger) as session:
            response = "plain string"
            result = session.wrap(response)
            assert result == "plain string"
            assert session.spent == 0.0
    assert "model could not be extracted" in caplog.text


def test_wrap_no_usage_with_known_model(caplog):
    class ResponseWithoutUsage:
        model = "gpt-4o"

    ledger = Ledger(budget=5.0)
    with caplog.at_level("WARNING", logger="agentbudget.session"):
        with BudgetSession(ledger) as session:
            result = session.wrap(ResponseWithoutUsage())
            assert isinstance(result, ResponseWithoutUsage)
            assert session.spent == 0.0
    assert "token usage could not be extracted" in caplog.text


def test_report_structure():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger, session_id="sess_test123") as session:
        session.track("x", cost=0.50, tool_name="my_tool")

    report = session.report()
    assert report["session_id"] == "sess_test123"
    assert report["budget"] == 5.0
    assert report["total_spent"] == 0.50
    assert report["remaining"] == 4.50
    assert report["duration_seconds"] is not None
    assert report["terminated_by"] is None
    assert len(report["events"]) == 1
    assert report["breakdown"]["tools"]["total"] == 0.50


def test_report_terminated_by_budget():
    ledger = Ledger(budget=0.10)
    try:
        with BudgetSession(ledger) as session:
            session.track("a", cost=0.05)
            session.track("b", cost=0.06)  # exceeds
    except BudgetExhausted:
        pass

    report = session.report()
    assert report["terminated_by"] == "budget_exhausted"


def test_wrap_anthropic_response():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        response = FakeAnthropicResponse(
            "claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500
        )
        result = session.wrap(response)
        assert result is response
        assert session.spent > 0


def test_wrap_anthropic_cost_correct():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:
        response = FakeAnthropicResponse(
            "claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500
        )
        session.wrap(response)
        # claude-3-5-sonnet: $3/1M input, $15/1M output
        expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        assert abs(session.spent - expected) < 1e-10


def test_track_tool_decorator():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:

        @session.track_tool(cost=0.02, tool_name="search")
        def my_search(query):
            return {"results": [query]}

        result = my_search("test query")
        assert result == {"results": ["test query"]}
        assert session.spent == 0.02


def test_track_tool_decorator_uses_func_name():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:

        @session.track_tool(cost=0.01)
        def fetch_data():
            return "data"

        fetch_data()

    report = session.report()
    assert report["breakdown"]["tools"]["by_tool"]["fetch_data"] == 0.01


def test_track_tool_decorator_multiple_calls():
    ledger = Ledger(budget=5.0)
    with BudgetSession(ledger) as session:

        @session.track_tool(cost=0.05, tool_name="api")
        def call_api():
            return "ok"

        call_api()
        call_api()
        call_api()
        assert abs(session.spent - 0.15) < 1e-10
