"""
Integration smoke tests for the AgentBudget Python SDK.

These tests make REAL network calls to live LLM APIs and require
environment variables to be set:

    OPENAI_API_KEY   — for OpenAI tests
    ANTHROPIC_API_KEY — for Anthropic tests

All tests are automatically skipped if the required key is absent.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_smoke.py -v

Or to run only OpenAI tests:
    OPENAI_API_KEY=sk-... pytest tests/test_smoke.py -k openai -v
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Key availability helpers
# ---------------------------------------------------------------------------

_HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
_HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))

_SKIP_OPENAI = pytest.mark.skipif(
    not _HAS_OPENAI_KEY,
    reason="OPENAI_API_KEY not set — skipping live OpenAI smoke tests",
)
_SKIP_ANTHROPIC = pytest.mark.skipif(
    not _HAS_ANTHROPIC_KEY,
    reason="ANTHROPIC_API_KEY not set — skipping live Anthropic smoke tests",
)

# ---------------------------------------------------------------------------
# Cheap models to keep CI costs minimal
# ---------------------------------------------------------------------------

_OPENAI_CHEAP_MODEL = "gpt-4o-mini"       # $0.15/1M input, $0.60/1M output
_ANTHROPIC_CHEAP_MODEL = "claude-3-haiku-20240307"  # $0.25/1M input, $1.25/1M output

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _teardown_global():
    """Guarantee agentbudget global state is clean before and after each test."""
    import agentbudget
    agentbudget.teardown()
    yield
    agentbudget.teardown()


# ===========================================================================
# 1. Full end-to-end: init() → real OpenAI call → spent() > 0 → teardown()
# ===========================================================================


@_SKIP_OPENAI
def test_e2e_init_openai_call_spent_teardown():
    """
    The most important smoke test: a real OpenAI call should be automatically
    tracked after agentbudget.init() patches the client.
    """
    import agentbudget
    import openai

    agentbudget.init(budget="$0.50")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=_OPENAI_CHEAP_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' in exactly one word."}],
        max_tokens=5,
    )

    # The patched client should have recorded cost automatically
    total_spent = agentbudget.spent()
    assert total_spent > 0.0, (
        f"Expected spent > 0 after a real LLM call, got {total_spent}"
    )

    rem = agentbudget.remaining()
    assert rem < 0.50, f"Remaining should be less than $0.50, got {rem}"
    assert abs(total_spent + rem - 0.50) < 1e-6, "spent + remaining must equal budget"

    r = agentbudget.teardown()
    assert r is not None
    assert r["total_spent"] == pytest.approx(total_spent, abs=1e-9)
    assert r["breakdown"]["llm"]["calls"] == 1
    assert _OPENAI_CHEAP_MODEL in r["breakdown"]["llm"]["by_model"]


@_SKIP_OPENAI
def test_e2e_report_has_correct_structure():
    """Full report after a real call should have all required fields."""
    import agentbudget
    import openai

    agentbudget.init(budget="$0.50", session_id="smoke_test_001")

    client = openai.OpenAI()
    client.chat.completions.create(
        model=_OPENAI_CHEAP_MODEL,
        messages=[{"role": "user", "content": "One-word answer: capital of France?"}],
        max_tokens=3,
    )

    r = agentbudget.report()
    assert r is not None

    required_keys = {
        "session_id", "budget", "total_spent", "remaining",
        "breakdown", "duration_seconds", "terminated_by", "events",
    }
    assert required_keys.issubset(r.keys()), (
        f"Missing keys: {required_keys - r.keys()}"
    )
    assert r["session_id"] == "smoke_test_001"
    assert r["budget"] == pytest.approx(0.50, abs=1e-6)
    assert len(r["events"]) >= 1

    event = r["events"][0]
    assert event["cost_type"] == "llm"
    assert "model" in event
    assert "input_tokens" in event
    assert "output_tokens" in event
    assert event["input_tokens"] > 0
    assert event["output_tokens"] > 0


@_SKIP_OPENAI
def test_e2e_multiple_calls_accumulate():
    """Multiple real calls should each be tracked and accumulate correctly."""
    import agentbudget
    import openai

    agentbudget.init(budget="$1.00")
    client = openai.OpenAI()

    spent_before = agentbudget.spent()

    for i in range(2):
        client.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": f"Say the number {i} only."}],
            max_tokens=3,
        )

    total_spent = agentbudget.spent()
    assert total_spent > spent_before
    r = agentbudget.report()
    assert r["breakdown"]["llm"]["calls"] == 2


# ===========================================================================
# 2. wrap_client() tracks a real call
# ===========================================================================


@_SKIP_OPENAI
def test_wrap_client_tracks_real_openai_call():
    """
    wrap_client() should track a real call on the specific client instance
    without affecting other clients.
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$0.50")
    with budget.session() as session:
        client = agentbudget.wrap_client(openai.OpenAI(), session)

        response = client.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=5,
        )

        assert session.spent > 0.0, (
            f"wrap_client should have tracked cost, got {session.spent}"
        )
        assert response.choices[0].message.content is not None

    r = session.report()
    assert r["breakdown"]["llm"]["calls"] == 1
    assert r["total_spent"] > 0


@_SKIP_OPENAI
def test_wrap_client_does_not_affect_unwrapped_client():
    """
    Calls on an un-wrapped client should NOT be tracked in the session.
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$1.00")
    with budget.session() as session:
        wrapped = agentbudget.wrap_client(openai.OpenAI(), session)
        unwrapped = openai.OpenAI()  # separate instance, not wrapped

        # Call only the unwrapped client
        unwrapped.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Ignore this."}],
            max_tokens=3,
        )

        # Session should have 0 cost (unwrapped client not tracked)
        # Note: if global patch is active this would differ, but we're using
        # a raw budget session here without init().
        assert session.spent == 0.0, (
            "Unwrapped client call should not be tracked in session"
        )

        # Now call the wrapped client
        wrapped.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=3,
        )
        assert session.spent > 0.0, "Wrapped client call must be tracked"


@_SKIP_OPENAI
def test_wrap_client_returns_same_client_object():
    """wrap_client() should return the same client object (mutates in place)."""
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$1.00")
    session = budget.session()
    client = openai.OpenAI()
    returned = agentbudget.wrap_client(client, session)
    assert returned is client


# ===========================================================================
# 3. Budget exhausted with real call (tiny budget)
# ===========================================================================


@_SKIP_OPENAI
def test_budget_exhausted_on_real_call_tiny_budget():
    """
    Setting a budget smaller than a single API call cost should raise
    BudgetExhausted during session.wrap().
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget, BudgetExhausted

    # $0.000001 is far too small for any real LLM call
    budget = AgentBudget(max_spend=0.000001)

    with pytest.raises(BudgetExhausted) as exc_info:
        with budget.session() as session:
            client = agentbudget.wrap_client(openai.OpenAI(), session)
            client.chat.completions.create(
                model=_OPENAI_CHEAP_MODEL,
                messages=[{"role": "user", "content": "Hello."}],
                max_tokens=5,
            )

    exc = exc_info.value
    assert exc.budget == pytest.approx(0.000001, rel=1e-3)
    assert exc.spent > exc.budget


@_SKIP_OPENAI
def test_budget_exhausted_sets_terminated_by():
    """Confirm terminated_by is 'budget_exhausted' after hard-limit hit."""
    import openai
    import agentbudget
    from agentbudget import AgentBudget, BudgetExhausted

    budget = AgentBudget(max_spend=0.000001)
    try:
        with budget.session() as session:
            client = agentbudget.wrap_client(openai.OpenAI(), session)
            client.chat.completions.create(
                model=_OPENAI_CHEAP_MODEL,
                messages=[{"role": "user", "content": "Hello."}],
                max_tokens=5,
            )
    except BudgetExhausted:
        pass

    assert session.report()["terminated_by"] == "budget_exhausted"


@_SKIP_OPENAI
def test_on_hard_limit_callback_fires_on_real_call():
    """on_hard_limit callback should be called when budget is exhausted."""
    import openai
    import agentbudget
    from agentbudget import AgentBudget, BudgetExhausted

    reports: list[dict] = []
    budget = AgentBudget(
        max_spend=0.000001,
        on_hard_limit=lambda r: reports.append(r),
    )
    try:
        with budget.session() as session:
            client = agentbudget.wrap_client(openai.OpenAI(), session)
            client.chat.completions.create(
                model=_OPENAI_CHEAP_MODEL,
                messages=[{"role": "user", "content": "Hello."}],
                max_tokens=5,
            )
    except BudgetExhausted:
        pass

    assert len(reports) == 1
    assert reports[0]["terminated_by"] == "budget_exhausted"


# ===========================================================================
# 4. session.wrap() with a real Anthropic call
# ===========================================================================


@_SKIP_ANTHROPIC
def test_e2e_anthropic_wrap_real_call():
    """
    session.wrap() should correctly extract usage from a real Anthropic
    Messages API response and record cost.
    """
    import anthropic
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$0.50")
    with budget.session() as session:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_ANTHROPIC_CHEAP_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say only: Hello"}],
        )
        session.wrap(response)

    assert session.spent > 0.0, (
        f"Expected spent > 0 after Anthropic call, got {session.spent}"
    )

    r = session.report()
    assert r["breakdown"]["llm"]["calls"] == 1
    assert _ANTHROPIC_CHEAP_MODEL in r["breakdown"]["llm"]["by_model"]

    # Verify cost math roughly: claude-3-haiku is $0.25/1M input, $1.25/1M output
    # A minimal call with 10 tokens output won't cost more than $0.01
    assert session.spent < 0.01, f"Cost for a tiny Haiku call seems too high: {session.spent}"


@_SKIP_ANTHROPIC
def test_e2e_anthropic_init_auto_tracking():
    """
    agentbudget.init() should auto-patch Anthropic messages.create so calls
    are tracked without explicit session.wrap().
    """
    import agentbudget
    import anthropic

    agentbudget.init(budget="$0.50")

    client = anthropic.Anthropic()
    client.messages.create(
        model=_ANTHROPIC_CHEAP_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": "One word: color of sky?"}],
    )

    total_spent = agentbudget.spent()
    assert total_spent > 0.0, (
        f"Expected auto-tracked cost > 0, got {total_spent}"
    )

    r = agentbudget.report()
    assert r["breakdown"]["llm"]["calls"] == 1


@_SKIP_ANTHROPIC
def test_e2e_anthropic_budget_exhausted():
    """A real Anthropic call should exhaust a micro budget."""
    import anthropic
    import agentbudget
    from agentbudget import AgentBudget, BudgetExhausted

    budget = AgentBudget(max_spend=0.000001)
    with pytest.raises(BudgetExhausted):
        with budget.session() as session:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=_ANTHROPIC_CHEAP_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi."}],
            )
            session.wrap(response)


@_SKIP_ANTHROPIC
def test_e2e_anthropic_usage_fields_populated():
    """Real Anthropic response should have input/output token counts in the report."""
    import anthropic
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$0.50")
    with budget.session() as session:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_ANTHROPIC_CHEAP_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        session.wrap(response)

    events = session.report()["events"]
    assert len(events) == 1
    e = events[0]
    assert e["input_tokens"] > 0
    assert e["output_tokens"] > 0
    assert e["model"] == _ANTHROPIC_CHEAP_MODEL


# ===========================================================================
# 5. Streaming smoke tests (OpenAI)
# ===========================================================================


@_SKIP_OPENAI
def test_openai_streaming_tracked_after_full_iteration():
    """
    When using stream=True with stream_options include_usage, cost should
    be recorded after the caller exhausts the iterator.
    """
    import openai
    import agentbudget

    agentbudget.init(budget="$0.50")
    client = openai.OpenAI()

    chunks_received = 0
    for chunk in client.chat.completions.create(
        model=_OPENAI_CHEAP_MODEL,
        messages=[{"role": "user", "content": "Count to three."}],
        max_tokens=20,
        stream=True,
        stream_options={"include_usage": True},
    ):
        chunks_received += 1

    assert chunks_received > 0

    total_spent = agentbudget.spent()
    assert total_spent > 0.0, (
        f"Streaming cost should be recorded after iteration, got {total_spent}"
    )


@_SKIP_OPENAI
def test_openai_streaming_without_usage_option_skips_gracefully():
    """
    Without stream_options include_usage, cost tracking is skipped but
    no error is raised — the stream still yields chunks normally.
    """
    import openai
    import agentbudget

    agentbudget.init(budget="$0.50")
    client = openai.OpenAI()

    chunks = list(client.chat.completions.create(
        model=_OPENAI_CHEAP_MODEL,
        messages=[{"role": "user", "content": "Say ok."}],
        max_tokens=5,
        stream=True,
        # Intentionally omitting stream_options={"include_usage": True}
    ))

    assert len(chunks) > 0
    # Cost may or may not be tracked depending on whether the provider
    # returns usage anyway; no assertion on spent() here — just no crash.


# ===========================================================================
# 6. Mixed LLM + tool calls in one session
# ===========================================================================


@_SKIP_OPENAI
def test_mixed_llm_and_tool_costs_in_one_session():
    """
    A realistic agent session: one LLM call + two tool calls, all tracked.
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$1.00")
    with budget.session() as session:
        client = agentbudget.wrap_client(openai.OpenAI(), session)

        # Simulate: agent calls LLM to plan
        client.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Plan a 3-step task. One sentence."}],
            max_tokens=30,
        )

        # Agent then calls two external tools
        session.track({"results": [1, 2, 3]}, cost=0.005, tool_name="web_search")
        session.track("<html>page</html>", cost=0.020, tool_name="scraper")

    r = session.report()
    assert r["breakdown"]["llm"]["calls"] == 1
    assert r["breakdown"]["tools"]["calls"] == 2
    assert r["total_spent"] > 0.025  # at least the two tool costs
    assert r["terminated_by"] is None

    # Breakdown by tool
    by_tool = r["breakdown"]["tools"]["by_tool"]
    assert by_tool["web_search"] == pytest.approx(0.005)
    assert by_tool["scraper"] == pytest.approx(0.020)


# ===========================================================================
# 7. Soft-limit callback fires on a real session
# ===========================================================================


@_SKIP_OPENAI
def test_soft_limit_fires_on_real_session():
    """
    With a soft_limit set to just above 0 (e.g. 0.01%), any real call
    should cross the threshold and fire the callback.
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    warnings: list[dict] = []
    # Set soft_limit to 0.0001% so any real spend triggers it
    budget = AgentBudget(
        max_spend="$5.00",
        soft_limit=0.000001,
        on_soft_limit=lambda r: warnings.append(r),
    )
    with budget.session() as session:
        client = agentbudget.wrap_client(openai.OpenAI(), session)
        client.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Hi."}],
            max_tokens=5,
        )

    assert len(warnings) == 1, (
        f"Expected 1 soft-limit warning, got {len(warnings)}"
    )
    assert warnings[0]["total_spent"] > 0


# ===========================================================================
# 8. would_exceed() pre-flight with real data
# ===========================================================================


@_SKIP_OPENAI
def test_would_exceed_preflight_after_real_call():
    """
    After a real call has been tracked, would_exceed() should correctly
    predict whether a future cost would push over the limit.
    """
    import openai
    import agentbudget
    from agentbudget import AgentBudget

    budget = AgentBudget(max_spend="$0.10")
    with budget.session() as session:
        client = agentbudget.wrap_client(openai.OpenAI(), session)
        client.chat.completions.create(
            model=_OPENAI_CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=5,
        )

        actual_spent = session.spent
        remaining = session.remaining

        # A cost of $0.00 should never exceed
        assert session.would_exceed(0.0) is False

        # A cost equal to remaining should NOT exceed (not strictly over)
        assert session.would_exceed(remaining) is False

        # A cost just over remaining SHOULD exceed
        assert session.would_exceed(remaining + 0.000001) is True


# ===========================================================================
# 9. teardown() returns final report with live data
# ===========================================================================


@_SKIP_OPENAI
def test_teardown_returns_accurate_final_report():
    """teardown() should return the final report including all tracked events."""
    import openai
    import agentbudget

    agentbudget.init(budget="$0.50")
    client = openai.OpenAI()

    client.chat.completions.create(
        model=_OPENAI_CHEAP_MODEL,
        messages=[{"role": "user", "content": "Hi."}],
        max_tokens=5,
    )
    agentbudget.track(result=None, cost=0.01, tool_name="my_tool")

    final_report = agentbudget.teardown()

    assert final_report is not None
    assert final_report["total_spent"] > 0.01
    assert final_report["breakdown"]["llm"]["calls"] == 1
    assert final_report["breakdown"]["tools"]["calls"] == 1
    assert final_report["breakdown"]["tools"]["by_tool"]["my_tool"] == pytest.approx(0.01)

    # After teardown, global state should be clean
    assert agentbudget.spent() == 0.0
    assert agentbudget.get_session() is None


# ===========================================================================
# 10. Concurrent real calls (context isolation)
# ===========================================================================


@_SKIP_OPENAI
def test_concurrent_thread_sessions_are_isolated():
    """
    Two threads calling agentbudget.init() independently should keep separate
    sessions and reports, even though the SDK patch itself is process-wide.
    """
    import threading
    import openai
    import agentbudget

    errors: list[Exception] = []
    reports: list[dict] = []

    def make_call(session_id: str):
        try:
            agentbudget.init(budget="$1.00", session_id=session_id)
            client = openai.OpenAI()
            client.chat.completions.create(
                model=_OPENAI_CHEAP_MODEL,
                messages=[{"role": "user", "content": "Say one word."}],
                max_tokens=3,
            )
            report = agentbudget.teardown()
            assert report is not None
            reports.append(report)
        except Exception as e:
            errors.append(e)
            agentbudget.teardown()

    threads = [
        threading.Thread(target=make_call, args=("thread_a",)),
        threading.Thread(target=make_call, args=("thread_b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(reports) == 2
    assert {report["session_id"] for report in reports} == {"thread_a", "thread_b"}
    assert all(report["breakdown"]["llm"]["calls"] == 1 for report in reports)
    assert all(report["total_spent"] > 0 for report in reports)
