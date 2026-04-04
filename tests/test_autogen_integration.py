"""Tests for the AutoGen integration.

Most tests use mock agents because pyautogen is not a test dependency.
They validate the budget enforcement plumbing independently of the
real AutoGen SDK.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentbudget import AgentBudget, BudgetExhausted
from agentbudget.integrations.autogen import (
    AutoGenBudgetTracker,
    _patch_generate_reply,
    _record_agent_llm_cost,
    _snapshot_usage,
)
from agentbudget.session import BudgetSession


class FakeClient:
    """Minimal stand-in for autogen's ModelClient."""

    def __init__(self, usage: dict | None = None):
        self.total_usage_summary: dict = usage or {}

    def add_usage(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        entry = self.total_usage_summary.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens


class FakeAgent:
    """Stand-in for autogen.ConversableAgent."""

    def __init__(self, name: str, client: FakeClient | None = None):
        self.name = name
        self.client = client or FakeClient()
        self._reply_count = 0

    def generate_reply(self, messages=None, sender=None, **kwargs) -> str:
        self._reply_count += 1
        # Simulate an LLM call adding tokens on every reply
        self.client.add_usage("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        return f"reply {self._reply_count}"


# ---------------------------------------------------------------------------
# _snapshot_usage
# ---------------------------------------------------------------------------

class TestSnapshotUsage:
    def test_returns_empty_for_agent_without_client(self):
        agent = FakeAgent("a")
        agent.client = None  # type: ignore[assignment]
        assert _snapshot_usage(agent) == {}

    def test_returns_copy_not_reference(self):
        agent = FakeAgent("a")
        agent.client.add_usage("gpt-4o", 10, 5)
        snap = _snapshot_usage(agent)
        # Mutate the original, snapshot should not change
        agent.client.add_usage("gpt-4o", 999, 999)
        assert snap["gpt-4o"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# _record_agent_llm_cost
# ---------------------------------------------------------------------------

class TestRecordAgentLlmCost:
    def _make_session(self, budget: float = 10.0) -> BudgetSession:
        ab = AgentBudget(max_spend=budget)
        s = ab.session()
        s.__enter__()
        return s

    def test_records_delta_cost(self):
        session = self._make_session()
        agent = FakeAgent("worker")

        # Snapshot BEFORE the call (empty)
        usage_before = _snapshot_usage(agent)

        # Simulate the call adding tokens
        agent.client.add_usage("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

        _record_agent_llm_cost(agent, session, usage_before)

        assert session.spent > 0.0

    def test_only_delta_is_charged(self):
        """Pre-existing usage should not be billed twice."""
        session = self._make_session()
        agent = FakeAgent("worker")
        agent.client.add_usage("gpt-4o-mini", prompt_tokens=500, completion_tokens=200)

        # Snapshot after first call
        usage_before = _snapshot_usage(agent)

        # Second call adds more tokens
        agent.client.add_usage("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)

        _record_agent_llm_cost(agent, session, usage_before)

        # Only the delta (100 in, 50 out) should be recorded
        events = session._ledger.events
        assert len(events) == 1
        assert events[0].input_tokens == 100
        assert events[0].output_tokens == 50

    def test_unknown_model_not_recorded(self):
        session = self._make_session()
        agent = FakeAgent("worker")
        usage_before = _snapshot_usage(agent)
        agent.client.add_usage("unknown-model-xyz", prompt_tokens=100, completion_tokens=50)
        _record_agent_llm_cost(agent, session, usage_before)
        # No events because pricing is unknown
        assert len(session._ledger.events) == 0

    def test_no_client_is_noop(self):
        session = self._make_session()
        agent = FakeAgent("worker")
        agent.client = None  # type: ignore[assignment]
        _record_agent_llm_cost(agent, session, {})
        assert session.spent == 0.0


# ---------------------------------------------------------------------------
# _patch_generate_reply
# ---------------------------------------------------------------------------

class TestPatchGenerateReply:
    def _make_session(self, budget: float = 10.0) -> BudgetSession:
        ab = AgentBudget(max_spend=budget)
        s = ab.session()
        s.__enter__()
        return s

    def test_patched_agent_records_cost(self):
        session = self._make_session()
        agent = FakeAgent("patched")
        _patch_generate_reply(agent, session)

        agent.generate_reply(messages=[], sender=None)

        assert session.spent > 0.0

    def test_patched_agent_returns_original_reply(self):
        session = self._make_session()
        agent = FakeAgent("patched")
        _patch_generate_reply(agent, session)

        reply = agent.generate_reply(messages=[], sender=None)
        assert reply == "reply 1"

    def test_multiple_calls_accumulate(self):
        session = self._make_session()
        agent = FakeAgent("patched")
        _patch_generate_reply(agent, session)

        agent.generate_reply()
        agent.generate_reply()
        agent.generate_reply()

        assert len(session._ledger.events) == 3

    def test_budget_exhausted_raised(self):
        # $0.0001 won't survive even one gpt-4o-mini call
        session = self._make_session(budget=0.00001)
        agent = FakeAgent("patched")
        _patch_generate_reply(agent, session)

        with pytest.raises(BudgetExhausted):
            agent.generate_reply()


# ---------------------------------------------------------------------------
# AutoGenBudgetTracker — import guard
# ---------------------------------------------------------------------------

class TestAutoGenImportGuard:
    def test_raises_without_autogen(self):
        """AutoGenBudgetTracker should raise ImportError when pyautogen is absent."""
        import agentbudget.integrations.autogen as mod

        original = mod._HAS_AUTOGEN
        try:
            mod._HAS_AUTOGEN = False
            with pytest.raises(ImportError, match="pyautogen"):
                AutoGenBudgetTracker(budget="$1.00")
        finally:
            mod._HAS_AUTOGEN = original


# ---------------------------------------------------------------------------
# AutoGenBudgetTracker — full behaviour (with mocked _HAS_AUTOGEN)
# ---------------------------------------------------------------------------

@pytest.fixture()
def enable_autogen(monkeypatch):
    """Pretend pyautogen is installed."""
    import agentbudget.integrations.autogen as mod
    monkeypatch.setattr(mod, "_HAS_AUTOGEN", True)


class TestAutoGenBudgetTracker:
    def test_basic_creation(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$5.00")
        assert tracker.remaining == 5.0
        assert tracker.spent == 0.0

    def test_patch_and_track(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$5.00")
        agent = FakeAgent("a")
        tracker.patch(agent)

        agent.generate_reply()

        assert tracker.spent > 0.0

    def test_patch_multiple_agents_shared_session(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$5.00")
        a1, a2 = FakeAgent("a1"), FakeAgent("a2")
        tracker.patch(a1, a2)

        a1.generate_reply()
        a2.generate_reply()

        assert len(tracker.session._ledger.events) == 2

    def test_chained_patch(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$5.00")
        a1, a2 = FakeAgent("a1"), FakeAgent("a2")
        tracker.patch(a1).patch(a2)

        a1.generate_reply()
        a2.generate_reply()

        assert tracker.spent > 0.0

    def test_report_structure(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$5.00", session_id="test-session")
        agent = FakeAgent("a")
        tracker.patch(agent)
        agent.generate_reply()

        report = tracker.report()
        assert report["session_id"] == "test-session"
        assert report["budget"] == 5.0
        assert "breakdown" in report
        assert "llm" in report["breakdown"]

    def test_budget_enforcement_across_agents(self, enable_autogen):
        tracker = AutoGenBudgetTracker(budget="$0.00001")
        agent = FakeAgent("a")
        tracker.patch(agent)

        with pytest.raises(BudgetExhausted):
            agent.generate_reply()

    def test_soft_limit_callback(self, enable_autogen):
        warnings = []
        tracker = AutoGenBudgetTracker(
            budget="$1.00",
            on_soft_limit=lambda r: warnings.append(r),
        )
        agent = FakeAgent("a")
        # Fake a big call directly via session to push past 90%
        tracker.patch(agent)
        tracker.session.track(None, cost=0.91, tool_name="big_call")

        assert len(warnings) == 1

    def test_context_manager(self, enable_autogen):
        with AutoGenBudgetTracker(budget="$5.00") as tracker:
            agent = FakeAgent("a")
            tracker.patch(agent)
            agent.generate_reply()
        assert tracker.spent > 0.0

    def test_child_tracker_sub_budget(self, enable_autogen):
        parent = AutoGenBudgetTracker(budget="$10.00")
        child_agent = FakeAgent("child_worker")
        child = parent.child_tracker(1.00, child_agent)

        # Child session should be capped at $1.00
        assert child.session._ledger.budget == 1.00

    def test_child_tracker_uses_separate_session(self, enable_autogen):
        parent = AutoGenBudgetTracker(budget="$10.00")
        parent_agent = FakeAgent("parent_worker")
        child_agent = FakeAgent("child_worker")

        tracker_parent = parent.patch(parent_agent)
        child = parent.child_tracker(1.00, child_agent)

        parent_agent.generate_reply()
        child_agent.generate_reply()

        # Child cost must show up in its own session
        assert child.spent > 0.0
        # And be independent from parent's direct session events
        parent_direct_events = [
            e for e in parent.session._ledger.events
            if e.tool_name != f"child:{child.session.session_id}"
        ]
        assert len(parent_direct_events) == 1  # only parent_agent's call
