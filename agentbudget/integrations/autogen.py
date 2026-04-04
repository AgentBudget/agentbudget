"""AutoGen integration for AgentBudget.

Two integration styles are provided:

1. **BudgetedAssistantAgent / BudgetedUserProxyAgent** — drop-in replacements
   for AutoGen's ``AssistantAgent`` and ``UserProxyAgent`` that automatically
   track every LLM call and raise ``BudgetExhausted`` when the session budget
   is hit.

2. **AutoGenBudgetTracker** — a lightweight orchestrator wrapper that patches a
   shared budget session onto a group of existing agents without subclassing.

Usage (subclass style)::

    from agentbudget.integrations.autogen import (
        BudgetedAssistantAgent,
        BudgetedUserProxyAgent,
    )

    assistant = BudgetedAssistantAgent(
        name="assistant",
        budget="$2.00",
        llm_config={"model": "gpt-4o", "api_key": "..."},
    )
    user_proxy = BudgetedUserProxyAgent(name="user", budget_session=assistant.budget_session)

    user_proxy.initiate_chat(assistant, message="Write a haiku about AI costs.")
    print(assistant.budget_session.report())

Usage (tracker style — wrap existing agents)::

    from agentbudget.integrations.autogen import AutoGenBudgetTracker

    tracker = AutoGenBudgetTracker(budget="$5.00")
    tracker.patch(assistant, user_proxy)          # patches both in-place

    user_proxy.initiate_chat(assistant, message="...")
    print(tracker.report())

Requires: pyautogen or autogen-agentchat (optional dependency)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from ..budget import AgentBudget
from ..exceptions import BudgetExhausted
from ..ledger import Ledger
from ..pricing import calculate_llm_cost
from ..session import BudgetSession
from ..types import CostEvent, CostType

# Optional-import guard: AutoGen is not a hard dependency

try:
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False

    # Stubs so the class bodies below don't raise NameError at import time
    class ConversableAgent:  # type: ignore[no-redef]
        """Stub — install pyautogen to use this integration."""

    class AssistantAgent(ConversableAgent):  # type: ignore[no-redef]
        pass

    class UserProxyAgent(ConversableAgent):  # type: ignore[no-redef]
        pass


# Internal helpers

def _require_autogen() -> None:
    if not _HAS_AUTOGEN:
        raise ImportError(
            "pyautogen is required for the AutoGen integration. "
            "Install it with: pip install pyautogen"
        )


def _extract_cost_from_response(response: Any) -> tuple[Optional[str], Optional[float]]:
    """Pull model name + dollar cost out of an AutoGen generate_reply response.

    AutoGen's ``generate_reply`` returns either a string (the reply text) or
    ``None``.  The actual token-level metadata lives on the *agent's*
    ``client`` object after the call.  We therefore inspect the last entry in
    ``client.total_usage_summary`` when it's available, which AutoGen populates
    automatically.

    Returns (model_name, cost_usd) — either may be None if unavailable.
    """
    # AutoGen >= 0.2 stores cumulative usage on the client
    # The response itself is just a string, so we signal the caller to pull
    # from the agent instead.  This function is kept as a hook for future
    # response-object introspection when AutoGen exposes richer objects.
    return None, None


def _record_agent_llm_cost(
    agent: "ConversableAgent",
    session: BudgetSession,
    usage_before: dict,
) -> None:
    """Compute the delta between cumulative usage snapshots and record it.

    AutoGen accumulates token usage in ``agent.client.total_usage_summary``
    (a dict keyed by model name).  By taking a snapshot before and after a
    ``generate_reply`` call we can derive exactly what that single call cost.
    """
    client = getattr(agent, "client", None)
    if client is None:
        return

    usage_after: dict = getattr(client, "total_usage_summary", {}) or {}

    for model, after_stats in usage_after.items():
        before_stats = usage_before.get(model, {})
        input_tokens = (after_stats.get("prompt_tokens", 0) -
                        before_stats.get("prompt_tokens", 0))
        output_tokens = (after_stats.get("completion_tokens", 0) -
                         before_stats.get("completion_tokens", 0))

        if input_tokens <= 0 and output_tokens <= 0:
            continue

        cost = calculate_llm_cost(model, input_tokens, output_tokens)
        if cost is None:
            continue

        event = CostEvent(
            cost=cost,
            cost_type=CostType.LLM,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        session._ledger.record(event)
        session._check_after_record(call_key=model)


def _snapshot_usage(agent: "ConversableAgent") -> dict:
    """Return a deep-ish copy of the agent's current cumulative token usage."""
    client = getattr(agent, "client", None)
    if client is None:
        return {}
    raw: dict = getattr(client, "total_usage_summary", {}) or {}
    # Shallow-copy each model's stats dict
    return {model: dict(stats) for model, stats in raw.items()}


def _patch_generate_reply(agent: "ConversableAgent", session: BudgetSession) -> None:
    """Monkey-patch ``agent.generate_reply`` to intercept LLM calls.

    We wrap the method rather than subclassing so this works on already
    constructed agent instances (the tracker style API).
    """
    original = agent.generate_reply.__func__ if hasattr(agent.generate_reply, "__func__") else None
    original_bound = agent.generate_reply

    @functools.wraps(original_bound)
    def _budgeted_generate_reply(messages=None, sender=None, **kwargs):
        usage_before = _snapshot_usage(agent)
        reply = original_bound(messages=messages, sender=sender, **kwargs)
        _record_agent_llm_cost(agent, session, usage_before)
        return reply

    agent.generate_reply = _budgeted_generate_reply  # type: ignore[method-assign]


# Subclass-style integration

class BudgetedConversableAgent(ConversableAgent):
    """A ``ConversableAgent`` subclass with built-in budget enforcement.

    Pass ``budget`` (e.g. ``"$2.00"``) to create a fresh session, or pass
    ``budget_session`` to share an existing session with other agents.
    """

    def __init__(
        self,
        *args: Any,
        budget: Optional[str | float | int] = None,
        budget_session: Optional[BudgetSession] = None,
        on_soft_limit: Optional[Callable] = None,
        on_hard_limit: Optional[Callable] = None,
        on_loop_detected: Optional[Callable] = None,
        **kwargs: Any,
    ):
        _require_autogen()
        super().__init__(*args, **kwargs)

        if budget_session is not None:
            self.budget_session = budget_session
        elif budget is not None:
            ab = AgentBudget(
                max_spend=budget,
                on_soft_limit=on_soft_limit,
                on_hard_limit=on_hard_limit,
                on_loop_detected=on_loop_detected,
            )
            self.budget_session = ab.session(session_id=f"autogen:{self.name}")
            self.budget_session.__enter__()
        else:
            raise ValueError(
                "Provide either `budget` (e.g. '$2.00') or an existing `budget_session`."
            )

    def generate_reply(self, messages=None, sender=None, **kwargs):  # type: ignore[override]
        usage_before = _snapshot_usage(self)
        reply = super().generate_reply(messages=messages, sender=sender, **kwargs)
        _record_agent_llm_cost(self, self.budget_session, usage_before)
        return reply


class BudgetedAssistantAgent(BudgetedConversableAgent, AssistantAgent):
    """Drop-in replacement for ``autogen.AssistantAgent`` with budget enforcement.

    Example::

        assistant = BudgetedAssistantAgent(
            name="assistant",
            budget="$1.00",
            llm_config={"model": "gpt-4o", "api_key": "..."},
        )
        print(f"Remaining: ${assistant.budget_session.remaining:.4f}")
    """


class BudgetedUserProxyAgent(BudgetedConversableAgent, UserProxyAgent):
    """Drop-in replacement for ``autogen.UserProxyAgent`` with budget enforcement.

    Typically you share the parent agent's session::

        user = BudgetedUserProxyAgent(
            name="user",
            budget_session=assistant.budget_session,
            human_input_mode="NEVER",
        )
    """


# Tracker style: wraps existing agent instances without subclassing

class AutoGenBudgetTracker:
    """Wraps a budget session around existing AutoGen agent instances.

    Patches ``generate_reply`` on each agent in-place so no subclassing is
    required.  The same shared session is used across all patched agents so
    the budget cap applies to the *entire* multi-agent conversation.

    Example::

        tracker = AutoGenBudgetTracker(budget="$5.00")
        tracker.patch(assistant, user_proxy, critic)

        user_proxy.initiate_chat(assistant, message="Analyse AAPL earnings.")

        print(tracker.report())
        # {
        #   "session_id": "autogen:tracker:...",
        #   "budget": 5.0,
        #   "total_spent": 0.0423,
        #   "breakdown": { "llm": {...}, "tools": {...} },
        #   ...
        # }
    """

    def __init__(
        self,
        budget: str | float | int,
        session_id: Optional[str] = None,
        on_soft_limit: Optional[Callable] = None,
        on_hard_limit: Optional[Callable] = None,
        on_loop_detected: Optional[Callable] = None,
    ):
        _require_autogen()
        self._agent_budget = AgentBudget(
            max_spend=budget,
            on_soft_limit=on_soft_limit,
            on_hard_limit=on_hard_limit,
            on_loop_detected=on_loop_detected,
        )
        _sid = session_id or f"autogen:tracker"
        self.session: BudgetSession = self._agent_budget.session(session_id=_sid)
        self.session.__enter__()
        self._patched_agents: list[ConversableAgent] = []

    def patch(self, *agents: "ConversableAgent") -> "AutoGenBudgetTracker":
        """Patch one or more existing agents to use this tracker's session.

        Returns ``self`` so calls can be chained::

            tracker.patch(agent_a, agent_b).patch(agent_c)
        """
        for agent in agents:
            _patch_generate_reply(agent, self.session)
            self._patched_agents.append(agent)
        return self

    @property
    def spent(self) -> float:
        """Total dollars spent so far across all patched agents."""
        return self.session.spent

    @property
    def remaining(self) -> float:
        """Remaining budget in dollars."""
        return self.session.remaining

    def report(self) -> dict[str, Any]:
        """Return the full structured cost report for this tracker session."""
        return self.session.report()

    def child_tracker(
        self,
        budget: float,
        *agents: "ConversableAgent",
    ) -> "AutoGenBudgetTracker":
        """Create a child tracker with a sub-budget allocated from this tracker.

        Useful for nested AutoGen crews where a sub-team should have its own cap::

            main_tracker = AutoGenBudgetTracker(budget="$10.00")
            sub_tracker = main_tracker.child_tracker(2.00, researcher, writer)
        """
        child_session = self.session.child_session(max_spend=budget)
        child_session.__enter__()

        child = AutoGenBudgetTracker.__new__(AutoGenBudgetTracker)
        child.session = child_session
        child._agent_budget = self._agent_budget
        child._patched_agents = []

        for agent in agents:
            _patch_generate_reply(agent, child_session)
            child._patched_agents.append(agent)

        return child

    def __enter__(self) -> "AutoGenBudgetTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        self.session.__exit__(*args)
