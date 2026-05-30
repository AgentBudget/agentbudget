"""LangChain / LangGraph integration for AgentBudget.

Provides a callback handler that tracks LLM (and optionally tool) costs
automatically and enforces a per-run budget.

Basic usage::

    from agentbudget.integrations.langchain import LangChainBudgetCallback

    with LangChainBudgetCallback(budget="$5.00") as callback:
        agent.invoke({"input": "Research competitors"}, config={"callbacks": [callback]})
    print(callback.get_report())

Tool cost tracking (optional)::

    callback = LangChainBudgetCallback(
        budget="$5.00",
        tool_costs={"web_search": 0.01, "code_exec": 0.005},
    )

Lifecycle: the callback enters its budget session on construction. Use it as a
context manager (or call ``close()``) so the session is finalized — that records
the run duration and fires ``on_hard_limit`` when the budget is exhausted.

Requires: langchain-core (optional dependency — ``pip install agentbudget[langchain]``)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..budget import AgentBudget
from ..pricing import calculate_llm_cost
from ..session import BudgetSession
from ..types import CostEvent, CostType

logger = logging.getLogger("agentbudget.integrations.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    # Provide a stub so the class definition doesn't fail at import.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


def _first_int(d: dict, keys: tuple[str, ...]) -> Optional[int]:
    """Return the first present, non-None value among *keys* in dict *d*."""
    for key in keys:
        value = d.get(key)
        if value is not None:
            return value
    return None


def _extract_llm_usage(response: Any) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract ``(model, input_tokens, output_tokens)`` from a LangChain LLM result.

    Handles the shapes LangChain actually emits:

    1. Legacy ``LLMResult.llm_output["token_usage"]`` with ``["model_name"]``
       (completion models and older chat models).
    2. Modern chat models — usage on ``message.usage_metadata``
       (``input_tokens`` / ``output_tokens``) and the model on
       ``message.response_metadata`` — which is what LangGraph runs surface.
    3. Anthropic-style ``message.response_metadata["usage"]`` with ``["model"]``.

    Tokens are summed across all generations. Returns ``None`` for any field
    that could not be determined.
    """
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # 1. Legacy llm_output path.
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict) and llm_output:
        model = llm_output.get("model_name") or llm_output.get("model")
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = _first_int(token_usage, ("prompt_tokens", "input_tokens"))
            output_tokens = _first_int(token_usage, ("completion_tokens", "output_tokens"))

    # 2/3. Modern chat-model path: aggregate usage across generations.
    if model is None or input_tokens is None or output_tokens is None:
        agg_in = 0
        agg_out = 0
        saw_tokens = False
        for batch in getattr(response, "generations", None) or []:
            for gen in batch:
                message = getattr(gen, "message", None)
                if message is None:
                    continue

                meta = getattr(message, "response_metadata", None) or {}
                if model is None and isinstance(meta, dict):
                    model = meta.get("model_name") or meta.get("model")

                usage_meta = getattr(message, "usage_metadata", None)
                if isinstance(usage_meta, dict) and usage_meta:
                    agg_in += int(usage_meta.get("input_tokens") or 0)
                    agg_out += int(usage_meta.get("output_tokens") or 0)
                    saw_tokens = True
                elif isinstance(meta, dict):
                    token_usage = meta.get("token_usage") or meta.get("usage") or {}
                    if isinstance(token_usage, dict) and token_usage:
                        in_t = _first_int(token_usage, ("prompt_tokens", "input_tokens"))
                        out_t = _first_int(token_usage, ("completion_tokens", "output_tokens"))
                        if in_t is not None:
                            agg_in += in_t
                            saw_tokens = True
                        if out_t is not None:
                            agg_out += out_t
                            saw_tokens = True

        if saw_tokens:
            if input_tokens is None:
                input_tokens = agg_in
            if output_tokens is None:
                output_tokens = agg_out

    return model, input_tokens, output_tokens


class LangChainBudgetCallback(BaseCallbackHandler):
    """LangChain callback handler that enforces a per-run budget.

    Tracks LLM call costs in real time (across legacy and modern chat-model
    usage shapes, including LangGraph) and raises
    :class:`~agentbudget.exceptions.BudgetExhausted` when the budget is exceeded.

    Tool costs are tracked when configured via *tool_costs* / *default_tool_cost*;
    by default tools record nothing (so existing callers are unaffected).

    Args:
        budget: Spend cap, e.g. ``"$5.00"`` or ``5.0``.
        session: Optionally reuse an existing :class:`BudgetSession`.
        tool_costs: Mapping of tool name to its per-call cost in USD.
        default_tool_cost: Cost applied to tools not present in *tool_costs*.
            Defaults to ``0.0`` (no tracking).
        on_soft_limit / on_hard_limit / on_loop_detected: Optional callbacks,
            forwarded to the underlying :class:`AgentBudget`.
    """

    def __init__(
        self,
        budget: str | float | int,
        session: Optional[BudgetSession] = None,
        *,
        tool_costs: Optional[dict[str, float]] = None,
        default_tool_cost: float = 0.0,
        on_soft_limit: Optional[Any] = None,
        on_hard_limit: Optional[Any] = None,
        on_loop_detected: Optional[Any] = None,
        **kwargs: Any,
    ):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for LangChainBudgetCallback. "
                "Install it with: pip install agentbudget[langchain]"
            )
        super().__init__(**kwargs)

        if session is not None:
            self.session = session
        else:
            self._agent_budget = AgentBudget(
                max_spend=budget,
                on_soft_limit=on_soft_limit,
                on_hard_limit=on_hard_limit,
                on_loop_detected=on_loop_detected,
            )
            self.session = self._agent_budget.session()

        self.session.__enter__()
        self._closed = False
        self._tool_costs = dict(tool_costs or {})
        self._default_tool_cost = default_tool_cost
        # run_id -> tool name, captured on_tool_start for use in on_tool_end.
        self._pending_tools: dict[Any, str] = {}

    # -- LLM tracking --------------------------------------------------------

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record the cost of a finished LLM call."""
        model, input_tokens, output_tokens = _extract_llm_usage(response)
        self._record_llm(model, input_tokens, output_tokens)

    def _record_llm(
        self,
        model: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        if not model or input_tokens is None or output_tokens is None:
            logger.warning(
                "AgentBudget: could not extract model/token usage from the LangChain "
                "LLM response; this call was not counted against the budget. If you are "
                "using a chat model, ensure usage metadata is enabled."
            )
            return

        cost = calculate_llm_cost(model, input_tokens, output_tokens)
        if cost is None:
            logger.warning(
                "AgentBudget: no pricing found for model %r; this call was not counted "
                "against the budget. Register it with agentbudget.register_model().",
                model,
            )
            return

        event = CostEvent(
            cost=cost,
            cost_type=CostType.LLM,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.session._ledger.record(event)
        self.session._check_after_record(call_key=model)

    # -- Tool tracking -------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Optional[dict],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Remember which tool started so its cost can be charged on end."""
        name = None
        if isinstance(serialized, dict):
            name = serialized.get("name")
        name = name or kwargs.get("name") or "tool"
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._pending_tools[run_id] = name

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Charge the configured cost for a finished tool call."""
        run_id = kwargs.get("run_id")
        name = self._pending_tools.pop(run_id, None) if run_id is not None else None
        if name is None:
            name = kwargs.get("name") or "tool"

        cost = self._tool_costs.get(name, self._default_tool_cost)
        if cost and cost > 0:
            self.session.track(output, cost=cost, tool_name=name)

    # -- Lifecycle -----------------------------------------------------------

    def __enter__(self) -> "LangChainBudgetCallback":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close(exc_type, exc_val, exc_tb)
        return False

    def close(
        self,
        exc_type: Any = None,
        exc_val: Any = None,
        exc_tb: Any = None,
    ) -> None:
        """Finalize the budget session. Idempotent.

        Records the run duration and, when the session ended due to a budget
        breach, fires the ``on_hard_limit`` callback.
        """
        if self._closed:
            return
        self._closed = True
        self.session.__exit__(exc_type, exc_val, exc_tb)

    def get_report(self) -> dict[str, Any]:
        """Get the cost report for this callback's session."""
        return self.session.report()
