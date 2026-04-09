"""LangChain/LangGraph integration for AgentBudget.

Provides a callback handler that tracks LLM and tool costs for LangChain runs.

Usage:
    from agentbudget.integrations.langchain import LangChainBudgetCallback

    with LangChainBudgetCallback(
        budget="$5.00",
        tool_costs={"search": 0.01},
    ) as callback:
        agent.run(callbacks=[callback])
        print(callback.get_report())

Requires: langchain-core (optional dependency)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from uuid import UUID

from ..budget import AgentBudget
from ..pricing import calculate_llm_cost
from ..session import BudgetSession

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    # Provide a stub so the class definition doesn't fail at import
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    class BaseMessage:  # type: ignore[no-redef]
        pass


ToolCostExtractor = Callable[[str, Any, dict[str, Any]], Optional[float]]


class LangChainBudgetCallback(BaseCallbackHandler):
    """LangChain callback handler that enforces a per-run budget.

    Tracks LLM call costs in real time and can optionally track tools via:
    1. explicit ``metadata={"agentbudget_cost": ...}`` on the tool
    2. a ``tool_costs`` mapping passed to the callback
    3. a custom ``tool_cost_extractor`` callback
    """

    def __init__(
        self,
        budget: str | float | int,
        session: Optional[BudgetSession] = None,
        tool_costs: Optional[dict[str, float]] = None,
        tool_cost_extractor: Optional[ToolCostExtractor] = None,
        **kwargs: Any,
    ):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for LangChainBudgetCallback. "
                "Install it with: pip install langchain-core"
            )
        super().__init__(**kwargs)
        self._agent_budget = AgentBudget(max_spend=budget)
        self.session = session or self._agent_budget.session()
        self._owns_session = session is None
        self._closed = False
        self._tool_costs = tool_costs or {}
        self._tool_cost_extractor = tool_cost_extractor
        self._run_models: dict[UUID, str] = {}
        self._tool_runs: dict[UUID, dict[str, Any]] = {}
        if self._owns_session:
            self.session.__enter__()

    def __enter__(self) -> "LangChainBudgetCallback":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def close(
        self,
        *,
        exc_type: Any = None,
        exc_val: Any = None,
        exc_tb: Any = None,
    ) -> dict[str, Any]:
        """Close the underlying budget session and return the final report."""
        if not self._closed:
            self._run_models.clear()
            self._tool_runs.clear()
            if self._owns_session:
                self.session.__exit__(exc_type, exc_val, exc_tb)
            self._closed = True
        return self.session.report()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Capture the model name for chat-model runs when available."""
        model_name = self._extract_model_name(serialized, kwargs)
        if model_name:
            self._run_models[run_id] = model_name

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Capture the model name for non-chat LLM runs when available."""
        del prompts
        model_name = self._extract_model_name(serialized, kwargs)
        if model_name:
            self._run_models[run_id] = model_name

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Called when an LLM call finishes. Records the cost."""
        model_name, input_tokens, output_tokens = self._extract_llm_usage(
            response, run_id=run_id, kwargs=kwargs
        )
        if model_name and input_tokens is not None and output_tokens is not None:
            cost = calculate_llm_cost(model_name, input_tokens, output_tokens)
            if cost is not None:
                self.session.wrap(
                    _LangChainTrackedResponse(
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
        self._run_models.pop(run_id, None)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool metadata so costs can be resolved on tool completion."""
        del input_str
        tool_name = (
            serialized.get("name")
            or serialized.get("id")
            or kwargs.get("name")
            or "tool"
        )
        self._tool_runs[run_id] = {
            "tool_name": tool_name,
            "metadata": metadata.copy() if metadata else {},
            "serialized": serialized,
            "kwargs": kwargs,
        }

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Called when a tool finishes. Tracks cost when configured."""
        tool_run = self._tool_runs.pop(run_id, {})
        tool_name = tool_run.get("tool_name", "tool")
        metadata = dict(tool_run.get("metadata", {}))
        metadata.update(kwargs.get("metadata") or {})

        cost = kwargs.get("cost")
        if cost is None:
            cost = metadata.get("agentbudget_cost")
        if cost is None:
            cost = self._tool_costs.get(tool_name)
        if cost is None and self._tool_cost_extractor is not None:
            cost = self._tool_cost_extractor(tool_name, output, metadata)
        if cost is None:
            return

        self.session.track(
            output,
            cost=float(cost),
            tool_name=tool_name,
            metadata=metadata or None,
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Clear any pending tool state if the tool errors."""
        del error, kwargs
        self._tool_runs.pop(run_id, None)

    def get_report(self) -> dict[str, Any]:
        """Get the cost report for this callback's session."""
        return self.session.report()

    def _extract_llm_usage(
        self,
        response: Any,
        *,
        run_id: UUID,
        kwargs: dict[str, Any],
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage", {})

        model_name = (
            llm_output.get("model_name")
            or llm_output.get("model")
            or self._run_models.get(run_id)
            or self._extract_model_name({}, kwargs)
        )
        input_tokens = token_usage.get("prompt_tokens")
        output_tokens = token_usage.get("completion_tokens")

        if input_tokens is not None and output_tokens is not None:
            return model_name, int(input_tokens), int(output_tokens)

        for generation_group in getattr(response, "generations", []) or []:
            for generation in generation_group or []:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                usage_metadata = getattr(message, "usage_metadata", None) or {}
                model_name = model_name or self._extract_message_model_name(message)
                input_tokens = usage_metadata.get("input_tokens")
                output_tokens = usage_metadata.get("output_tokens")
                if input_tokens is not None and output_tokens is not None:
                    return model_name, int(input_tokens), int(output_tokens)

        return model_name, None, None

    def _extract_model_name(
        self,
        serialized: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> Optional[str]:
        invocation_params = kwargs.get("invocation_params") or {}
        metadata = kwargs.get("metadata") or {}
        return (
            invocation_params.get("model")
            or invocation_params.get("model_name")
            or metadata.get("ls_model_name")
            or metadata.get("model_name")
            or serialized.get("name")
            or serialized.get("id")
        )

    def _extract_message_model_name(self, message: BaseMessage) -> Optional[str]:
        response_metadata = getattr(message, "response_metadata", None) or {}
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        return (
            response_metadata.get("model_name")
            or response_metadata.get("model")
            or additional_kwargs.get("model_name")
            or additional_kwargs.get("model")
        )


class _LangChainTrackedUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens


class _LangChainTrackedResponse:
    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.model = model
        self.usage = _LangChainTrackedUsage(input_tokens, output_tokens)
