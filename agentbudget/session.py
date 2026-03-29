"""Budget session — context manager that wraps LLM and tool calls."""

from __future__ import annotations

import time
from typing import Any, Optional, TypeVar

from .circuit_breaker import CircuitBreaker
from .exceptions import BudgetExhausted
from .ledger import Ledger
from .pricing import calculate_llm_cost
from .types import CostEvent, CostType, generate_session_id

T = TypeVar("T")


class LoopDetected(Exception):
    """Raised when the circuit breaker detects a call loop."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Loop detected: repeated calls to {key!r}")


class BudgetSession:
    """Tracks costs for a single agent session.

    Used as a context manager:
        with budget.session() as session:
            response = session.wrap(openai_call(...))
            session.track(tool_call(), cost=0.01)
    """

    def __init__(
        self,
        ledger: Ledger,
        session_id: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        on_soft_limit: Optional[Any] = None,
        on_hard_limit: Optional[Any] = None,
        on_loop_detected: Optional[Any] = None,
    ):
        self._ledger = ledger
        self._session_id = session_id or generate_session_id()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._on_soft_limit = on_soft_limit
        self._on_hard_limit = on_hard_limit
        self._on_loop_detected = on_loop_detected
        self._parent: Optional["BudgetSession"] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._terminated_by: Optional[str] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def spent(self) -> float:
        return self._ledger.spent

    @property
    def remaining(self) -> float:
        return self._ledger.remaining

    def __enter__(self) -> "BudgetSession":
        self._start_time = time.time()
        return self

    def _check_after_record(self, call_key: Optional[str] = None) -> None:
        """Run circuit breaker checks after recording a cost event."""
        # Soft limit check
        warning = self._circuit_breaker.check_budget(
            self._ledger.spent, self._ledger.budget
        )
        if warning and self._on_soft_limit:
            self._on_soft_limit(self.report())

        # Loop detection
        if call_key:
            if self._circuit_breaker.check_loop(call_key):
                self._terminated_by = "loop_detected"
                if self._on_loop_detected:
                    self._on_loop_detected(self.report())
                raise LoopDetected(call_key)

    def wrap(self, response: T) -> T:
        """Wrap an LLM API response and record its cost.

        Extracts model and token usage from the response object.
        Supports OpenAI-style response objects with a `usage` attribute.
        """
        model = _extract_model(response)
        input_tokens, output_tokens = _extract_usage(response)

        cost = None
        if model and input_tokens is not None and output_tokens is not None:
            cost = calculate_llm_cost(model, input_tokens, output_tokens)

        if cost is not None:
            event = CostEvent(
                cost=cost,
                cost_type=CostType.LLM,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._ledger.record(event)
            self._check_after_record(call_key=model)

        return response

    def track(
        self,
        result: T,
        cost: float,
        tool_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> T:
        """Track a tool/API call with a known cost.

        Returns the result passthrough so it can be used inline:
            data = session.track(call_api(), cost=0.01, tool_name="my_api")
        """
        event = CostEvent(
            cost=cost,
            cost_type=CostType.TOOL,
            tool_name=tool_name,
            metadata=metadata,
        )
        self._ledger.record(event)
        self._check_after_record(call_key=tool_name)
        return result

    def track_tool(self, cost: float, tool_name: Optional[str] = None):
        """Decorator to track a function's cost on every call.

        Usage:
            @session.track_tool(cost=0.02, tool_name="search")
            def my_tool(query):
                return api.search(query)
        """
        import functools

        def decorator(func):
            name = tool_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                self.track(result, cost=cost, tool_name=name)
                return result

            return wrapper

        return decorator

    def child_session(
        self,
        max_spend: float,
        session_id: Optional[str] = None,
    ) -> "BudgetSession":
        """Create a child session with its own sub-budget.

        The child's budget is capped at max_spend AND the parent's remaining
        budget (whichever is smaller). When the child session ends, its total
        spend is charged to the parent.

        Usage:
            with budget.session() as parent:
                child = parent.child_session(max_spend=1.0)
                with child:
                    child.track("result", cost=0.50, tool_name="sub_task")
                # parent is now charged $0.50
        """
        cap = min(max_spend, self.remaining)
        child_ledger = Ledger(budget=cap)
        child = BudgetSession(
            ledger=child_ledger,
            session_id=session_id,
            circuit_breaker=CircuitBreaker(),
        )
        child._parent = self
        return child

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._end_time = time.time()
        if exc_type and issubclass(exc_type, BudgetExhausted):
            self._terminated_by = "budget_exhausted"
            if self._on_hard_limit:
                self._on_hard_limit(self.report())
        elif exc_type and issubclass(exc_type, LoopDetected):
            self._terminated_by = "loop_detected"

        # Roll up child spend to parent
        if self._parent is not None and self._ledger.spent > 0:
            self._parent.track(
                None,
                cost=self._ledger.spent,
                tool_name=f"child:{self._session_id}",
            )

    def report(self) -> dict[str, Any]:
        """Generate a structured cost report for this session."""
        duration = None
        if self._start_time:
            end = self._end_time or time.time()
            duration = round(end - self._start_time, 2)

        return {
            "session_id": self._session_id,
            "budget": self._ledger.budget,
            "total_spent": round(self._ledger.spent, 6),
            "remaining": round(self._ledger.remaining, 6),
            "breakdown": self._ledger.breakdown(),
            "duration_seconds": duration,
            "terminated_by": self._terminated_by,
            "events": [e.to_dict() for e in self._ledger.events],
        }


class AsyncBudgetSession(BudgetSession):
    """Async version of BudgetSession.

    Usage:
        async with budget.async_session() as session:
            response = await session.wrap(openai_acall(...))
            session.track(tool_result, cost=0.01)
    """

    async def __aenter__(self) -> "AsyncBudgetSession":
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    async def wrap_async(self, coroutine):
        """Await an LLM coroutine and record its cost.

        Usage:
            response = await session.wrap_async(
                client.chat.completions.acreate(...)
            )
        """
        response = await coroutine
        return self.wrap(response)

    def track_tool(self, cost: float, tool_name: Optional[str] = None):
        """Decorator that works for both sync and async functions."""
        import asyncio
        import functools

        def decorator(func):
            name = tool_name or func.__name__

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    result = await func(*args, **kwargs)
                    self.track(result, cost=cost, tool_name=name)
                    return result
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    result = func(*args, **kwargs)
                    self.track(result, cost=cost, tool_name=name)
                    return result
                return sync_wrapper

        return decorator


def _extract_model(response: Any) -> Optional[str]:
    """Extract model name from an LLM response object."""
    return getattr(response, "model", None)


def _extract_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    """Extract token usage from an LLM response object.

    Supports both OpenAI-style (prompt_tokens/completion_tokens)
    and Anthropic-style (input_tokens/output_tokens) responses.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None

    # OpenAI style
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)

    # Anthropic style fallback
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)

    return input_tokens, output_tokens
