"""Global state for drop-in auto-instrumentation.

Usage:
    import agentbudget
    agentbudget.init(budget="$5.00")

    # All OpenAI/Anthropic calls are now tracked automatically
    client = openai.OpenAI()
    response = client.chat.completions.create(...)

    print(agentbudget.spent())
    print(agentbudget.remaining())
    print(agentbudget.report())

    agentbudget.teardown()
"""

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .budget import AgentBudget
from .session import BudgetSession
from ._patch import patch_openai, patch_anthropic, unpatch_all


@dataclass
class _DropInState:
    """Context-local state for drop-in auto-instrumentation."""

    session: BudgetSession
    scope_id: tuple[str, int]
    token: Optional[Token[Optional["_DropInState"]]] = None


_current_state: ContextVar[Optional[_DropInState]] = ContextVar(
    "agentbudget_current_state",
    default=None,
)
_patch_lock = threading.Lock()
_active_contexts = 0


def _current_scope_id() -> tuple[str, int]:
    """Identify the current logical execution scope.

    Async tasks get their own scope. Synchronous code falls back to the
    current thread, which keeps thread-local and task-local init() calls
    isolated from each other.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None

    if task is not None:
        return ("task", id(task))
    return ("thread", threading.get_ident())


def _get_state() -> Optional[_DropInState]:
    return _current_state.get()


def _owns_state(state: _DropInState) -> bool:
    return state.scope_id == _current_scope_id()


def _acquire_patches() -> None:
    """Install global SDK patches while at least one context is active."""
    global _active_contexts

    with _patch_lock:
        if _active_contexts == 0:
            patch_openai(_get_session)
            patch_anthropic(_get_session)
        _active_contexts += 1


def _release_patches() -> None:
    """Remove global SDK patches after the last active context exits."""
    global _active_contexts

    with _patch_lock:
        if _active_contexts == 0:
            return
        _active_contexts -= 1
        if _active_contexts == 0:
            unpatch_all()


def _teardown_state(state: _DropInState) -> dict[str, Any]:
    """Close an owned session and restore the previous visible context."""
    try:
        state.session.__exit__(None, None, None)
        return state.session.report()
    finally:
        if state.token is not None:
            _current_state.reset(state.token)
        else:
            _current_state.set(None)
        _release_patches()


def _get_session() -> Optional[BudgetSession]:
    """Get the active context-local session. Used by patched methods."""
    state = _get_state()
    if state is None:
        return None
    return state.session


def init(
    budget: str | float | int,
    soft_limit: float = 0.9,
    max_repeated_calls: int = 10,
    loop_window_seconds: float = 60.0,
    on_soft_limit: Optional[Callable] = None,
    on_hard_limit: Optional[Callable] = None,
    on_loop_detected: Optional[Callable] = None,
    webhook_url: Optional[str] = None,
    session_id: Optional[str] = None,
) -> BudgetSession:
    """Initialize budget tracking for the current thread/task context.

    Patches OpenAI and Anthropic clients so every LLM call is
    automatically tracked. Call teardown() to stop tracking.

    Returns the active BudgetSession for manual tracking if needed.
    """
    state = _get_state()
    if state is not None and _owns_state(state):
        teardown()

    budget_obj = AgentBudget(
        max_spend=budget,
        soft_limit=soft_limit,
        max_repeated_calls=max_repeated_calls,
        loop_window_seconds=loop_window_seconds,
        on_soft_limit=on_soft_limit,
        on_hard_limit=on_hard_limit,
        on_loop_detected=on_loop_detected,
        webhook_url=webhook_url,
    )
    session = budget_obj.session(session_id=session_id)
    scope_id = _current_scope_id()

    entered = False
    _acquire_patches()
    try:
        session.__enter__()
        entered = True
        new_state = _DropInState(session=session, scope_id=scope_id)
        new_state.token = _current_state.set(new_state)
    except Exception:
        if entered:
            session.__exit__(None, None, None)
        _release_patches()
        raise

    return session


def teardown() -> Optional[dict[str, Any]]:
    """Stop tracking in the current context.

    Returns the final cost report, or None if not initialized.
    """
    state = _get_state()
    if state is None or not _owns_state(state):
        return None
    return _teardown_state(state)


def get_session() -> Optional[BudgetSession]:
    """Get the active session visible in the current context."""
    return _get_session()


def spent() -> float:
    """Get total amount spent in the current session."""
    session = _get_session()
    if session is None:
        return 0.0
    return session.spent


def remaining() -> float:
    """Get remaining budget in the current session."""
    session = _get_session()
    if session is None:
        return 0.0
    return session.remaining


def report() -> Optional[dict[str, Any]]:
    """Get the cost report for the current session."""
    session = _get_session()
    if session is None:
        return None
    return session.report()


def track(
    result: Any = None,
    cost: float = 0.0,
    tool_name: Optional[str] = None,
) -> Any:
    """Track a tool/API call cost in the active context-local session."""
    session = _get_session()
    if session is None:
        raise RuntimeError("agentbudget.init() must be called before tracking costs")
    return session.track(result, cost=cost, tool_name=tool_name)
