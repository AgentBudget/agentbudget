"""AgentBudget - Real-time cost enforcement for AI agent sessions."""

__version__ = "0.3.0"

from .budget import AgentBudget
from .exceptions import AgentBudgetError, BudgetExhausted, InvalidBudget, InvalidCost
from .session import AsyncBudgetSession, BudgetSession, LoopDetected
from .pricing import register_model, register_models

# Drop-in auto-instrumentation API
from ._global import (
    init,
    teardown,
    get_session,
    spent,
    remaining,
    report,
    track,
)

# Explicit per-client tracking
from ._patch import wrap_client

__all__ = [
    # Core classes
    "AgentBudget",
    "AgentBudgetError",
    "AsyncBudgetSession",
    "BudgetExhausted",
    "BudgetSession",
    "InvalidBudget",
    "InvalidCost",
    "LoopDetected",
    # Pricing
    "register_model",
    "register_models",
    # Drop-in API
    "init",
    "teardown",
    "get_session",
    "spent",
    "remaining",
    "report",
    "track",
    # Explicit per-client API
    "wrap_client",
]
