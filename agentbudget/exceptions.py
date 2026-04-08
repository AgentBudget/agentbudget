"""Exceptions for AgentBudget."""


class AgentBudgetError(Exception):
    """Base exception for all AgentBudget errors."""


class BudgetExhausted(AgentBudgetError):
    """Raised when a session exceeds its allocated budget."""

    def __init__(self, budget: float, spent: float):
        self.budget = budget
        self.spent = spent
        super().__init__(
            f"Budget exhausted: spent ${spent:.4f} of ${budget:.2f} budget"
        )


class InvalidBudget(AgentBudgetError):
    """Raised when a budget value is invalid."""

    def __init__(self, value: str):
        self.value = value
        super().__init__(f"Invalid budget value: {value!r}")


class InvalidCost(AgentBudgetError):
    """Raised when a tracked cost value is not finite and non-negative."""

    def __init__(self, cost: float):
        self.cost = cost
        super().__init__(
            f"Invalid cost value: {cost!r}. "
            "Cost must be a finite, non-negative number."
        )
