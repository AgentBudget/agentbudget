"""Tests for cost validation — negative, NaN, and infinite cost rejection.

Covers the fix for:
  https://github.com/AgentBudget/agentbudget/issues/21
"""

import math

import pytest

from agentbudget.exceptions import InvalidCost
from agentbudget.ledger import Ledger, validate_cost
from agentbudget.session import BudgetSession
from agentbudget.types import CostEvent, CostType


# ---------------------------------------------------------------------------
# validate_cost() unit tests
# ---------------------------------------------------------------------------


class TestValidateCost:
    """Direct tests for the validate_cost helper."""

    def test_zero_is_valid(self):
        validate_cost(0.0)

    def test_positive_float_is_valid(self):
        validate_cost(1.23)

    def test_positive_int_is_valid(self):
        validate_cost(5)

    def test_negative_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(-1.0)

    def test_negative_small_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(-0.001)

    def test_nan_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(float("nan"))

    def test_positive_inf_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(float("inf"))

    def test_negative_inf_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(float("-inf"))

    def test_math_nan_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(math.nan)

    def test_math_inf_raises(self):
        with pytest.raises(InvalidCost):
            validate_cost(math.inf)


# ---------------------------------------------------------------------------
# Ledger.record() validation tests
# ---------------------------------------------------------------------------


class TestLedgerRecordValidation:
    """Ensure Ledger.record() rejects invalid cost values."""

    def _make_event(self, cost: float) -> CostEvent:
        return CostEvent(cost=cost, cost_type=CostType.TOOL, tool_name="test")

    def test_negative_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(-1.0))

    def test_nan_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(float("nan")))

    def test_positive_inf_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(float("inf")))

    def test_negative_inf_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(float("-inf")))

    def test_invalid_cost_does_not_mutate_spent(self):
        ledger = Ledger(budget=5.0)
        ledger.record(self._make_event(1.0))  # valid
        assert ledger.spent == 1.0

        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(-5.0))

        # Spent must remain unchanged after invalid record attempt
        assert ledger.spent == 1.0
        assert ledger.remaining == 4.0

    def test_invalid_cost_does_not_add_event(self):
        ledger = Ledger(budget=5.0)
        ledger.record(self._make_event(1.0))
        assert len(ledger.events) == 1

        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(float("nan")))

        assert len(ledger.events) == 1

    def test_invalid_cost_does_not_corrupt_breakdown(self):
        ledger = Ledger(budget=5.0)
        ledger.record(self._make_event(1.0))

        with pytest.raises(InvalidCost):
            ledger.record(self._make_event(-2.0))

        bd = ledger.breakdown()
        assert bd["tools"]["total"] == 1.0
        assert bd["tools"]["calls"] == 1


# ---------------------------------------------------------------------------
# Ledger.would_exceed() validation tests
# ---------------------------------------------------------------------------


class TestLedgerWouldExceedValidation:
    def test_negative_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.would_exceed(-1.0)

    def test_nan_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.would_exceed(float("nan"))

    def test_inf_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with pytest.raises(InvalidCost):
            ledger.would_exceed(float("inf"))


# ---------------------------------------------------------------------------
# BudgetSession.track() validation tests
# ---------------------------------------------------------------------------


class TestSessionTrackValidation:
    """Ensure session.track() rejects invalid costs at the API boundary."""

    def test_negative_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.track("result", cost=-1.0, tool_name="api")

    def test_nan_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.track("result", cost=float("nan"), tool_name="api")

    def test_inf_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.track("result", cost=float("inf"), tool_name="api")

    def test_negative_inf_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.track("result", cost=float("-inf"), tool_name="api")

    def test_invalid_cost_does_not_alter_session_state(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            session.track("a", cost=1.0, tool_name="ok")

            with pytest.raises(InvalidCost):
                session.track("b", cost=-5.0, tool_name="bad")

            assert session.spent == 1.0
            assert session.remaining == 4.0

    def test_zero_cost_is_accepted(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            session.track("result", cost=0.0, tool_name="free")
            assert session.spent == 0.0


# ---------------------------------------------------------------------------
# BudgetSession.would_exceed() validation tests
# ---------------------------------------------------------------------------


class TestSessionWouldExceedValidation:
    def test_negative_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.would_exceed(-1.0)

    def test_nan_cost_rejected(self):
        ledger = Ledger(budget=5.0)
        with BudgetSession(ledger) as session:
            with pytest.raises(InvalidCost):
                session.would_exceed(float("nan"))


# ---------------------------------------------------------------------------
# Regression: negative cost must never increase remaining budget
# ---------------------------------------------------------------------------


class TestNegativeCostRegression:
    """Reproduces the scenario from the issue where a negative cost
    increased the remaining budget to more than the original limit.
    """

    def test_negative_cost_cannot_increase_remaining(self):
        ledger = Ledger(budget=1.0)
        with pytest.raises(InvalidCost):
            ledger.record(CostEvent(cost=-5.0, cost_type=CostType.TOOL))

        assert ledger.spent == 0.0
        assert ledger.remaining == 1.0
