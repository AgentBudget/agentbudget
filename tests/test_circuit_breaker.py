"""Tests for circuit breaker and loop detection."""

import time

from agentbudget.circuit_breaker import (
    CircuitBreaker,
    LoopDetector,
    LoopDetectorConfig,
)


def test_loop_detector_no_loop():
    detector = LoopDetector(LoopDetectorConfig(max_repeated_calls=5))
    for _ in range(4):
        assert detector.record_call("tool_a") is False


def test_loop_detector_detects_loop():
    detector = LoopDetector(LoopDetectorConfig(max_repeated_calls=3))
    detector.record_call("tool_a")
    detector.record_call("tool_a")
    assert detector.record_call("tool_a") is True


def test_loop_detector_different_keys():
    detector = LoopDetector(LoopDetectorConfig(max_repeated_calls=3))
    detector.record_call("tool_a")
    detector.record_call("tool_b")
    detector.record_call("tool_a")
    assert detector.record_call("tool_b") is False  # only 2, not >= 3


def test_loop_detector_reset():
    detector = LoopDetector(LoopDetectorConfig(max_repeated_calls=2))
    detector.record_call("tool_a")
    detector.record_call("tool_a")
    detector.reset()
    assert detector.record_call("tool_a") is False


def test_loop_detector_time_window():
    detector = LoopDetector(LoopDetectorConfig(max_repeated_calls=2, time_window_seconds=0.1))
    detector.record_call("tool_a")
    detector.record_call("tool_a")
    time.sleep(0.15)
    # Old calls should be pruned
    assert detector.record_call("tool_a") is False


def test_circuit_breaker_no_warning_below_soft_limit():
    cb = CircuitBreaker(soft_limit_fraction=0.9)
    assert cb.check_budget(spent=4.0, budget=5.0) is None


def test_circuit_breaker_soft_limit_warning():
    cb = CircuitBreaker(soft_limit_fraction=0.9)
    warning = cb.check_budget(spent=4.5, budget=5.0)
    assert warning is not None
    assert "90%" in warning


def test_circuit_breaker_soft_limit_fires_once():
    cb = CircuitBreaker(soft_limit_fraction=0.9)
    first = cb.check_budget(spent=4.5, budget=5.0)
    second = cb.check_budget(spent=4.6, budget=5.0)
    assert first is not None
    assert second is None


def test_circuit_breaker_soft_limit_triggered_property():
    cb = CircuitBreaker(soft_limit_fraction=0.9)
    assert cb.soft_limit_triggered is False
    cb.check_budget(spent=4.5, budget=5.0)
    assert cb.soft_limit_triggered is True


def test_circuit_breaker_check_loop():
    cb = CircuitBreaker(loop_config=LoopDetectorConfig(max_repeated_calls=2))
    assert cb.check_loop("tool_x") is False
    assert cb.check_loop("tool_x") is True
