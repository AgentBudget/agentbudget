"""Circuit breaker — loop detection and budget threshold warnings."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoopDetectorConfig:
    """Configuration for loop detection."""

    max_repeated_calls: int = 10
    time_window_seconds: float = 60.0


class LoopDetector:
    """Detects when the same tool/model is called repeatedly in a short window."""

    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        self._config = config or LoopDetectorConfig()
        self._call_log: dict[str, list[float]] = defaultdict(list)

    def record_call(self, key: str) -> bool:
        """Record a call and return True if a loop is detected."""
        now = time.time()
        cutoff = now - self._config.time_window_seconds

        # Prune old entries
        self._call_log[key] = [
            t for t in self._call_log[key] if t > cutoff
        ]
        self._call_log[key].append(now)

        return len(self._call_log[key]) >= self._config.max_repeated_calls

    def reset(self) -> None:
        """Clear all recorded calls."""
        self._call_log.clear()


class CircuitBreaker:
    """Monitors budget usage and detects runaway loops."""

    def __init__(
        self,
        soft_limit_fraction: float = 0.9,
        loop_config: Optional[LoopDetectorConfig] = None,
    ):
        self._soft_limit_fraction = soft_limit_fraction
        self._loop_detector = LoopDetector(loop_config)
        self._soft_limit_triggered = False

    @property
    def soft_limit_triggered(self) -> bool:
        return self._soft_limit_triggered

    def check_budget(self, spent: float, budget: float) -> Optional[str]:
        """Check budget thresholds. Returns warning message or None."""
        if budget <= 0:
            return None
        fraction = spent / budget
        if fraction >= self._soft_limit_fraction and not self._soft_limit_triggered:
            self._soft_limit_triggered = True
            return f"Soft limit reached: {fraction:.0%} of budget used (${spent:.4f} / ${budget:.2f})"
        return None

    def check_loop(self, key: str) -> bool:
        """Record a call and return True if a loop is detected."""
        return self._loop_detector.record_call(key)
