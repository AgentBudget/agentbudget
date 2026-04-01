package agentbudget

import (
	"fmt"
	"sync"
	"time"
)

// CircuitBreaker monitors budget usage (soft limit) and detects runaway loops.
type CircuitBreaker struct {
	mu                   sync.Mutex
	softLimitFraction    float64
	softLimitTriggered   bool
	loopMaxCalls         int
	loopWindowSeconds    float64
	callLog              map[string][]time.Time
}

func newCircuitBreaker(softLimit float64, maxCalls int, windowSeconds float64) *CircuitBreaker {
	return &CircuitBreaker{
		softLimitFraction: softLimit,
		loopMaxCalls:      maxCalls,
		loopWindowSeconds: windowSeconds,
		callLog:           map[string][]time.Time{},
	}
}

// CheckBudget returns a warning message if the soft limit is reached (fires once).
// Returns empty string if not triggered.
func (cb *CircuitBreaker) CheckBudget(spent, budget float64) string {
	if budget <= 0 {
		return ""
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()
	fraction := spent / budget
	if fraction >= cb.softLimitFraction && !cb.softLimitTriggered {
		cb.softLimitTriggered = true
		return fmt.Sprintf("soft limit reached: %.0f%% of budget used ($%.4f / $%.2f)", fraction*100, spent, budget)
	}
	return ""
}

// CheckLoop records a call for the given key and returns true if a loop is detected.
func (cb *CircuitBreaker) CheckLoop(key string) bool {
	if cb.loopMaxCalls <= 0 {
		return false
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Duration(cb.loopWindowSeconds * float64(time.Second)))

	// Prune old entries outside window
	existing := cb.callLog[key]
	pruned := existing[:0]
	for _, t := range existing {
		if t.After(cutoff) {
			pruned = append(pruned, t)
		}
	}
	pruned = append(pruned, now)
	cb.callLog[key] = pruned

	return len(pruned) >= cb.loopMaxCalls
}
