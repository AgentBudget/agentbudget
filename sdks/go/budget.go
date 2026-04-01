package agentbudget

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Budget is the entry point for AgentBudget. Create one per application or agent type,
// then call NewSession() for each agent run.
type Budget struct {
	budget         float64 // enforced ceiling (after finalization reserve)
	reserveBudget  float64 // full budget including reserve
	softLimit      float64
	loopMaxCalls   int
	loopWindowSecs float64
	onSoftLimit    func(Report)
	onHardLimit    func(Report)
	onLoopDetected func(Report)
}

// Option configures a Budget.
type Option func(*budgetConfig)

type budgetConfig struct {
	softLimit      float64
	loopMaxCalls   int
	loopWindowSecs float64
	onSoftLimit    func(Report)
	onHardLimit    func(Report)
	onLoopDetected func(Report)
	finReserve     float64
}

// WithSoftLimit sets the fraction of budget at which the soft-limit callback fires.
// Default: 0.9 (90%).
func WithSoftLimit(fraction float64) Option {
	return func(c *budgetConfig) { c.softLimit = fraction }
}

// WithLoopDetection configures loop detection. maxCalls is the max number of repeated
// calls in window seconds before LoopDetected is raised.
func WithLoopDetection(maxCalls int, windowSeconds float64) Option {
	return func(c *budgetConfig) {
		c.loopMaxCalls = maxCalls
		c.loopWindowSecs = windowSeconds
	}
}

// WithOnSoftLimit sets a callback invoked once when spending exceeds the soft limit.
func WithOnSoftLimit(fn func(Report)) Option {
	return func(c *budgetConfig) { c.onSoftLimit = fn }
}

// WithOnHardLimit sets a callback invoked when BudgetExhausted is raised.
func WithOnHardLimit(fn func(Report)) Option {
	return func(c *budgetConfig) { c.onHardLimit = fn }
}

// WithOnLoopDetected sets a callback invoked when LoopDetected is raised.
func WithOnLoopDetected(fn func(Report)) Option {
	return func(c *budgetConfig) { c.onLoopDetected = fn }
}

// WithFinalizationReserve reserves a fraction of the budget for a final step.
// E.g. WithFinalizationReserve(0.05) on a $1.00 budget fires the hard limit at $0.95.
func WithFinalizationReserve(fraction float64) Option {
	return func(c *budgetConfig) { c.finReserve = fraction }
}

// New creates a Budget with the given maximum spend.
// maxSpend may be a float64, int, or a string like "$5.00" or "5.00".
func New(maxSpend any, opts ...Option) (*Budget, error) {
	raw, err := parseBudget(maxSpend)
	if err != nil {
		return nil, err
	}

	cfg := &budgetConfig{
		softLimit:      0.9,
		loopMaxCalls:   10,
		loopWindowSecs: 60.0,
	}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.finReserve < 0 || cfg.finReserve >= 1 {
		return nil, fmt.Errorf("finalization reserve must be in [0, 1), got %.2f", cfg.finReserve)
	}

	enforced := raw * (1.0 - cfg.finReserve)

	return &Budget{
		budget:         enforced,
		reserveBudget:  raw,
		softLimit:      cfg.softLimit,
		loopMaxCalls:   cfg.loopMaxCalls,
		loopWindowSecs: cfg.loopWindowSecs,
		onSoftLimit:    cfg.onSoftLimit,
		onHardLimit:    cfg.onHardLimit,
		onLoopDetected: cfg.onLoopDetected,
	}, nil
}

// MaxSpend returns the total configured budget including any finalization reserve.
func (b *Budget) MaxSpend() float64 { return b.reserveBudget }

// SessionOption configures a Session.
type SessionOption func(*sessionConfig)

type sessionConfig struct {
	id string
}

// WithSessionID sets a custom session identifier.
func WithSessionID(id string) SessionOption {
	return func(c *sessionConfig) { c.id = id }
}

var sessionCounter uint64

// NewSession creates a new budget-enforced session.
// Each call to the agent should use its own session.
func (b *Budget) NewSession(opts ...SessionOption) *Session {
	cfg := &sessionConfig{}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.id == "" {
		n := atomic.AddUint64(&sessionCounter, 1)
		cfg.id = fmt.Sprintf("sess_%d_%d", time.Now().UnixNano(), n)
	}

	ledger := newLedger(b.budget)
	breaker := newCircuitBreaker(b.softLimit, b.loopMaxCalls, b.loopWindowSecs)
	return newSession(cfg.id, ledger, breaker, b.onSoftLimit, b.onHardLimit, b.onLoopDetected)
}

// parseBudget converts various types to a float64 budget.
func parseBudget(value any) (float64, error) {
	switch v := value.(type) {
	case float64:
		if v <= 0 {
			return 0, &InvalidBudget{Value: fmt.Sprintf("%v", v)}
		}
		return v, nil
	case float32:
		return parseBudget(float64(v))
	case int:
		return parseBudget(float64(v))
	case int64:
		return parseBudget(float64(v))
	case string:
		cleaned := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(v), "$"))
		amount, err := strconv.ParseFloat(cleaned, 64)
		if err != nil || amount <= 0 {
			return 0, &InvalidBudget{Value: v}
		}
		return amount, nil
	default:
		return 0, &InvalidBudget{Value: fmt.Sprintf("%v", value)}
	}
}

// round6 rounds to 6 decimal places (defined in ledger.go, referenced here for clarity).
var _ = math.Round // ensure math imported
