package agentbudget

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Report is the structured cost report returned by Session.Report().
type Report struct {
	SessionID     string         `json:"session_id"`
	Budget        float64        `json:"budget"`
	TotalSpent    float64        `json:"total_spent"`
	Remaining     float64        `json:"remaining"`
	Breakdown     map[string]any `json:"breakdown"`
	DurationSecs  *float64       `json:"duration_seconds"`
	TerminatedBy  string         `json:"terminated_by"` // "" | "budget_exhausted" | "loop_detected"
	EventCount    int            `json:"event_count"`
}

// Session tracks costs for a single agent run. Create via Budget.NewSession().
// It is safe to use from multiple goroutines.
type Session struct {
	id             string
	ledger         *Ledger
	breaker        *CircuitBreaker
	onSoftLimit    func(Report)
	onHardLimit    func(Report)
	onLoopDetected func(Report)

	mu           sync.Mutex
	startTime    time.Time
	endTime      time.Time
	terminatedBy string
}

func newSession(
	id string,
	ledger *Ledger,
	breaker *CircuitBreaker,
	onSoftLimit, onHardLimit, onLoopDetected func(Report),
) *Session {
	return &Session{
		id:             id,
		ledger:         ledger,
		breaker:        breaker,
		onSoftLimit:    onSoftLimit,
		onHardLimit:    onHardLimit,
		onLoopDetected: onLoopDetected,
		startTime:      time.Now(),
	}
}

// ID returns the session identifier.
func (s *Session) ID() string { return s.id }

// Spent returns total dollars spent so far.
func (s *Session) Spent() float64 { return s.ledger.Spent() }

// Remaining returns dollars left in the budget.
func (s *Session) Remaining() float64 { return s.ledger.Remaining() }

// WouldExceed returns true if the given cost would exceed the remaining budget.
// Does not record anything.
func (s *Session) WouldExceed(cost float64) bool { return s.ledger.WouldExceed(cost) }

// WrapUsage records the cost of a completed LLM call given its model and token counts.
// Returns *BudgetExhausted if the budget is exceeded, *LoopDetected if a loop is found.
func (s *Session) WrapUsage(model string, inputTokens, outputTokens int64) error {
	cost, ok := CalculateCost(model, inputTokens, outputTokens)
	if !ok {
		// Unknown model — skip silently (same as Python behaviour)
		return nil
	}

	event := CostEvent{
		Cost:         cost,
		Type:         CostTypeLLM,
		Timestamp:    time.Now(),
		Model:        model,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}
	if err := s.ledger.Record(event); err != nil {
		s.setTerminated("budget_exhausted")
		if s.onHardLimit != nil {
			s.onHardLimit(s.Report())
		}
		return err
	}
	return s.checkAfterRecord(model)
}

// Track manually records a fixed-cost action (tool call, API call, etc.).
// Returns *BudgetExhausted or *LoopDetected if limits are reached.
func (s *Session) Track(cost float64, toolName string, metadata ...map[string]any) error {
	var meta map[string]any
	if len(metadata) > 0 {
		meta = metadata[0]
	}
	event := CostEvent{
		Cost:      cost,
		Type:      CostTypeTool,
		Timestamp: time.Now(),
		ToolName:  toolName,
		Metadata:  meta,
	}
	if err := s.ledger.Record(event); err != nil {
		s.setTerminated("budget_exhausted")
		if s.onHardLimit != nil {
			s.onHardLimit(s.Report())
		}
		return err
	}
	return s.checkAfterRecord(toolName)
}

// ChildSession creates a sub-session with its own budget cap.
// When the child is closed via Close(), its total spend is NOT automatically
// charged to the parent — call parent.Track(child.Spent(), "child:"+child.ID())
// after closing if you want rollup.
func (s *Session) ChildSession(maxSpend float64, opts ...SessionOption) *Session {
	cap := math.Min(maxSpend, s.Remaining())
	childLedger := newLedger(cap)
	childBreaker := newCircuitBreaker(
		s.breaker.softLimitFraction,
		s.breaker.loopMaxCalls,
		s.breaker.loopWindowSeconds,
	)
	childID := fmt.Sprintf("sess_%s_child", s.id)
	cfg := &sessionConfig{}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.id != "" {
		childID = cfg.id
	}
	return newSession(childID, childLedger, childBreaker, nil, nil, nil)
}

// Close marks the session as ended. Safe to call multiple times.
func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.endTime.IsZero() {
		s.endTime = time.Now()
	}
}

// Report returns a structured cost summary for this session.
func (s *Session) Report() Report {
	s.mu.Lock()
	end := s.endTime
	terminated := s.terminatedBy
	s.mu.Unlock()

	var dur *float64
	if !s.startTime.IsZero() {
		ref := end
		if ref.IsZero() {
			ref = time.Now()
		}
		d := ref.Sub(s.startTime).Seconds()
		d = math.Round(d*100) / 100
		dur = &d
	}

	spent := s.ledger.Spent()
	remaining := s.ledger.Remaining()
	events := s.ledger.Events()

	return Report{
		SessionID:    s.id,
		Budget:       s.ledger.Budget(),
		TotalSpent:   round6(spent),
		Remaining:    round6(remaining),
		Breakdown:    s.ledger.Breakdown(),
		DurationSecs: dur,
		TerminatedBy: terminated,
		EventCount:   len(events),
	}
}

// checkAfterRecord runs soft-limit and loop-detection checks after a successful record.
func (s *Session) checkAfterRecord(key string) error {
	spent := s.ledger.Spent()
	budget := s.ledger.Budget()

	if msg := s.breaker.CheckBudget(spent, budget); msg != "" {
		if s.onSoftLimit != nil {
			s.onSoftLimit(s.Report())
		}
	}

	if key != "" && s.breaker.CheckLoop(key) {
		s.setTerminated("loop_detected")
		if s.onLoopDetected != nil {
			s.onLoopDetected(s.Report())
		}
		return &LoopDetected{Key: key}
	}
	return nil
}

func (s *Session) setTerminated(reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.terminatedBy == "" {
		s.terminatedBy = reason
	}
	if s.endTime.IsZero() {
		s.endTime = time.Now()
	}
}
