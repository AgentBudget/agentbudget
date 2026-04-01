package agentbudget

import (
	"math"
	"sync"
	"time"
)

// CostType distinguishes LLM calls from manual tool/API cost entries.
type CostType string

const (
	CostTypeLLM  CostType = "llm"
	CostTypeTool CostType = "tool"
)

// CostEvent records a single costed action within a session.
type CostEvent struct {
	Cost         float64
	Type         CostType
	Timestamp    time.Time
	Model        string // set for LLM events
	InputTokens  int64  // set for LLM events
	OutputTokens int64  // set for LLM events
	ToolName     string // set for tool events
	Metadata     map[string]any
}

// Ledger is a thread-safe running total of costs within a session.
type Ledger struct {
	mu     sync.Mutex
	budget float64
	spent  float64
	events []CostEvent
}

func newLedger(budget float64) *Ledger {
	return &Ledger{budget: budget}
}

// Budget returns the configured budget ceiling.
func (l *Ledger) Budget() float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.budget
}

// Spent returns the total amount spent so far.
func (l *Ledger) Spent() float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.spent
}

// Remaining returns how much budget is left.
func (l *Ledger) Remaining() float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return math.Max(0, l.budget-l.spent)
}

// WouldExceed returns true if adding cost would exceed the budget,
// without recording anything.
func (l *Ledger) WouldExceed(cost float64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.spent+cost > l.budget
}

// Record adds a cost event. Returns *BudgetExhausted if the budget is exceeded.
func (l *Ledger) Record(event CostEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	newTotal := l.spent + event.Cost
	if newTotal > l.budget {
		return &BudgetExhausted{Budget: l.budget, Spent: newTotal}
	}
	l.spent = newTotal
	l.events = append(l.events, event)
	return nil
}

// Events returns a copy of all recorded events.
func (l *Ledger) Events() []CostEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]CostEvent, len(l.events))
	copy(out, l.events)
	return out
}

// Breakdown returns a summary map grouped by type and model/tool.
func (l *Ledger) Breakdown() map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()

	llmTotal := 0.0
	llmCalls := 0
	byModel := map[string]float64{}

	toolTotal := 0.0
	toolCalls := 0
	byTool := map[string]float64{}

	for _, e := range l.events {
		switch e.Type {
		case CostTypeLLM:
			llmTotal += e.Cost
			llmCalls++
			if e.Model != "" {
				byModel[e.Model] += e.Cost
			}
		case CostTypeTool:
			toolTotal += e.Cost
			toolCalls++
			if e.ToolName != "" {
				byTool[e.ToolName] += e.Cost
			}
		}
	}

	return map[string]any{
		"llm": map[string]any{
			"total":    round6(llmTotal),
			"calls":    llmCalls,
			"by_model": byModel,
		},
		"tools": map[string]any{
			"total":   round6(toolTotal),
			"calls":   toolCalls,
			"by_tool": byTool,
		},
	}
}

func round6(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
