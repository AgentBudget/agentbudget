package agentbudget

import "fmt"

// BudgetExhausted is raised when a recorded cost pushes total spend over the budget.
type BudgetExhausted struct {
	Budget float64
	Spent  float64
}

func (e *BudgetExhausted) Error() string {
	return fmt.Sprintf("budget exhausted: spent $%.4f of $%.2f budget", e.Spent, e.Budget)
}

// LoopDetected is raised when the same model or tool is called too many times
// within the configured time window.
type LoopDetected struct {
	Key string
}

func (e *LoopDetected) Error() string {
	return fmt.Sprintf("loop detected: repeated calls to %q", e.Key)
}

// InvalidBudget is raised when the budget value cannot be parsed or is non-positive.
type InvalidBudget struct {
	Value string
}

func (e *InvalidBudget) Error() string {
	return fmt.Sprintf("invalid budget value: %q", e.Value)
}
