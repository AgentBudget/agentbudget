package agentbudget_test

import (
	"errors"
	"fmt"
	"log"

	agentbudget "github.com/AgentBudget/agentbudget/sdks/go"
)

func Example_basicSession() {
	budget, err := agentbudget.New("$5.00")
	if err != nil {
		log.Fatal(err)
	}

	session := budget.NewSession()
	defer session.Close()

	// After your LLM call, record usage:
	err = session.WrapUsage("gpt-4o", 1000, 500)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("spent:     $%.4f\n", session.Spent())
	fmt.Printf("remaining: $%.4f\n", session.Remaining())
}

func Example_trackTool() {
	budget, _ := agentbudget.New(10.0)
	session := budget.NewSession()
	defer session.Close()

	// Track a tool/API call with a known cost
	if err := session.Track(0.05, "serp_api"); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("spent: $%.4f\n", session.Spent())
}

func Example_budgetExhausted() {
	budget, _ := agentbudget.New(0.001) // very small budget
	session := budget.NewSession()
	defer session.Close()

	err := session.WrapUsage("gpt-4o", 10000, 5000)
	if err != nil {
		var exhausted *agentbudget.BudgetExhausted
		if errors.As(err, &exhausted) {
			fmt.Printf("hard limit hit: budget=%.4f spent=%.4f\n",
				exhausted.Budget, exhausted.Spent)
		}
	}
}

func Example_callbacks() {
	budget, _ := agentbudget.New("$1.00",
		agentbudget.WithSoftLimit(0.8),
		agentbudget.WithOnSoftLimit(func(r agentbudget.Report) {
			fmt.Printf("⚠️  soft limit: spent $%.4f of $%.2f\n", r.TotalSpent, r.Budget)
		}),
		agentbudget.WithOnHardLimit(func(r agentbudget.Report) {
			fmt.Printf("🛑 hard limit hit after $%.4f\n", r.TotalSpent)
		}),
	)

	session := budget.NewSession()
	defer session.Close()

	_ = session.WrapUsage("gpt-4o", 100000, 50000)
}

func Example_childSession() {
	budget, _ := agentbudget.New(10.0)
	parent := budget.NewSession()
	defer parent.Close()

	child := parent.ChildSession(2.0)
	defer func() {
		child.Close()
		// Roll up child spend to parent manually
		if child.Spent() > 0 {
			_ = parent.Track(child.Spent(), "child:"+child.ID())
		}
	}()

	_ = child.WrapUsage("gpt-4o-mini", 5000, 2000)
	fmt.Printf("child spent: $%.4f\n", child.Spent())
}

func Example_customPricing() {
	if err := agentbudget.RegisterModel("my-model", 5.00, 20.00); err != nil {
		log.Fatal(err)
	}

	budget, _ := agentbudget.New(5.0)
	session := budget.NewSession()
	defer session.Close()

	_ = session.WrapUsage("my-model", 1000, 500)
	fmt.Printf("spent: $%.6f\n", session.Spent())
}

func Example_wouldExceed() {
	budget, _ := agentbudget.New(1.0, agentbudget.WithFinalizationReserve(0.05))
	session := budget.NewSession()
	defer session.Close()

	// Do work...
	_ = session.WrapUsage("gpt-4o", 50000, 25000)

	// Before a final expensive call, check first
	estimatedFinalCost := 0.10
	if session.WouldExceed(estimatedFinalCost) {
		fmt.Println("not enough budget for final call — wrapping up")
		return
	}

	_ = session.WrapUsage("gpt-4o", 10000, 5000)
}
