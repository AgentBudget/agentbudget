package agentbudget_test

import (
	"errors"
	"testing"

	agentbudget "github.com/AgentBudget/agentbudget/sdks/go"
)

// ---------------------------------------------------------------------------
// Budget creation
// ---------------------------------------------------------------------------

func TestNew_BudgetParsing(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		input   any
		wantMax float64
	}{
		{"dollar-string", "$5.00", 5.00},
		{"plain-string", "5.00", 5.00},
		{"float64", 5.0, 5.0},
		{"int", 10, 10.0},
		{"int64", int64(3), 3.0},
		{"float32", float32(2.5), 2.5},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			b, err := agentbudget.New(tc.input)
			if err != nil {
				t.Fatalf("New(%v) unexpected error: %v", tc.input, err)
			}
			if b.MaxSpend() != tc.wantMax {
				t.Errorf("MaxSpend() = %v, want %v", b.MaxSpend(), tc.wantMax)
			}
		})
	}
}

func TestNew_InvalidBudget(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		input any
	}{
		{"zero-float", 0.0},
		{"negative-float", -1.0},
		{"zero-int", 0},
		{"negative-string", "-5.00"},
		{"zero-string", "0.00"},
		{"garbage-string", "not-a-number"},
		{"empty-string", ""},
		{"unsupported-type", []int{1, 2}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := agentbudget.New(tc.input)
			if err == nil {
				t.Fatalf("New(%v) expected error, got nil", tc.input)
			}
			var inv *agentbudget.InvalidBudget
			if !errors.As(err, &inv) {
				t.Errorf("error type = %T, want *InvalidBudget", err)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WrapUsage — cost recording for known models
// ---------------------------------------------------------------------------

func TestWrapUsage_KnownModels(t *testing.T) {
	t.Parallel()

	// gpt-4o: $2.50/M input, $10.00/M output
	// 1000 input + 500 output = 0.0025 + 0.005 = 0.0075
	const gpt4oCost = 1000*2.50/1e6 + 500*10.00/1e6

	// claude-opus-4-6: $5.00/M input, $25.00/M output
	// 1000 input + 500 output = 0.005 + 0.0125 = 0.0175
	const claudeOpusCost = 1000*5.00/1e6 + 500*25.00/1e6

	cases := []struct {
		name         string
		model        string
		inputTokens  int64
		outputTokens int64
		wantCost     float64
	}{
		{"gpt-4o", "gpt-4o", 1000, 500, gpt4oCost},
		{"claude-opus-4-6", "claude-opus-4-6", 1000, 500, claudeOpusCost},
		{"gpt-4o-mini", "gpt-4o-mini", 10000, 5000, 10000*0.15/1e6 + 5000*0.60/1e6},
		{"gpt-4o-date-suffix", "gpt-4o-2025-06-15", 1000, 500, gpt4oCost},
		{"openrouter-prefix", "openai/gpt-4o", 1000, 500, gpt4oCost},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			b, _ := agentbudget.New(100.0)
			sess := b.NewSession()
			defer sess.Close()

			if err := sess.WrapUsage(tc.model, tc.inputTokens, tc.outputTokens); err != nil {
				t.Fatalf("WrapUsage error: %v", err)
			}
			if diff := sess.Spent() - tc.wantCost; diff < -1e-9 || diff > 1e-9 {
				t.Errorf("Spent() = %.9f, want %.9f (diff %.9f)", sess.Spent(), tc.wantCost, diff)
			}
		})
	}
}

func TestWrapUsage_UnknownModelSilent(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(1.0)
	sess := b.NewSession()
	defer sess.Close()

	err := sess.WrapUsage("unknown-model-xyz", 1000, 500)
	if err != nil {
		t.Errorf("WrapUsage with unknown model should be silent, got: %v", err)
	}
	if sess.Spent() != 0 {
		t.Errorf("Spent() = %v, want 0 for unknown model", sess.Spent())
	}
}

// ---------------------------------------------------------------------------
// WrapUsage — BudgetExhausted
// ---------------------------------------------------------------------------

func TestWrapUsage_BudgetExhausted(t *testing.T) {
	t.Parallel()

	// budget just below what gpt-4o costs for 10k/5k tokens
	// gpt-4o 10k in + 5k out = 10000*2.50/1e6 + 5000*10.00/1e6 = 0.025 + 0.05 = 0.075
	b, _ := agentbudget.New(0.001) // well under 0.075
	sess := b.NewSession()
	defer sess.Close()

	err := sess.WrapUsage("gpt-4o", 10000, 5000)
	if err == nil {
		t.Fatal("expected BudgetExhausted error, got nil")
	}

	var exhausted *agentbudget.BudgetExhausted
	if !errors.As(err, &exhausted) {
		t.Fatalf("error type = %T, want *BudgetExhausted", err)
	}
	if exhausted.Budget <= 0 {
		t.Errorf("BudgetExhausted.Budget = %v, want > 0", exhausted.Budget)
	}
	if exhausted.Spent <= exhausted.Budget {
		t.Errorf("BudgetExhausted.Spent (%v) should exceed Budget (%v)", exhausted.Spent, exhausted.Budget)
	}
	// Error string should be non-empty
	if exhausted.Error() == "" {
		t.Error("BudgetExhausted.Error() should be non-empty")
	}
}

func TestWrapUsage_BudgetExhausted_ReportTerminated(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(0.001)
	sess := b.NewSession()
	defer sess.Close()

	_ = sess.WrapUsage("gpt-4o", 10000, 5000)

	r := sess.Report()
	if r.TerminatedBy != "budget_exhausted" {
		t.Errorf("TerminatedBy = %q, want %q", r.TerminatedBy, "budget_exhausted")
	}
}

// ---------------------------------------------------------------------------
// Soft limit callback fires once at threshold
// ---------------------------------------------------------------------------

func TestSoftLimitCallback_FiresOnce(t *testing.T) {
	t.Parallel()

	fireCount := 0
	b, _ := agentbudget.New(1.0,
		agentbudget.WithSoftLimit(0.5),
		agentbudget.WithOnSoftLimit(func(r agentbudget.Report) {
			fireCount++
		}),
	)
	sess := b.NewSession()
	defer sess.Close()

	// gpt-4o: each call of 100k/50k tokens costs:
	// 100000*2.50/1e6 + 50000*10.00/1e6 = 0.25 + 0.50 = 0.75
	// First call at 0.75 already passes 50% of $1.00 budget.
	_ = sess.WrapUsage("gpt-4o", 100000, 50000) // pushes past 50%

	if fireCount != 1 {
		t.Errorf("soft limit callback fired %d times, want 1", fireCount)
	}

	// A second call (different tokens) should NOT re-fire the callback.
	// The remaining budget is $0.25, so 1000/500 tokens won't exceed budget.
	_ = sess.WrapUsage("gpt-4o", 1000, 500)

	if fireCount != 1 {
		t.Errorf("soft limit callback fired %d times after second call, want 1", fireCount)
	}
}

// ---------------------------------------------------------------------------
// Loop detection
// ---------------------------------------------------------------------------

func TestLoopDetected_SameModelRepeated(t *testing.T) {
	t.Parallel()

	// loopMaxCalls=3 within a very large window so timing doesn't matter.
	b, _ := agentbudget.New(100.0,
		agentbudget.WithLoopDetection(3, 3600),
	)
	sess := b.NewSession()
	defer sess.Close()

	var loopErr *agentbudget.LoopDetected

	for i := 0; i < 4; i++ {
		err := sess.WrapUsage("gpt-4o-mini", 100, 50)
		if err != nil {
			if errors.As(err, &loopErr) {
				// detected on the 3rd call (index 2) or 4th — whichever hits the cap
				if loopErr.Key != "gpt-4o-mini" {
					t.Errorf("LoopDetected.Key = %q, want %q", loopErr.Key, "gpt-4o-mini")
				}
				return // success
			}
			t.Fatalf("unexpected error type: %v", err)
		}
	}
	t.Fatal("expected LoopDetected error, got none after 4 identical calls")
}

func TestLoopDetected_CallbackFires(t *testing.T) {
	t.Parallel()

	loopFired := false
	b, _ := agentbudget.New(100.0,
		agentbudget.WithLoopDetection(3, 3600),
		agentbudget.WithOnLoopDetected(func(r agentbudget.Report) {
			loopFired = true
		}),
	)
	sess := b.NewSession()
	defer sess.Close()

	for i := 0; i < 5; i++ {
		_ = sess.WrapUsage("gpt-4o-mini", 100, 50)
	}

	if !loopFired {
		t.Error("expected onLoopDetected callback to fire, it did not")
	}
}

func TestLoopDetected_ReportTerminated(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(100.0,
		agentbudget.WithLoopDetection(3, 3600),
	)
	sess := b.NewSession()
	defer sess.Close()

	for i := 0; i < 5; i++ {
		_ = sess.WrapUsage("gpt-4o-mini", 100, 50)
	}

	r := sess.Report()
	if r.TerminatedBy != "loop_detected" {
		t.Errorf("TerminatedBy = %q, want %q", r.TerminatedBy, "loop_detected")
	}
}

// ---------------------------------------------------------------------------
// Track() — fixed tool costs
// ---------------------------------------------------------------------------

func TestTrack_RecordsCost(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(10.0)
	sess := b.NewSession()
	defer sess.Close()

	if err := sess.Track(0.05, "serp_api"); err != nil {
		t.Fatalf("Track error: %v", err)
	}
	if err := sess.Track(0.10, "browser_tool"); err != nil {
		t.Fatalf("Track error: %v", err)
	}

	const wantSpent = 0.15
	if diff := sess.Spent() - wantSpent; diff < -1e-9 || diff > 1e-9 {
		t.Errorf("Spent() = %.9f, want %.9f", sess.Spent(), wantSpent)
	}
}

func TestTrack_BudgetExhausted(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(0.04)
	sess := b.NewSession()
	defer sess.Close()

	err := sess.Track(0.05, "expensive_tool")
	if err == nil {
		t.Fatal("expected BudgetExhausted, got nil")
	}
	var exhausted *agentbudget.BudgetExhausted
	if !errors.As(err, &exhausted) {
		t.Fatalf("error type = %T, want *BudgetExhausted", err)
	}
}

func TestTrack_WithMetadata(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(10.0)
	sess := b.NewSession()
	defer sess.Close()

	err := sess.Track(0.01, "my_tool", map[string]any{"key": "value"})
	if err != nil {
		t.Fatalf("Track with metadata error: %v", err)
	}
	if sess.Spent() == 0 {
		t.Error("Spent() should be > 0 after Track with metadata")
	}
}

func TestTrack_LoopDetectedOnRepeatedTool(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(100.0, agentbudget.WithLoopDetection(3, 3600))
	sess := b.NewSession()
	defer sess.Close()

	var loopErr *agentbudget.LoopDetected
	for i := 0; i < 5; i++ {
		err := sess.Track(0.01, "repeat_tool")
		if err != nil && errors.As(err, &loopErr) {
			if loopErr.Key != "repeat_tool" {
				t.Errorf("LoopDetected.Key = %q, want %q", loopErr.Key, "repeat_tool")
			}
			return
		}
	}
	t.Fatal("expected LoopDetected after repeated Track calls")
}

// ---------------------------------------------------------------------------
// WouldExceed() — pre-flight check
// ---------------------------------------------------------------------------

func TestWouldExceed(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(1.0)
	sess := b.NewSession()
	defer sess.Close()

	if sess.WouldExceed(0.50) {
		t.Error("WouldExceed(0.50) = true on fresh $1.00 budget, want false")
	}
	if !sess.WouldExceed(1.01) {
		t.Error("WouldExceed(1.01) = false on $1.00 budget, want true")
	}

	// Spend some, then check again
	_ = sess.Track(0.80, "tool")
	if !sess.WouldExceed(0.25) {
		t.Error("WouldExceed(0.25) = false after spending $0.80 of $1.00, want true")
	}
	if sess.WouldExceed(0.10) {
		t.Error("WouldExceed(0.10) = true when $0.20 remains, want false")
	}
}

func TestWouldExceed_DoesNotRecord(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(1.0)
	sess := b.NewSession()
	defer sess.Close()

	sess.WouldExceed(0.50)
	if sess.Spent() != 0 {
		t.Errorf("WouldExceed must not record anything; Spent() = %v", sess.Spent())
	}
}

// ---------------------------------------------------------------------------
// FinalizationReserve — reduces effective limit
// ---------------------------------------------------------------------------

func TestFinalizationReserve(t *testing.T) {
	t.Parallel()

	// 5% reserve on $1.00 → effective budget = $0.95
	b, err := agentbudget.New(1.0, agentbudget.WithFinalizationReserve(0.05))
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// MaxSpend should still report the full $1.00
	if b.MaxSpend() != 1.0 {
		t.Errorf("MaxSpend() = %v, want 1.0", b.MaxSpend())
	}

	sess := b.NewSession()
	defer sess.Close()

	// $0.90 should succeed (under $0.95 effective limit)
	if err := sess.Track(0.90, "work"); err != nil {
		t.Fatalf("Track(0.90) unexpected error: %v", err)
	}

	// $0.10 more would bring total to $1.00, but effective limit is $0.95 → should fail
	err = sess.Track(0.10, "extra_work")
	if err == nil {
		t.Fatal("expected BudgetExhausted when exceeding effective (reserved) limit, got nil")
	}
	var exhausted *agentbudget.BudgetExhausted
	if !errors.As(err, &exhausted) {
		t.Fatalf("error type = %T, want *BudgetExhausted", err)
	}
}

func TestFinalizationReserve_InvalidFraction(t *testing.T) {
	t.Parallel()

	cases := []float64{-0.01, 1.0, 1.5}
	for _, frac := range cases {
		frac := frac
		t.Run("", func(t *testing.T) {
			t.Parallel()
			_, err := agentbudget.New(1.0, agentbudget.WithFinalizationReserve(frac))
			if err == nil {
				t.Errorf("New with finReserve=%.2f expected error, got nil", frac)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ChildSession — caps to min(requested, parent.remaining)
// ---------------------------------------------------------------------------

func TestChildSession_CapsToParentRemaining(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(1.0)
	parent := b.NewSession()
	defer parent.Close()

	// Spend $0.70 from parent, leaving $0.30
	_ = parent.Track(0.70, "parent_work")

	// Request $2.00 child budget but only $0.30 remains
	child := parent.ChildSession(2.0)
	defer child.Close()

	r := child.Report()
	// Use epsilon comparison: IEEE 754 subtraction (1.0 - 0.70) may yield 0.30000000000000004
	const wantChildBudget = 0.30
	if diff := r.Budget - wantChildBudget; diff < -1e-9 || diff > 1e-9 {
		t.Errorf("child Budget = %.15f, want %.2f (min of 2.0 and 0.30)", r.Budget, wantChildBudget)
	}
}

func TestChildSession_RequestedLessThanRemaining(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(10.0)
	parent := b.NewSession()
	defer parent.Close()

	// Request $2.00 child budget; parent has $10.00 remaining
	child := parent.ChildSession(2.0)
	defer child.Close()

	r := child.Report()
	if r.Budget != 2.0 {
		t.Errorf("child Budget = %v, want 2.0", r.Budget)
	}
}

func TestChildSession_BudgetEnforcedSeparately(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(10.0)
	parent := b.NewSession()
	defer parent.Close()

	child := parent.ChildSession(0.001) // tiny budget
	defer child.Close()

	err := child.WrapUsage("gpt-4o", 10000, 5000)
	if err == nil {
		t.Fatal("expected BudgetExhausted on child, got nil")
	}
	var exhausted *agentbudget.BudgetExhausted
	if !errors.As(err, &exhausted) {
		t.Fatalf("error type = %T, want *BudgetExhausted", err)
	}

	// Parent should be unaffected (child failure doesn't roll up automatically)
	if parent.Spent() != 0 {
		t.Errorf("parent.Spent() = %v after child exhausted, want 0", parent.Spent())
	}
}

func TestChildSession_CustomID(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(10.0)
	parent := b.NewSession()
	defer parent.Close()

	child := parent.ChildSession(1.0, agentbudget.WithSessionID("my-child"))
	defer child.Close()

	if child.ID() != "my-child" {
		t.Errorf("child ID = %q, want %q", child.ID(), "my-child")
	}
}

// ---------------------------------------------------------------------------
// Report() — field correctness after spend
// ---------------------------------------------------------------------------

func TestReport_FieldsAfterSpend(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(5.0)
	sess := b.NewSession(agentbudget.WithSessionID("report-test"))
	defer sess.Close()

	_ = sess.Track(0.10, "tool_a")
	_ = sess.WrapUsage("gpt-4o-mini", 2000, 1000)

	r := sess.Report()

	if r.SessionID != "report-test" {
		t.Errorf("SessionID = %q, want %q", r.SessionID, "report-test")
	}
	if r.Budget != 5.0 {
		t.Errorf("Budget = %v, want 5.0", r.Budget)
	}
	if r.TotalSpent <= 0 {
		t.Errorf("TotalSpent = %v, want > 0", r.TotalSpent)
	}
	expectedRemaining := r.Budget - r.TotalSpent
	if diff := r.Remaining - expectedRemaining; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("Remaining = %v, want %.6f", r.Remaining, expectedRemaining)
	}
	if r.EventCount != 2 {
		t.Errorf("EventCount = %d, want 2", r.EventCount)
	}
	if r.DurationSecs == nil {
		t.Error("DurationSecs should not be nil")
	}
	if r.TerminatedBy != "" {
		t.Errorf("TerminatedBy = %q, want empty on healthy session", r.TerminatedBy)
	}
	if r.Breakdown == nil {
		t.Error("Breakdown should not be nil")
	}
}

func TestReport_BreakdownByModelAndTool(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(10.0)
	sess := b.NewSession()
	defer sess.Close()

	_ = sess.WrapUsage("gpt-4o", 1000, 500)
	_ = sess.Track(0.05, "my_tool")

	r := sess.Report()

	llm, ok := r.Breakdown["llm"].(map[string]any)
	if !ok {
		t.Fatal("Breakdown[\"llm\"] is missing or wrong type")
	}
	if llm["calls"].(int) != 1 {
		t.Errorf("llm calls = %v, want 1", llm["calls"])
	}

	tools, ok := r.Breakdown["tools"].(map[string]any)
	if !ok {
		t.Fatal("Breakdown[\"tools\"] is missing or wrong type")
	}
	if tools["calls"].(int) != 1 {
		t.Errorf("tool calls = %v, want 1", tools["calls"])
	}
}

// ---------------------------------------------------------------------------
// RegisterModel + CalculateCost — custom models
// ---------------------------------------------------------------------------

func TestRegisterModel_CustomPricing(t *testing.T) {
	t.Parallel()

	model := "custom-test-model-unit"
	if err := agentbudget.RegisterModel(model, 8.0, 24.0); err != nil {
		t.Fatalf("RegisterModel error: %v", err)
	}

	// 1000 input @ $8/M + 500 output @ $24/M = 0.008 + 0.012 = 0.020
	const wantCost = 1000*8.0/1e6 + 500*24.0/1e6

	cost, ok := agentbudget.CalculateCost(model, 1000, 500)
	if !ok {
		t.Fatal("CalculateCost returned ok=false for registered custom model")
	}
	if diff := cost - wantCost; diff < -1e-9 || diff > 1e-9 {
		t.Errorf("CalculateCost = %.9f, want %.9f", cost, wantCost)
	}
}

func TestRegisterModel_OverridesBuiltin(t *testing.T) {
	t.Parallel()

	// Override gpt-4o pricing temporarily (unique model string to avoid race)
	model := "override-gpt-test"
	if err := agentbudget.RegisterModel(model, 1.0, 2.0); err != nil {
		t.Fatalf("RegisterModel error: %v", err)
	}

	cost, ok := agentbudget.CalculateCost(model, 1_000_000, 0)
	if !ok {
		t.Fatal("CalculateCost returned ok=false")
	}
	// $1.00/M input, 1M tokens = $1.00
	if diff := cost - 1.0; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("CalculateCost = %.6f, want 1.0", cost)
	}
}

func TestRegisterModel_NegativePrice(t *testing.T) {
	t.Parallel()

	err := agentbudget.RegisterModel("bad-model", -1.0, 2.0)
	if err == nil {
		t.Error("RegisterModel with negative input price expected error, got nil")
	}
}

func TestCalculateCost_UnknownModel(t *testing.T) {
	t.Parallel()
	_, ok := agentbudget.CalculateCost("totally-unknown-xyz-model", 1000, 500)
	if ok {
		t.Error("CalculateCost for unknown model should return ok=false")
	}
}

func TestCalculateCost_TableDriven(t *testing.T) {
	t.Parallel()

	cases := []struct {
		model   string
		in      int64
		out     int64
		wantUSD float64
	}{
		// gpt-4o: $2.50/$10.00 per M
		{"gpt-4o", 1_000_000, 0, 2.50},
		{"gpt-4o", 0, 1_000_000, 10.00},
		{"gpt-4o", 1_000_000, 1_000_000, 12.50},
		// gpt-4o-mini: $0.15/$0.60 per M
		{"gpt-4o-mini", 1_000_000, 0, 0.15},
		{"gpt-4o-mini", 0, 1_000_000, 0.60},
		// claude-opus-4-6: $5.00/$25.00 per M
		{"claude-opus-4-6", 1_000_000, 0, 5.00},
		{"claude-opus-4-6", 0, 1_000_000, 25.00},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			cost, ok := agentbudget.CalculateCost(tc.model, tc.in, tc.out)
			if !ok {
				t.Fatalf("CalculateCost returned ok=false for known model %q", tc.model)
			}
			if diff := cost - tc.wantUSD; diff < -1e-6 || diff > 1e-6 {
				t.Errorf("CalculateCost(%q, %d, %d) = %.6f, want %.6f", tc.model, tc.in, tc.out, cost, tc.wantUSD)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Session Close() — marks session done
// ---------------------------------------------------------------------------

func TestSession_Close_SetsDuration(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(5.0)
	sess := b.NewSession()

	// Before close, duration may be set (computed as time.Now() - start)
	sess.Close()

	r := sess.Report()
	if r.DurationSecs == nil {
		t.Error("DurationSecs should not be nil after Close()")
	}
	if *r.DurationSecs < 0 {
		t.Errorf("DurationSecs = %v, want >= 0", *r.DurationSecs)
	}
}

func TestSession_Close_Idempotent(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(5.0)
	sess := b.NewSession()

	sess.Close()
	sess.Close() // must not panic

	r := sess.Report()
	if r.DurationSecs == nil {
		t.Error("DurationSecs should not be nil after double Close()")
	}
}

func TestSession_IDIsNonEmpty(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(5.0)
	sess := b.NewSession()
	defer sess.Close()

	if sess.ID() == "" {
		t.Error("Session.ID() should not be empty")
	}
}

func TestSession_WithSessionID(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New(5.0)
	sess := b.NewSession(agentbudget.WithSessionID("my-unique-id"))
	defer sess.Close()

	if sess.ID() != "my-unique-id" {
		t.Errorf("Session.ID() = %q, want %q", sess.ID(), "my-unique-id")
	}
}

// ---------------------------------------------------------------------------
// Remaining / Spent consistency
// ---------------------------------------------------------------------------

func TestRemainingAndSpentConsistency(t *testing.T) {
	t.Parallel()

	b, _ := agentbudget.New(10.0)
	sess := b.NewSession()
	defer sess.Close()

	_ = sess.Track(1.23, "tool1")
	_ = sess.Track(0.77, "tool2")

	spent := sess.Spent()
	remaining := sess.Remaining()

	if diff := (spent + remaining) - 10.0; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("spent(%v) + remaining(%v) = %v, want 10.0", spent, remaining, spent+remaining)
	}
}

// ---------------------------------------------------------------------------
// Hard limit callback fires on budget exhaustion
// ---------------------------------------------------------------------------

func TestHardLimitCallback(t *testing.T) {
	t.Parallel()

	hardFired := false
	b, _ := agentbudget.New(0.001,
		agentbudget.WithOnHardLimit(func(r agentbudget.Report) {
			hardFired = true
		}),
	)
	sess := b.NewSession()
	defer sess.Close()

	_ = sess.WrapUsage("gpt-4o", 10000, 5000)

	if !hardFired {
		t.Error("expected onHardLimit callback to fire, it did not")
	}
}

// ---------------------------------------------------------------------------
// MaxSpend on Budget
// ---------------------------------------------------------------------------

func TestBudget_MaxSpend(t *testing.T) {
	t.Parallel()
	b, _ := agentbudget.New("$7.50")
	if b.MaxSpend() != 7.50 {
		t.Errorf("MaxSpend() = %v, want 7.50", b.MaxSpend())
	}
}

func TestBudget_MaxSpend_WithReserve(t *testing.T) {
	t.Parallel()
	// Reserve reduces effective limit but MaxSpend() returns full amount
	b, _ := agentbudget.New(2.0, agentbudget.WithFinalizationReserve(0.10))
	if b.MaxSpend() != 2.0 {
		t.Errorf("MaxSpend() = %v, want 2.0 (full budget, not reduced)", b.MaxSpend())
	}
}

// ---------------------------------------------------------------------------
// Error string coverage
// ---------------------------------------------------------------------------

func TestErrorStrings(t *testing.T) {
	t.Parallel()

	exhausted := &agentbudget.BudgetExhausted{Budget: 1.0, Spent: 1.5}
	if exhausted.Error() == "" {
		t.Error("BudgetExhausted.Error() is empty")
	}

	loop := &agentbudget.LoopDetected{Key: "my-model"}
	if loop.Error() == "" {
		t.Error("LoopDetected.Error() is empty")
	}

	inv := &agentbudget.InvalidBudget{Value: "bad"}
	if inv.Error() == "" {
		t.Error("InvalidBudget.Error() is empty")
	}
}
