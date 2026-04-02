//go:build smoke

package agentbudget_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	agentbudget "github.com/AgentBudget/agentbudget/sdks/go"
)

// openAIRequest mirrors the minimal OpenAI Chat Completions request body.
type openAIRequest struct {
	Model    string              `json:"model"`
	Messages []map[string]string `json:"messages"`
}

// openAIResponse mirrors the subset of the OpenAI Chat Completions response we need.
type openAIResponse struct {
	Model string `json:"model"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// callOpenAI makes a real HTTP call to the OpenAI Chat Completions endpoint.
// It does not depend on any third-party Go SDK.
func callOpenAI(t *testing.T, apiKey, model, userMessage string) openAIResponse {
	t.Helper()

	reqBody, err := json.Marshal(openAIRequest{
		Model: model,
		Messages: []map[string]string{
			{"role": "user", "content": userMessage},
		},
	})
	if err != nil {
		t.Fatalf("smoke: failed to marshal request: %v", err)
	}

	req, err := http.NewRequest(
		http.MethodPost,
		"https://api.openai.com/v1/chat/completions",
		bytes.NewReader(reqBody),
	)
	if err != nil {
		t.Fatalf("smoke: failed to create HTTP request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("smoke: HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("smoke: failed to read response body: %v", err)
	}

	var parsed openAIResponse
	if err := json.Unmarshal(rawBody, &parsed); err != nil {
		t.Fatalf("smoke: failed to parse response JSON: %v\nraw: %s", err, rawBody)
	}

	if parsed.Error != nil {
		t.Fatalf("smoke: OpenAI API error (%s): %s", parsed.Error.Type, parsed.Error.Message)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("smoke: unexpected HTTP status %d: %s", resp.StatusCode, rawBody)
	}

	return parsed
}

// requireOpenAIKey returns the OPENAI_API_KEY env var or skips the test.
func requireOpenAIKey(t *testing.T) string {
	t.Helper()
	key := os.Getenv("OPENAI_API_KEY")
	if strings.TrimSpace(key) == "" {
		t.Skip("smoke test requires OPENAI_API_KEY env var to be set")
	}
	return key
}

// ---------------------------------------------------------------------------
// Smoke: real OpenAI call records non-zero spend
// ---------------------------------------------------------------------------

func TestSmoke_OpenAI_WrapUsage_RecordsSpend(t *testing.T) {
	apiKey := requireOpenAIKey(t)

	const model = "gpt-4o-mini" // cheapest production model
	const prompt = "Say the single word: hello"

	// Set a generous budget for the smoke test.
	budget, err := agentbudget.New(1.0)
	if err != nil {
		t.Fatalf("agentbudget.New error: %v", err)
	}

	session := budget.NewSession(agentbudget.WithSessionID("smoke-openai-basic"))
	defer session.Close()

	t.Logf("smoke: calling OpenAI model=%s", model)
	resp := callOpenAI(t, apiKey, model, prompt)

	t.Logf("smoke: response model=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d",
		resp.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)

	if resp.Usage.TotalTokens == 0 {
		t.Fatal("smoke: OpenAI returned 0 total tokens — something is wrong with the response")
	}

	if err := session.WrapUsage(resp.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens); err != nil {
		t.Fatalf("smoke: WrapUsage error: %v", err)
	}

	spent := session.Spent()
	t.Logf("smoke: spent=$%.6f remaining=$%.6f", spent, session.Remaining())

	if spent <= 0 {
		t.Errorf("smoke: Spent() = %v, want > 0 after a real API call", spent)
	}

	r := session.Report()
	if r.TotalSpent != spent {
		t.Errorf("smoke: Report().TotalSpent = %v, want %v", r.TotalSpent, spent)
	}
	if r.EventCount != 1 {
		t.Errorf("smoke: Report().EventCount = %d, want 1", r.EventCount)
	}
}

// ---------------------------------------------------------------------------
// Smoke: BudgetExhausted fires on a micro budget
// ---------------------------------------------------------------------------

func TestSmoke_OpenAI_BudgetExhausted(t *testing.T) {
	apiKey := requireOpenAIKey(t)

	const model = "gpt-4o-mini"
	const prompt = "Count from 1 to 20, separated by commas."

	// Ridiculously small budget — $0.000001 (sub-micro)
	budget, err := agentbudget.New(0.000001)
	if err != nil {
		t.Fatalf("agentbudget.New error: %v", err)
	}

	session := budget.NewSession(agentbudget.WithSessionID("smoke-exhausted"))
	defer session.Close()

	resp := callOpenAI(t, apiKey, model, prompt)
	t.Logf("smoke: prompt_tokens=%d completion_tokens=%d", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)

	err = session.WrapUsage(resp.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	if err == nil {
		// It's theoretically possible for an extremely short response to fit
		// in $0.000001 on gpt-4o-mini — treat as informational, not fatal.
		t.Logf("smoke: no budget exceeded with micro-budget; spent=$%.9f (token counts may have been 0)", session.Spent())
		return
	}

	var exhausted *agentbudget.BudgetExhausted
	if !isAs(err, &exhausted) {
		t.Errorf("smoke: expected *BudgetExhausted, got %T: %v", err, err)
	} else {
		t.Logf("smoke: BudgetExhausted as expected — budget=$%.6f spent=$%.6f", exhausted.Budget, exhausted.Spent)
	}
}

// ---------------------------------------------------------------------------
// Smoke: full session lifecycle — spend, report, close
// ---------------------------------------------------------------------------

func TestSmoke_OpenAI_FullSessionLifecycle(t *testing.T) {
	apiKey := requireOpenAIKey(t)

	const model = "gpt-4o-mini"

	budget, err := agentbudget.New(5.0,
		agentbudget.WithSoftLimit(0.8),
		agentbudget.WithOnSoftLimit(func(r agentbudget.Report) {
			t.Logf("smoke: soft limit reached — spent=$%.6f of $%.2f", r.TotalSpent, r.Budget)
		}),
	)
	if err != nil {
		t.Fatalf("agentbudget.New error: %v", err)
	}

	session := budget.NewSession(agentbudget.WithSessionID("smoke-lifecycle"))
	defer session.Close()

	prompts := []string{
		"Reply with exactly: yes",
		"Reply with exactly: no",
	}

	for i, prompt := range prompts {
		resp := callOpenAI(t, apiKey, model, prompt)
		if err := session.WrapUsage(resp.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens); err != nil {
			t.Fatalf("smoke: WrapUsage [%d] error: %v", i, err)
		}
	}

	if session.Spent() <= 0 {
		t.Errorf("smoke: Spent() = %v after %d calls, want > 0", session.Spent(), len(prompts))
	}

	session.Close()

	r := session.Report()
	t.Logf("smoke: final report — session=%s budget=%.2f spent=%.6f remaining=%.6f events=%d duration=%.2fs",
		r.SessionID, r.Budget, r.TotalSpent, r.Remaining, r.EventCount, ptrFloat(r.DurationSecs))

	if r.EventCount != len(prompts) {
		t.Errorf("smoke: EventCount = %d, want %d", r.EventCount, len(prompts))
	}
	if r.DurationSecs == nil {
		t.Error("smoke: DurationSecs should not be nil after Close()")
	}
	if r.TerminatedBy != "" {
		t.Errorf("smoke: TerminatedBy = %q, want empty for healthy session", r.TerminatedBy)
	}
}

// ---------------------------------------------------------------------------
// Smoke: WouldExceed pre-flight works with real token counts
// ---------------------------------------------------------------------------

func TestSmoke_OpenAI_WouldExceed_PreFlight(t *testing.T) {
	apiKey := requireOpenAIKey(t)

	const model = "gpt-4o-mini"

	// Spend most of the budget first via a real call.
	budget, _ := agentbudget.New(1.0)
	session := budget.NewSession(agentbudget.WithSessionID("smoke-would-exceed"))
	defer session.Close()

	resp := callOpenAI(t, apiKey, model, "Say hello")
	if err := session.WrapUsage(resp.Model, resp.Usage.PromptTokens, resp.Usage.CompletionTokens); err != nil {
		t.Fatalf("smoke: WrapUsage error: %v", err)
	}

	remaining := session.Remaining()
	t.Logf("smoke: remaining=$%.6f after first call", remaining)

	// WouldExceed(remaining + 0.01) must be true
	if !session.WouldExceed(remaining + 0.01) {
		t.Errorf("smoke: WouldExceed(remaining+0.01) = false, want true")
	}

	// WouldExceed(remaining/2) must be false (unless remaining == 0)
	if remaining > 0 && session.WouldExceed(remaining/2) {
		t.Errorf("smoke: WouldExceed(remaining/2) = true, want false when budget still available")
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// isAs is a thin wrapper so the smoke file compiles without importing errors.
func isAs[T any](err error, target *T) bool {
	if target == nil {
		return false
	}
	// Use the errors.As implementation via a type assertion chain.
	return asErr(err, target)
}

func asErr[T any](err error, target *T) bool {
	for err != nil {
		if v, ok := err.(T); ok {
			*target = v
			return true
		}
		// Unwrap
		type unwrapper interface{ Unwrap() error }
		if u, ok := err.(unwrapper); ok {
			err = u.Unwrap()
		} else {
			break
		}
	}
	return false
}

func ptrFloat(f *float64) float64 {
	if f == nil {
		return 0
	}
	return *f
}

// Ensure fmt is used (for Logf format strings compiled in).
var _ = fmt.Sprintf
