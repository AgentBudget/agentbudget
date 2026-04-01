# AgentBudget Go SDK

Real-time cost enforcement for AI agent sessions — Go edition.

[![Go Reference](https://pkg.go.dev/badge/github.com/AgentBudget/agentbudget/sdks/go.svg)](https://pkg.go.dev/github.com/AgentBudget/agentbudget/sdks/go)
[![License](https://img.shields.io/github/license/AgentBudget/agentbudget)](https://github.com/AgentBudget/agentbudget/blob/main/LICENSE)

---

## Install

```bash
go get github.com/AgentBudget/agentbudget/sdks/go
```

Go 1.21+. Zero external dependencies.

---

## Quickstart

```go
import agentbudget "github.com/AgentBudget/agentbudget/sdks/go"

budget, err := agentbudget.New("$5.00")
if err != nil {
    log.Fatal(err)
}

session := budget.NewSession()
defer session.Close()

// Call your LLM, then record token usage:
resp, _ := openaiClient.CreateChatCompletion(ctx, req)
if err := session.WrapUsage(
    resp.Model,
    int64(resp.Usage.PromptTokens),
    int64(resp.Usage.CompletionTokens),
); err != nil {
    // *agentbudget.BudgetExhausted → hard limit hit
    // *agentbudget.LoopDetected    → repeated calls detected
    log.Fatal(err)
}

fmt.Printf("spent: $%.4f  remaining: $%.4f\n", session.Spent(), session.Remaining())
fmt.Printf("%+v\n", session.Report())
```

---

## OpenAI Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    openai "github.com/sashabaranov/go-openai"
    agentbudget "github.com/AgentBudget/agentbudget/sdks/go"
)

func main() {
    budget, _ := agentbudget.New("$5.00",
        agentbudget.WithOnSoftLimit(func(r agentbudget.Report) {
            fmt.Printf("⚠️  90%% budget used — $%.4f spent\n", r.TotalSpent)
        }),
        agentbudget.WithOnHardLimit(func(r agentbudget.Report) {
            fmt.Printf("🛑 hard limit hit: $%.4f\n", r.TotalSpent)
        }),
    )

    client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
    session := budget.NewSession()
    defer session.Close()

    resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
        Model: openai.GPT4o,
        Messages: []openai.ChatCompletionMessage{
            {Role: openai.ChatMessageRoleUser, Content: "Summarize this market report..."},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    if err := session.WrapUsage(
        resp.Model,
        int64(resp.Usage.PromptTokens),
        int64(resp.Usage.CompletionTokens),
    ); err != nil {
        log.Fatalf("budget error: %v", err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
    fmt.Printf("\nspent: $%.4f  remaining: $%.4f\n", session.Spent(), session.Remaining())
}
```

---

## Anthropic Example

```go
import (
    anthropic "github.com/anthropics/anthropic-sdk-go"
    agentbudget "github.com/AgentBudget/agentbudget/sdks/go"
)

session := budget.NewSession()
defer session.Close()

resp, err := anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{
    Model:     anthropic.F(anthropic.ModelClaude3Opus20240229),
    MaxTokens: anthropic.F(int64(1024)),
    Messages: anthropic.F([]anthropic.MessageParam{
        anthropic.NewUserMessage(anthropic.NewTextBlock("Research competitors...")),
    }),
})
if err != nil {
    log.Fatal(err)
}

if err := session.WrapUsage(
    string(resp.Model),
    resp.Usage.InputTokens,
    resp.Usage.OutputTokens,
); err != nil {
    log.Fatalf("budget error: %v", err)
}
```

---

## Tracking Tool Costs

```go
// Track any fixed-cost action: API calls, database queries, web scraping, etc.
result, err := callSerpAPI(query)
if trackErr := session.Track(0.01, "serp_api"); trackErr != nil {
    log.Fatal(trackErr)
}
```

---

## Circuit Breaker

```go
budget, _ := agentbudget.New("$5.00",
    agentbudget.WithSoftLimit(0.8),                    // warn at 80%
    agentbudget.WithLoopDetection(10, 60.0),           // 10 calls in 60s
    agentbudget.WithOnSoftLimit(func(r agentbudget.Report) {
        alertOpsTeam(r)
    }),
    agentbudget.WithOnHardLimit(func(r agentbudget.Report) {
        log.Printf("agent %s exceeded budget", r.SessionID)
    }),
    agentbudget.WithOnLoopDetected(func(r agentbudget.Report) {
        log.Printf("loop detected in session %s", r.SessionID)
    }),
)
```

Three protection levels:
- **Soft limit** — fires callback once when spending exceeds threshold. Agent can finish gracefully.
- **Hard limit** — `WrapUsage`/`Track` returns `*BudgetExhausted`. No further spend allowed.
- **Loop detection** — same model or tool called N times in a window returns `*LoopDetected`.

---

## Finalization Reserve

Prevent agents from being cut off mid-response. Reserve a fraction for the final step:

```go
budget, _ := agentbudget.New("$1.00",
    agentbudget.WithFinalizationReserve(0.05), // hard limit fires at $0.95
)

session := budget.NewSession()

// Check before the final call:
if session.WouldExceed(estimatedFinalCost) {
    return "Here's what I completed so far..."
}
// Safe — won't hit the hard limit
session.WrapUsage("gpt-4o", inputTokens, outputTokens)
```

---

## Child Sessions

Allocate sub-budgets to parallel tasks:

```go
parent := budget.NewSession()
defer parent.Close()

child := parent.ChildSession(2.0) // capped at $2 or parent's remaining, whichever is lower
defer func() {
    child.Close()
    // Roll up child spend to parent
    _ = parent.Track(child.Spent(), "child:"+child.ID())
}()

_ = child.WrapUsage("gpt-4o-mini", inputTokens, outputTokens)
```

---

## Custom Model Pricing

```go
agentbudget.RegisterModel("my-model-v1", 5.00, 20.00) // per million tokens
```

---

## Cost Report

```go
report := session.Report()
// Report{
//     SessionID:    "sess_...",
//     Budget:       5.0,
//     TotalSpent:   3.421,
//     Remaining:    1.579,
//     Breakdown:    map[llm:[...] tools:[...]],
//     DurationSecs: 34.2,
//     TerminatedBy: "",     // "" | "budget_exhausted" | "loop_detected"
//     EventCount:   12,
// }
```

---

## Error Handling

```go
err := session.WrapUsage("gpt-4o", inputTokens, outputTokens)

var exhausted *agentbudget.BudgetExhausted
var loop *agentbudget.LoopDetected

switch {
case errors.As(err, &exhausted):
    fmt.Printf("hard limit: $%.4f spent of $%.2f budget\n", exhausted.Spent, exhausted.Budget)
case errors.As(err, &loop):
    fmt.Printf("loop on key: %s\n", loop.Key)
}
```

---

## API Reference

### `agentbudget.New(maxSpend, opts...) (*Budget, error)`

| Option | Default | Description |
|---|---|---|
| `WithSoftLimit(fraction)` | `0.9` | Fraction of budget at which soft-limit callback fires |
| `WithLoopDetection(n, secs)` | `10, 60` | Calls per window before LoopDetected |
| `WithFinalizationReserve(fraction)` | `0.0` | Fraction held back from hard limit |
| `WithOnSoftLimit(fn)` | nil | Called once when soft limit triggers |
| `WithOnHardLimit(fn)` | nil | Called when BudgetExhausted is returned |
| `WithOnLoopDetected(fn)` | nil | Called when LoopDetected is returned |

### `(*Budget).NewSession(opts...) *Session`

| Option | Description |
|---|---|
| `WithSessionID(id)` | Set a custom session ID |

### `(*Session)` methods

| Method | Description |
|---|---|
| `WrapUsage(model, inputTokens, outputTokens)` | Record LLM call cost |
| `Track(cost, toolName, metadata...)` | Record fixed-cost action |
| `WouldExceed(cost) bool` | Pre-flight check without recording |
| `Spent() float64` | Total dollars spent |
| `Remaining() float64` | Dollars remaining |
| `ChildSession(maxSpend) *Session` | Create a sub-budget session |
| `Report() Report` | Structured cost summary |
| `Close()` | Mark session ended |

---

## License

Apache 2.0 — see [LICENSE](../../LICENSE)
