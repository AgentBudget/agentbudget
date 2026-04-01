# feat: Go SDK — real-time cost enforcement for AI agents

## What this PR does

Adds `/sdks/go/` — a first-party Go SDK that replicates the full AgentBudget core in idiomatic Go.

**Install (after merge + tag):**
```bash
go get github.com/AgentBudget/agentbudget/sdks/go
```

Go 1.21+. Zero external dependencies.

---

## Files added

| File | Description |
|---|---|
| `go.mod` | Module `github.com/AgentBudget/agentbudget/sdks/go` |
| `agentbudget.go` | Package-level doc with usage examples |
| `errors.go` | `BudgetExhausted`, `LoopDetected`, `InvalidBudget` error types |
| `pricing.go` | 35+ built-in model prices, `RegisterModel()`, fuzzy date + OpenRouter matching |
| `ledger.go` | Thread-safe (`sync.Mutex`) cost accumulator with `Record()`, `WouldExceed()`, `Breakdown()` |
| `circuit_breaker.go` | Soft-limit (fires once) + windowed loop detection |
| `session.go` | `WrapUsage`, `Track`, `ChildSession`, `WouldExceed`, `Report`, `Close` |
| `budget.go` | `New()` factory with functional options (`WithSoftLimit`, `WithLoopDetection`, `WithFinalizationReserve`, etc.) |
| `example_test.go` | Runnable Go examples for all major flows |
| `README.md` | Full install + usage docs with OpenAI and Anthropic examples |

---

## Usage

```go
import agentbudget "github.com/AgentBudget/agentbudget/sdks/go"

budget, err := agentbudget.New("$5.00",
    agentbudget.WithSoftLimit(0.9),
    agentbudget.WithOnHardLimit(func(r agentbudget.Report) {
        log.Printf("hard limit hit: $%.4f", r.TotalSpent)
    }),
)

session := budget.NewSession()
defer session.Close()

// After your OpenAI call:
resp, _ := openaiClient.CreateChatCompletion(ctx, req)
if err := session.WrapUsage(
    resp.Model,
    int64(resp.Usage.PromptTokens),
    int64(resp.Usage.CompletionTokens),
); err != nil {
    var exhausted *agentbudget.BudgetExhausted
    if errors.As(err, &exhausted) {
        log.Fatalf("over budget: $%.4f spent", exhausted.Spent)
    }
}

fmt.Printf("spent: $%.4f  remaining: $%.4f\n", session.Spent(), session.Remaining())
fmt.Printf("%+v\n", session.Report())
```

```go
// After your Anthropic call:
resp, _ := anthropicClient.Messages.New(ctx, anthropic.MessageNewParams{...})
session.WrapUsage(string(resp.Model), resp.Usage.InputTokens, resp.Usage.OutputTokens)

// Track a tool/API call with a known cost:
session.Track(0.01, "serp_api")

// Pre-flight check before an expensive final call:
if session.WouldExceed(estimatedCost) {
    return "Budget nearly exhausted — wrapping up"
}
```

---

## Feature coverage

| Feature | Implemented |
|---|---|
| Budget session creation | ✅ |
| Hard budget limit (`BudgetExhausted`) | ✅ |
| Soft limit callback (fires once at configurable %) | ✅ |
| Loop detection (windowed, per model/tool key) | ✅ |
| OpenAI support | ✅ |
| Anthropic support | ✅ |
| 35+ built-in model prices | ✅ |
| Custom model pricing (`RegisterModel`) | ✅ |
| Fuzzy model matching (date suffixes + OpenRouter prefix) | ✅ |
| Tool/API cost tracking (`Track`) | ✅ |
| Child sessions with sub-budgets | ✅ |
| Finalization reserve (`WithFinalizationReserve`) | ✅ |
| `WouldExceed` pre-flight check | ✅ |
| Thread safety (`sync.Mutex` on all shared state) | ✅ |
| Functional options pattern (`With...` opts) | ✅ |
| Cost report struct | ✅ |

---

## Commits

```
3e9c0fa  feat(go): add Go SDK core — errors, pricing, ledger, circuit breaker
4fc00d9  feat(go): add Budget factory and Session — WrapUsage, Track, ChildSession
b16c44a  feat(go): add examples and README for Go SDK
```

---

## Test plan

- [x] `go build ./...` — passes locally
- [x] `go vet ./...` — passes locally
- [ ] `go test ./...` — example tests compile; add unit tests in follow-up
- [ ] Verify module path resolves after tagging `sdks/go/v0.3.0`

---

## After merge

Tag the release so `go get` resolves cleanly:

```bash
git tag sdks/go/v0.3.0
git push origin sdks/go/v0.3.0
```

> Note: LangChain and CrewAI integrations for Go are planned for a future release.
