# AgentBudget TypeScript SDK

Real-time cost enforcement for AI agent sessions — TypeScript/JavaScript edition.

[![npm](https://img.shields.io/npm/v/agentbudget)](https://www.npmjs.com/package/agentbudget)
[![License](https://img.shields.io/github/license/AgentBudget/agentbudget)](https://github.com/AgentBudget/agentbudget/blob/main/LICENSE)

---

## Install

```bash
npm install agentbudget
```

Works with `openai` ≥ 4.0 and `@anthropic-ai/sdk` ≥ 0.20 (both optional peer dependencies).
Node.js 18+. Zero runtime dependencies.

---

## Quickstart

```ts
import { AgentBudget } from "agentbudget";
import OpenAI from "openai";

const budget = new AgentBudget("$5.00");
const session = budget.newSession();

const openai = new OpenAI();
const resp = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Summarize this market report..." }],
});

// Wrap the response — extracts model + tokens and records cost
session.wrapOpenAI(resp);

console.log(`spent: $${session.spent.toFixed(4)}  remaining: $${session.remaining.toFixed(4)}`);
console.log(session.report());

session.close();
```

---

## OpenAI Example

```ts
import { AgentBudget, BudgetExhausted, LoopDetected } from "agentbudget";
import OpenAI from "openai";

const budget = new AgentBudget("$5.00", {
  onSoftLimit: (r) => console.warn(`⚠️ 90% budget used — $${r.total_spent.toFixed(4)}`),
  onHardLimit: (r) => console.error(`🛑 hard limit hit: $${r.total_spent.toFixed(4)}`),
});

const openai = new OpenAI();
const session = budget.newSession();

try {
  const resp = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Research CRM competitors" }],
  });

  session.wrapOpenAI(resp);
  console.log(resp.choices[0]?.message.content);
} catch (err) {
  if (err instanceof BudgetExhausted) {
    console.error(`Hard limit: $${err.spent.toFixed(4)} spent of $${err.budget.toFixed(2)}`);
  } else if (err instanceof LoopDetected) {
    console.error(`Loop detected on: ${err.key}`);
  }
} finally {
  session.close();
}
```

---

## Anthropic Example

```ts
import Anthropic from "@anthropic-ai/sdk";
import { AgentBudget } from "agentbudget";

const budget = new AgentBudget("$5.00");
const session = budget.newSession();

const anthropic = new Anthropic();
const resp = await anthropic.messages.create({
  model: "claude-opus-4-6-20250514",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Analyze this dataset" }],
});

session.wrapAnthropic(resp);
console.log(`spent: $${session.spent.toFixed(4)}`);
session.close();
```

---

## Auto-patching a Client with `wrapClient`

Attach a session to a specific client instance — every call on that instance is tracked automatically:

```ts
import { AgentBudget, wrapClient } from "agentbudget";
import OpenAI from "openai";

const budget = new AgentBudget("$5.00");
const session = budget.newSession();

// Only this client instance is tracked
const client = wrapClient(new OpenAI(), session);
await client.chat.completions.create({ ... }); // tracked

const other = new OpenAI();
await other.chat.completions.create({ ... }); // NOT tracked
```

Works with OpenAI and Anthropic client instances.

---

## Tracking Tool Costs

```ts
const searchResult = await callSerpAPI(query);
// Record a fixed cost for the tool call
session.track(searchResult, 0.01, "serp_api");
// track() returns the result unchanged, so you can inline it:
const data = session.track(await fetchData(url), 0.05, "fetch");
```

---

## Circuit Breaker

```ts
const budget = new AgentBudget("$5.00", {
  softLimit: 0.8,             // warn at 80% (default 0.9)
  loopMaxCalls: 10,           // trips after 10 repeated calls
  loopWindowSeconds: 60,      // within a 60-second window
  onSoftLimit: (r) => alertOps(r),
  onHardLimit: (r) => logBudgetEvent(r),
  onLoopDetected: (r) => killAgent(r),
});
```

- **Soft limit** — fires callback once, agent can wrap up gracefully
- **Hard limit** — throws `BudgetExhausted`, no further spend allowed
- **Loop detection** — same model/tool called N times in window throws `LoopDetected`

---

## Finalization Reserve

```ts
const budget = new AgentBudget("$1.00", {
  finalizationReserve: 0.05, // hard limit fires at $0.95
});

const session = budget.newSession();

if (session.wouldExceed(estimatedFinalCost)) {
  return "Here's what I completed so far...";
}
// Safe to proceed
session.wrapOpenAI(await openai.chat.completions.create({ ... }));
```

---

## Child Sessions

```ts
const parent = budget.newSession();

const child = parent.childSession(2.0); // sub-budget: $2 or parent's remaining
child.wrapOpenAI(resp);
child.close();

// Roll up child spend to parent
parent.track(null, child.spent, "subtask");
```

---

## Custom Model Pricing

```ts
import { registerModel } from "agentbudget";

registerModel("my-model-v1", 5.00, 20.00); // per million tokens
```

---

## Cost Report

```ts
const report = session.report();
// {
//   session_id:       "sess_...",
//   budget:           5.0,
//   total_spent:      3.421,
//   remaining:        1.579,
//   breakdown:        { llm: { total, calls, by_model }, tools: { ... } },
//   duration_seconds: 34.2,
//   terminated_by:    null,  // null | "budget_exhausted" | "loop_detected"
//   event_count:      12,
// }
```

---

## Error Handling

```ts
import { BudgetExhausted, LoopDetected } from "agentbudget";

try {
  session.wrapOpenAI(resp);
} catch (err) {
  if (err instanceof BudgetExhausted) {
    // err.budget, err.spent
  } else if (err instanceof LoopDetected) {
    // err.key
  }
}
```

---

## API Reference

### `new AgentBudget(maxSpend, opts?)`

| Option | Default | Description |
|---|---|---|
| `softLimit` | `0.9` | Fraction of budget at which soft-limit callback fires |
| `loopMaxCalls` | `10` | Repeated calls per window before LoopDetected |
| `loopWindowSeconds` | `60` | Loop detection window |
| `finalizationReserve` | `0` | Fraction held back from hard limit |
| `onSoftLimit` | — | Called once when soft limit triggers |
| `onHardLimit` | — | Called when BudgetExhausted is thrown |
| `onLoopDetected` | — | Called when LoopDetected is thrown |

### `budget.newSession(opts?): BudgetSession`

| Option | Description |
|---|---|
| `id` | Custom session ID |

### `BudgetSession` methods

| Method | Description |
|---|---|
| `wrapOpenAI(response)` | Record cost from OpenAI response |
| `wrapAnthropic(response)` | Record cost from Anthropic response |
| `wrapUsage(model, inputTokens, outputTokens)` | Record cost from raw token counts |
| `track(result, cost, toolName?, metadata?)` | Record fixed-cost action, returns result |
| `wouldExceed(cost): boolean` | Pre-flight check without recording |
| `spent: number` | Total dollars spent |
| `remaining: number` | Dollars remaining |
| `childSession(maxSpend, opts?): BudgetSession` | Create a sub-budget session |
| `report(): Report` | Structured cost summary |
| `close()` | Mark session ended |

### `wrapClient(client, session): client`

Attach a session to an OpenAI or Anthropic client instance for automatic tracking.

---

## Publishing to npm

```bash
cd sdks/typescript
npm run build
npm publish
```

---

## License

Apache 2.0 — see [LICENSE](../../LICENSE)
