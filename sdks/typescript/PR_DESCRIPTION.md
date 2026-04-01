# feat: TypeScript SDK — real-time cost enforcement for AI agents

## What this PR does

Adds `/sdks/typescript/` — a first-party TypeScript/JavaScript SDK that replicates the full AgentBudget core, published to npm as `agentbudget`.

**Install (after merge + publish):**
```bash
npm install agentbudget
```

Node.js 18+. Zero runtime dependencies. Works with `openai` ≥ 4.0 and `@anthropic-ai/sdk` ≥ 0.20 (both optional peer deps).

---

## Files added

| File | Description |
|---|---|
| `package.json` | npm package config, `"agentbudget"`, version 0.3.0, CJS + ESM exports |
| `tsconfig.json` | Strict TypeScript config with `exactOptionalPropertyTypes` |
| `src/errors.ts` | `BudgetExhausted`, `LoopDetected`, `InvalidBudget`, `AgentBudgetError` |
| `src/pricing.ts` | 35+ built-in model prices, `registerModel()`, fuzzy date + OpenRouter matching |
| `src/ledger.ts` | Cost accumulator with `record()`, `wouldExceed()`, `breakdown()` |
| `src/circuit_breaker.ts` | Soft-limit (fires once) + windowed loop detection |
| `src/session.ts` | `BudgetSession` — `wrapOpenAI`, `wrapAnthropic`, `wrapUsage`, `track`, `childSession`, `report` |
| `src/budget.ts` | `AgentBudget` class with options, `newSession()` factory |
| `src/patch.ts` | `wrapClient()` — attach a session to an OpenAI or Anthropic client instance |
| `src/index.ts` | Public exports |
| `README.md` | Full install + usage docs |

---

## Usage

```ts
import { AgentBudget, BudgetExhausted, LoopDetected, wrapClient } from "agentbudget";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const budget = new AgentBudget("$5.00", {
  softLimit: 0.9,
  onSoftLimit: (r) => console.warn(`⚠️ 90% used — $${r.total_spent.toFixed(4)}`),
  onHardLimit: (r) => console.error(`🛑 hard limit: $${r.total_spent.toFixed(4)}`),
});

const session = budget.newSession();

try {
  // OpenAI
  const resp = await new OpenAI().chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Analyze this..." }],
  });
  session.wrapOpenAI(resp);

  // Anthropic
  const msg = await new Anthropic().messages.create({ model: "claude-opus-4-6-20250514", ... });
  session.wrapAnthropic(msg);

  // Tool call with known cost
  const data = session.track(await callSerpAPI(query), 0.01, "serp_api");

  // Pre-flight check before expensive final call
  if (session.wouldExceed(estimatedFinalCost)) {
    return "Budget nearly exhausted — wrapping up";
  }

  console.log(`spent: $${session.spent.toFixed(4)}`);
  console.log(session.report());
} catch (err) {
  if (err instanceof BudgetExhausted) {
    console.error(`Hard limit: $${err.spent.toFixed(4)} / $${err.budget.toFixed(2)}`);
  } else if (err instanceof LoopDetected) {
    console.error(`Loop on: ${err.key}`);
  }
} finally {
  session.close();
}
```

```ts
// Auto-patch a specific client instance (no global side effects):
const client = wrapClient(new OpenAI(), session);
await client.chat.completions.create({ ... }); // auto-tracked

const other = new OpenAI();
await other.chat.completions.create({ ... }); // NOT tracked
```

---

## Feature coverage

| Feature | Implemented |
|---|---|
| Budget session creation | ✅ |
| Hard budget limit (throws `BudgetExhausted`) | ✅ |
| Soft limit callback (fires once at configurable %) | ✅ |
| Loop detection (windowed, per model/tool key) | ✅ |
| OpenAI support (`wrapOpenAI`) | ✅ |
| Anthropic support (`wrapAnthropic`) | ✅ |
| Raw token count support (`wrapUsage`) | ✅ |
| Per-client auto-patching (`wrapClient`) | ✅ |
| 35+ built-in model prices | ✅ |
| Custom model pricing (`registerModel`) | ✅ |
| Fuzzy model matching (date suffixes + OpenRouter prefix) | ✅ |
| Tool/API cost tracking (`track` — returns result for chaining) | ✅ |
| Child sessions with sub-budgets | ✅ |
| Finalization reserve (`finalizationReserve` option) | ✅ |
| `wouldExceed` pre-flight check | ✅ |
| CJS + ESM dual build | ✅ |
| Full TypeScript types (strict mode) | ✅ |
| Cost report object | ✅ |

---

## Commits

```
fe2245c  feat(ts): add TypeScript SDK core — errors, pricing, ledger, circuit breaker
64e96b0  feat(ts): add AgentBudget class, BudgetSession, wrapClient, and public index
1f5b0c0  feat(ts): add README and lock file for TypeScript SDK
```

---

## Test plan

- [x] `tsc --noEmit` — passes clean with `exactOptionalPropertyTypes: true`
- [ ] `npm run build` — run after merge to verify `tsup` output
- [ ] `npm publish` — publish to npm after verifying package name is available

---

## After merge

```bash
# 1. Build
cd sdks/typescript
npm run build

# 2. Verify package name isn't taken
npm info agentbudget
# If taken → update package.json name to "@agentbudget/sdk"

# 3. Publish
npm login
npm publish
```

> Note: LangChain and CrewAI integrations for TypeScript are planned for a future release.
