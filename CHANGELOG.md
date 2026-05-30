# Changelog

All notable changes to AgentBudget are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.4.0] — 2026-05-30

### Added
- **LangChain / LangGraph coverage** — `LangChainBudgetCallback` now extracts usage from modern chat-model `usage_metadata` and Anthropic-style `response_metadata` (not just legacy `LLMResult.token_usage`), so chat models and LangGraph runs are counted instead of silently recording `$0.00`. Adds optional tool-cost tracking (`tool_costs` / `default_tool_cost`), a context-manager/`close()` lifecycle that finalizes the session and fires `on_hard_limit`, and `on_soft_limit`/`on_hard_limit`/`on_loop_detected` forwarding. Warns instead of silently skipping when usage can't be extracted or a model has no pricing. Thanks [@veeradyani222](https://github.com/veeradyani222)! (#26)
- **`InvalidCost` exception** — exported from the package and raised when a tracked cost is negative, `NaN`, or infinite.
- **Contributor docs** — issue/PR templates and a `CONTRIBUTING.md` covering per-SDK test commands and the generated-pricing workflow. Thanks [@sundaram2021](https://github.com/sundaram2021)! (#25)

### Changed
- **Drop-in API isolation** — `init()` / `teardown()` session state is now scoped per thread and per async task via `contextvars` instead of a single process-global session. Concurrent requests no longer overwrite or tear down each other's sessions, and one context's `teardown()` no longer unpatches the SDKs for everyone. Thanks [@sundaram2021](https://github.com/sundaram2021)! (#20)

### Fixed
- **Streaming cost on early exit** — `stream=True` cost is now recorded when the caller breaks out of the stream early or an async stream ends incompletely (`try/finally` for sync wrappers; instance-state + eager recording for async). OpenAI streaming auto-injects `stream_options={"include_usage": True}`, and incomplete-usage streams now log a warning instead of silently under-counting. Thanks [@Kunal1522](https://github.com/Kunal1522)! (#15)
- **Invalid cost values** — negative, `NaN`, and infinite costs are now rejected with `InvalidCost` before touching the ledger. Previously they corrupted the running total (`remaining` could exceed the budget, or become `NaN`). Thanks [@drewster99](https://github.com/drewster99)! (#21)

---

## [0.3.0] — 2026-03-28

### Added
- **Streaming cost tracking** — `stream=True` calls are now fully tracked for OpenAI and Anthropic (sync + async). Chunks pass through unchanged; cost is recorded after the iterator is exhausted. Requires `stream_options={"include_usage": True}` for OpenAI. Thanks [@KTS-o7](https://github.com/KTS-o7)!
- **`agentbudget.wrap_client(client, session)`** — explicit per-client tracking as an alternative to global monkey-patching. Only the wrapped instance is tracked; other clients of the same type are unaffected.
- **`finalization_reserve` param on `AgentBudget`** — reserves a fraction of the budget (e.g. `finalization_reserve=0.05`) for a final completion step. The hard limit fires at `budget * (1 - reserve)`, keeping the remainder free so agents aren't cut off mid-answer.
- **`session.would_exceed(estimated_cost)`** — pre-flight budget check that returns `True` if a cost would exceed the remaining budget without recording anything. Use before expensive final calls.
- **CI workflow** — tests now run on every push and PR across Python 3.9–3.13. Thanks [@KTS-o7](https://github.com/KTS-o7)!

### Fixed
- **OpenRouter model names** — model identifiers with a `provider/` prefix (e.g. `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet"`) now resolve correctly in the pricing table. Previously these silently recorded `$0.00`.
- **Thread safety in `CircuitBreaker`** — the soft limit check-and-set was not atomic under concurrent calls. Fixed with a `threading.Lock`. Thanks [@MistakenPirate](https://github.com/MistakenPirate)!
- **Off-by-one in loop detection** — `max_repeated_calls=N` was allowing N+1 calls before triggering (`>` vs `>=`). Thanks [@MistakenPirate](https://github.com/MistakenPirate)!
- **Exception matching** — exception type checks now use `issubclass()` instead of string comparison on `__name__`, fixing false negatives with subclasses. Thanks [@MistakenPirate](https://github.com/MistakenPirate)!
- **Negative price validation** — `register_model()` now raises `ValueError` for negative input/output prices. Thanks [@MistakenPirate](https://github.com/MistakenPirate)!

---

## [0.2.3] — 2026-03-01

### Added
- Pricing for `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3-pro`, `o4-mini`
- Pricing for `claude-opus-4-6`, `claude-opus-4-5`, `claude-haiku-4-5`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `register_models()` for batch model pricing registration
- Fuzzy matching for dated model variants (e.g. `gpt-4o-2025-06-15` → `gpt-4o`)
- Mobile navigation for docs site

### Fixed
- Import ordering in `Hero` component

---

## [0.2.0] — 2026-02-10

### Added
- `AsyncBudgetSession` with full `async with` and `wrap_async()` support
- `child_session(max_spend)` for nested budgets with automatic cost rollup
- Webhook support — POST budget events to any HTTP endpoint
- `on_soft_limit`, `on_hard_limit`, `on_loop_detected` callbacks
- `track_tool()` decorator for auto-tracking function costs
- `AgentBudgetError` base exception class

### Changed
- `AgentBudget` is now the primary entry point; functional API deprecated in favour of class-based sessions

---

## [0.1.0] — 2026-01-20

### Added
- Initial release
- `agentbudget.init()` drop-in mode with global OpenAI/Anthropic patching
- `BudgetSession` with `wrap()`, `track()`, and `report()`
- Built-in pricing for OpenAI and Anthropic models
- `CircuitBreaker` with soft limit, hard limit, and loop detection
- `Ledger` for thread-safe cost tracking
- `register_model()` for custom model pricing
