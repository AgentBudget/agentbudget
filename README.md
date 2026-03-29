# AgentBudget

### Real-time cost enforcement for AI agent sessions.

[![PyPI](https://img.shields.io/pypi/v/agentbudget)](https://pypi.org/project/agentbudget/)
[![Python](https://img.shields.io/pypi/pyversions/agentbudget)](https://pypi.org/project/agentbudget/)
[![Downloads](https://static.pepy.tech/personalized-badge/agentbudget?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/agentbudget)
[![License](https://img.shields.io/github/license/sahiljagtap08/agentbudget)](https://github.com/sahiljagtap08/agentbudget/blob/main/LICENSE)

[Website](https://agentbudget.dev) · [Docs](https://agentbudget.dev/docs) · [PyPI](https://pypi.org/project/agentbudget/)

---

## What is AgentBudget?

AgentBudget is an open-source Python SDK that puts a hard dollar limit on any AI agent session. It wraps LLM calls, tool calls, and external API requests with real-time cost tracking and automatic circuit breaking — so your agent can never silently burn through your budget.

**One line to set a budget. Zero infrastructure to manage. Works with any LLM provider.**

---

## Quickstart

### Drop-in Mode (Recommended)

Two lines. Zero code changes to your existing agent.

```python
import agentbudget
import openai

agentbudget.init("$5.00")

# Your existing code — no changes needed
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze this market..."}]
)

print(agentbudget.spent())      # e.g. 0.0035
print(agentbudget.remaining())  # e.g. 4.9965
print(agentbudget.report())     # Full cost breakdown

agentbudget.teardown()  # Stop tracking, get final report
```

`agentbudget.init()` patches OpenAI and Anthropic SDKs so every call is tracked automatically. `teardown()` restores the originals. Same pattern as Sentry and Datadog.

### Manual Mode

For full control, use the context manager API directly.

```python
from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$5.00")

with budget.session() as session:
    # Auto-cost LLM responses
    response = session.wrap(openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Research competitors in the CRM space"}]
    ))

    # Track tool/API calls with known costs
    result = session.track(call_serp_api(query="CRM market"), cost=0.01)

    # When the $5 limit is hit, BudgetExhausted is raised
    # No silent overruns. No surprise bills.

print(session.report())
```

---

## Install

```bash
pip install agentbudget
```

Python 3.9+. No external dependencies.

For LangChain integration:
```bash
pip install agentbudget[langchain]
```

---

## Drop-in API

| Function | Description |
|---|---|
| `agentbudget.init(budget)` | Start tracking. Patches OpenAI/Anthropic. Returns session. |
| `agentbudget.spent()` | Total dollars spent so far. |
| `agentbudget.remaining()` | Dollars left in the budget. |
| `agentbudget.report()` | Full cost breakdown as a dict. |
| `agentbudget.track(result, cost, tool_name)` | Manually track a tool/API call cost. |
| `agentbudget.wrap_client(client, session)` | Attach tracking to a specific client instance only. |
| `agentbudget.register_model(name, input, output)` | Add pricing for a new model at runtime. |
| `agentbudget.register_models(dict)` | Batch register pricing for multiple models. |
| `agentbudget.get_session()` | Get the active session for advanced use. |
| `agentbudget.teardown()` | Stop tracking, unpatch SDKs, return final report. |

---

## Features

### Streaming Support

Streaming responses (`stream=True`) are fully tracked. Cost is recorded after the stream is exhausted — chunks pass through to your code unchanged.

```python
# Drop-in mode — works automatically
agentbudget.init("$5.00")
client = openai.OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this report"}],
    stream=True,
    stream_options={"include_usage": True},  # required for OpenAI cost tracking
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

print(agentbudget.spent())  # cost recorded after stream exhausted
```

> **OpenAI note:** You must pass `stream_options={"include_usage": True}` for token counts to appear on the final chunk. Without it, streaming calls are silently tracked as `$0.00` (no error). Anthropic streams always include usage — no extra option needed.

Async streaming works the same way:

```python
async for chunk in await client.chat.completions.create(stream=True, stream_options={"include_usage": True}, ...):
    ...
```

---

### Explicit Per-Client Tracking

By default, `agentbudget.init()` patches all OpenAI/Anthropic calls globally. If you need finer control — multiple budgets, different clients per task, or just prefer explicit scope — use `wrap_client()`:

```python
from agentbudget import AgentBudget
import agentbudget
import openai

budget = AgentBudget(max_spend="$5.00")
with budget.session() as session:
    # Only this client instance is tracked
    client = agentbudget.wrap_client(openai.OpenAI(), session)
    response = client.chat.completions.create(...)   # tracked

    other = openai.OpenAI()
    other.chat.completions.create(...)               # NOT tracked
```

Works with `openai.OpenAI`, `openai.AsyncOpenAI`, `anthropic.Anthropic`, and `anthropic.AsyncAnthropic`.

---

### Finalization Reserve

Prevent your agent from being cut off mid-task. Reserve a fraction of the budget exclusively for the final response step:

```python
budget = AgentBudget(
    max_spend="$1.00",
    finalization_reserve=0.05,  # hard limit fires at $0.95, last $0.05 stays free
)
```

For manual control, check before the final call:

```python
with budget.session() as session:
    # ... do work ...

    if session.would_exceed(estimated_final_cost):
        return "Here's what I completed so far: ..."

    # Safe to proceed — won't hit the hard limit
    response = session.wrap(client.chat.completions.create(...))
```

---

### Circuit Breaker

Three levels of protection against runaway spend:

```python
budget = AgentBudget(
    max_spend="$5.00",
    soft_limit=0.9,               # Warn at 90% spent
    max_repeated_calls=10,        # Trip after 10 repeated calls
    loop_window_seconds=60.0,     # Within a 60-second window
    on_soft_limit=lambda r: print("90% budget used"),
    on_hard_limit=lambda r: alert_ops_team(r),
    on_loop_detected=lambda r: print("Loop detected!"),
)
```

- **Soft limit** — Fires a callback when spending exceeds a threshold. Agent can wrap up gracefully.
- **Hard limit** — Raises `BudgetExhausted`. No more calls allowed.
- **Loop detection** — Catches infinite loops before they drain the budget.

### Async Support

```python
async with budget.async_session() as session:
    response = await session.wrap_async(
        client.chat.completions.acreate(model="gpt-4o", messages=[...])
    )

    @session.track_tool(cost=0.01)
    async def async_search(query):
        return await api.search(query)
```

### Nested Budgets

Parent sessions allocate sub-budgets to child tasks. Costs roll up automatically.

```python
with budget.session() as parent:
    child = parent.child_session(max_spend=2.0)
    with child:
        child.track("result", cost=1.50, tool_name="sub_task")

    print(parent.spent)      # 1.50
    print(parent.remaining)  # 8.50
```

### Webhooks

Stream budget events to any HTTP endpoint for alerting and billing.

```python
budget = AgentBudget(
    max_spend="$5.00",
    webhook_url="https://your-app.com/api/budget-events",
)
```

Events are sent as JSON with `event_type` (`soft_limit`, `hard_limit`, `loop_detected`) and the full cost report.

### Track Tool Decorator

Annotate any function to auto-track cost on every call.

```python
@session.track_tool(cost=0.02, tool_name="search")
def my_search(query):
    return api.search(query)
```

---

## Integrations

### LangChain / LangGraph

```python
from agentbudget.integrations.langchain import LangChainBudgetCallback

callback = LangChainBudgetCallback(budget="$5.00")
agent.run("Research competitors", callbacks=[callback])
print(callback.get_report())
```

### CrewAI

```python
from agentbudget.integrations.crewai import CrewAIBudgetMiddleware

with CrewAIBudgetMiddleware(budget="$3.00") as middleware:
    result = middleware.track(crew.kickoff(), cost=0.50, tool_name="crew_run")
print(middleware.get_report())
```

### Raw OpenAI / Anthropic SDK

```python
from agentbudget import AgentBudget

budget = AgentBudget("$5.00")
with budget.session() as s:
    response = s.wrap(client.chat.completions.create(...))
```

---

## Supported Models

Built-in pricing for 40+ models across OpenAI, Anthropic, Google Gemini, Mistral, and Cohere.

| Provider | Models |
|---|---|
| **OpenAI** | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3, o3-pro, o4-mini |
| **Anthropic** | claude-opus-4-6, claude-opus-4-5, claude-sonnet-4-5, claude-sonnet-4, claude-haiku-4-5, claude-3-opus, claude-3-sonnet, claude-3-haiku |
| **Google** | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| **Mistral** | mistral-large, mistral-small, mistral-medium, codestral, open-mistral-nemo |
| **Cohere** | command-r-plus, command-r, command, command-light |

### Custom Model Pricing

New model just launched? Don't wait for a release — register it at runtime:

```python
import agentbudget

agentbudget.register_model(
    "gpt-5",
    input_price_per_million=5.00,
    output_price_per_million=20.00,
)

# Or batch register multiple models:
agentbudget.register_models({
    "gpt-5": (5.00, 20.00),
    "gpt-5-mini": (0.50, 2.00),
})
```

Dated model variants (e.g. `gpt-4o-2025-06-15`) are automatically matched to their base model pricing.

**OpenRouter** model names (e.g. `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet"`) are supported — the provider prefix is stripped automatically before the pricing lookup.

Missing a model from built-in pricing? PRs welcome — pricing data is in `agentbudget/pricing.py`.

---

## Cost Report

Every session produces a structured cost report:

```python
{
    "session_id": "sess_abc123",
    "budget": 5.00,
    "total_spent": 3.42,
    "remaining": 1.58,
    "breakdown": {
        "llm": {"total": 3.12, "calls": 8, "by_model": {"gpt-4o": 2.80, "gpt-4o-mini": 0.32}},
        "tools": {"total": 0.30, "calls": 6, "by_tool": {"serp_api": 0.05, "scrape": 0.25}},
    },
    "duration_seconds": 34.2,
    "terminated_by": null,  # or "budget_exhausted" or "loop_detected"
    "events": [...]
}
```

Pipe it to your observability stack, billing system, or just log it.

---

## The Problem

AI agents are unpredictable by design. An agent might make 3 LLM calls or 300, use cheap models or expensive ones, invoke 1 tool or 50.

- **The Loop Problem** — A stuck agent makes 200 LLM calls in 10 minutes. $50-$200 before anyone notices.
- **The Invisible Spend** — Tokens aren't dollars. GPT-4o costs 15x more than GPT-4o-mini for similar token counts.
- **Multi-Provider Chaos** — One session calls OpenAI, Anthropic, Google, and 3 APIs. No unified real-time view.
- **The Scaling Problem** — 1,000 concurrent sessions with 5% failure rate = 50 runaway agents.

**AgentBudget fills the gap:** Real-time, dollar-denominated, per-session budget enforcement that spans LLM calls + tool calls + external APIs, works across providers, and kills runaway sessions automatically.

---

## What It's NOT

- **Not an LLM proxy.** Wraps your existing client calls in-process.
- **Not an observability platform.** Produces cost data — pipe it wherever you want.
- **Not a billing system.** Enforces budgets, doesn't invoice customers.
- **Not infrastructure.** No Redis, no servers, no cloud account. It's a library.

---

## License

Apache 2.0

---

**Ship your agents with confidence. Set a budget. Move on.**
