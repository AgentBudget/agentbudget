"use client";

import { Nav } from "@/components/nav";
import { CodeBlock } from "@/components/code-block";
import { DocsMobileNav } from "@/components/docs-mobile-nav";
import { MultiLangCode } from "@/components/multi-lang-code";

const sidebarSections = [
  {
    title: "Getting Started",
    items: [
      { label: "Installation", id: "installation" },
      { label: "Quickstart", id: "quickstart" },
      { label: "Drop-in Mode", id: "drop-in" },
      { label: "Manual Mode", id: "manual" },
    ],
  },
  {
    title: "Core Concepts",
    items: [
      { label: "Budget Envelope", id: "budget-envelope" },
      { label: "Cost Sources", id: "cost-sources" },
      { label: "Circuit Breaker", id: "circuit-breaker" },
      { label: "Cost Report", id: "cost-report" },
    ],
  },
  {
    title: "Features",
    items: [
      { label: "Streaming", id: "streaming" },
      { label: "Per-Client Tracking", id: "per-client" },
      { label: "Finalization Reserve", id: "finalization-reserve" },
      { label: "Async Support", id: "async" },
      { label: "Nested Budgets", id: "nested-budgets" },
      { label: "Webhooks", id: "webhooks" },
      { label: "Event Callbacks", id: "callbacks" },
    ],
  },
  {
    title: "Integrations",
    items: [
      { label: "LangChain", id: "langchain" },
      { label: "CrewAI", id: "crewai" },
    ],
  },
  {
    title: "Reference",
    items: [
      { label: "API Reference", id: "api-reference" },
      { label: "Supported Models", id: "supported-models" },
      { label: "Custom Model Pricing", id: "custom-pricing" },
      { label: "Exceptions", id: "exceptions" },
    ],
  },
];

export default function DocsPage() {

  return (
    <div className="min-h-screen bg-noise">
      <Nav />
      <div className="mx-auto flex max-w-[1200px]">
        {/* Sidebar */}
        <aside className="sticky top-14 hidden h-[calc(100vh-56px)] w-[220px] shrink-0 overflow-y-auto border-x border-border py-8 md:block">
          {/* Language switcher */}
          <div className="mb-6 border-b border-border pb-6">
            <h4 className="mb-2 px-5 font-mono text-[10px] font-semibold uppercase tracking-widest text-muted">
              Language
            </h4>
            <a href="#installation" className="flex items-center gap-2 px-5 py-1.5 text-[13px] text-accent-bright transition-colors hover:bg-surface hover:no-underline">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" />
              Python
            </a>
            <a href="https://github.com/AgentBudget/agentbudget/tree/main/sdks/go" className="block px-5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-surface hover:text-foreground hover:no-underline" target="_blank" rel="noopener noreferrer">
              Go
              <span className="ml-2 font-mono text-[10px] text-muted">↗</span>
            </a>
            <a href="https://github.com/AgentBudget/agentbudget/tree/main/sdks/typescript" className="block px-5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-surface hover:text-foreground hover:no-underline" target="_blank" rel="noopener noreferrer">
              TypeScript
              <span className="ml-2 font-mono text-[10px] text-muted">↗</span>
            </a>
          </div>
          {sidebarSections.map((section) => (
            <div key={section.title} className="mb-6">
              <h4 className="mb-2 px-5 font-mono text-[10px] font-semibold uppercase tracking-widest text-muted">
                {section.title}
              </h4>
              {section.items.map((item) => (
                <a
                  key={item.id}
                  href={`#${item.id}`}
                  className="block px-5 py-1.5 text-[13px] text-muted-foreground transition-colors hover:bg-surface hover:text-foreground hover:no-underline"
                >
                  {item.label}
                </a>
              ))}
            </div>
          ))}
        </aside>

        <DocsMobileNav />

        {/* Content */}
        <main className="min-w-0 flex-1 border-r border-border px-8 py-12 md:px-12">
          <h1 className="mb-3 text-3xl font-bold tracking-tight">Documentation</h1>
          <p className="mb-10 text-[15px] text-muted-foreground">
            Everything you need to add real-time cost enforcement to your AI agents.
          </p>

          {/* Installation */}
          <h2 id="installation" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Installation
          </h2>
          <MultiLangCode
            python={`pip install agentbudget`}
            go={`go get github.com/AgentBudget/agentbudget/sdks/go`}
            typescript={`npm install agentbudget`}
            pythonLang="bash"
            goLang="bash"
            tsLang="bash"
          />
          <p className="mt-4 text-[14px] text-muted-foreground">
            Python 3.9+ · Go 1.21+ · Node.js 18+. Zero external dependencies in all three SDKs.
          </p>
          <p className="mt-2 text-[14px] text-muted-foreground">
            For Python LangChain integration:
          </p>
          <CodeBlock lang="bash">{`pip install agentbudget[langchain]`}</CodeBlock>

          {/* Quickstart */}
          <h2 id="quickstart" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Quickstart
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            AgentBudget offers two modes: <strong className="text-foreground">drop-in</strong> (zero code changes) and{" "}
            <strong className="text-foreground">manual</strong> (explicit wrapping).
          </p>

          {/* Drop-in */}
          <h3 id="drop-in" className="mb-4 mt-10 text-lg font-semibold">
            Drop-in Mode <span className="ml-2 bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent-bright">Recommended</span>
          </h3>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Add two lines to the top of your script. Every OpenAI and Anthropic call is tracked automatically.
          </p>
          <CodeBlock>{`import agentbudget
import openai

agentbudget.init("$5.00")

# Your existing code — no changes needed
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

print(agentbudget.spent())      # e.g. 0.0035
print(agentbudget.remaining())  # e.g. 4.9965
print(agentbudget.report())     # Full cost breakdown

agentbudget.teardown()  # Stop tracking, get final report`}</CodeBlock>

          <div className="mt-4 border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13px] text-muted-foreground">
            <strong className="text-foreground">How it works:</strong>{" "}
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.init()</code> monkey-patches{" "}
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">Completions.create</code> and{" "}
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">Messages.create</code> on the OpenAI and Anthropic SDKs.{" "}
            Same pattern used by Sentry, Datadog, and other observability tools.
          </div>

          <h3 className="mb-3 mt-8 text-base font-semibold">Drop-in API</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b-2 border-border">
                  <th className="py-2 pr-4 text-left font-semibold">Function</th>
                  <th className="py-2 text-left font-semibold">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.init(budget)</code></td><td className="py-2">Start tracking. Patches OpenAI/Anthropic. Returns the session.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.spent()</code></td><td className="py-2">Total dollars spent so far.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.remaining()</code></td><td className="py-2">Dollars left in the budget.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.report()</code></td><td className="py-2">Full cost breakdown as a dict.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.track(result, cost, tool_name)</code></td><td className="py-2">Manually track a tool/API call cost.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.wrap_client(client, session)</code></td><td className="py-2">Attach tracking to a specific client instance only.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.register_model(name, input, output)</code></td><td className="py-2">Add pricing for a new model at runtime.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.register_models(dict)</code></td><td className="py-2">Batch register pricing for multiple models.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.get_session()</code></td><td className="py-2">Get the active session for advanced use.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.teardown()</code></td><td className="py-2">Stop tracking, unpatch SDKs, return final report.</td></tr>
              </tbody>
            </table>
          </div>

          {/* Manual Mode */}
          <h3 id="manual" className="mb-4 mt-10 text-lg font-semibold">Manual Mode</h3>
          <MultiLangCode
            python={`from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$5.00")

with budget.session() as session:
    response = session.wrap(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Analyze this..."}]
        )
    )

    data = session.track(call_serp_api(query), cost=0.01, tool_name="serp")

print(session.report())`}
            go={`import agentbudget "github.com/AgentBudget/agentbudget/sdks/go"

budget, _ := agentbudget.New("$5.00")
session := budget.NewSession()
defer session.Close()

// After your OpenAI or Anthropic call:
resp, _ := openaiClient.CreateChatCompletion(ctx, req)
if err := session.WrapUsage(resp.Model,
    int64(resp.Usage.PromptTokens),
    int64(resp.Usage.CompletionTokens),
); err != nil {
    log.Fatal(err) // *agentbudget.BudgetExhausted or *agentbudget.LoopDetected
}

// Track a tool call:
session.Track(0.01, "serp_api")

fmt.Printf("%+v\\n", session.Report())`}
            typescript={`import { AgentBudget } from "agentbudget";
import OpenAI from "openai";

const budget = new AgentBudget("$5.00");
const session = budget.newSession();

const resp = await new OpenAI().chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Analyze this..." }],
});

session.wrapOpenAI(resp);  // extracts model + tokens, records cost
session.track(null, 0.01, "serp_api");  // track a tool call

console.log(session.report());
session.close();`}
          />

          {/* Budget Envelope */}
          <h2 id="budget-envelope" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Budget Envelope
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            A budget envelope is a dollar amount assigned to a unit of work. Every cost is tracked in real time. When exhausted, <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">BudgetExhausted</code> is raised.
          </p>
          <CodeBlock>{`# All of these work:
AgentBudget(max_spend="$5.00")
AgentBudget(max_spend="5.00")
AgentBudget(max_spend=5.0)
AgentBudget(max_spend=5)`}</CodeBlock>

          {/* Cost Sources */}
          <h2 id="cost-sources" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Cost Sources
          </h2>
          <ul className="mb-4 list-inside list-disc space-y-2 text-[14px] text-muted-foreground">
            <li><strong className="text-foreground">LLM calls</strong> — Automatically costed using a built-in pricing table. Use <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">session.wrap(response)</code> or drop-in mode.</li>
            <li><strong className="text-foreground">Tool calls</strong> — External APIs with known per-call costs. Use <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">session.track(result, cost=0.01)</code>.</li>
            <li><strong className="text-foreground">Decorated functions</strong> — Annotate with <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">@session.track_tool(cost=0.02)</code> to auto-track on every call.</li>
          </ul>

          {/* Circuit Breaker */}
          <h2 id="circuit-breaker" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Circuit Breaker
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">Three levels of protection:</p>
          <CodeBlock>{`budget = AgentBudget(
    max_spend="$5.00",
    soft_limit=0.9,               # Warn at 90%
    max_repeated_calls=10,        # Trip after 10 repeated calls
    loop_window_seconds=60.0,     # Within a 60-second window
    on_soft_limit=lambda r: print("Warning: 90% budget used"),
    on_hard_limit=lambda r: alert_ops_team(r),
    on_loop_detected=lambda r: print("Loop detected!"),
)`}</CodeBlock>
          <ul className="mt-4 list-inside list-disc space-y-2 text-[14px] text-muted-foreground">
            <li><strong className="text-foreground">Soft limit</strong> (default 90%) — Fires a callback. Agent can wrap up gracefully.</li>
            <li><strong className="text-foreground">Hard limit</strong> (100%) — Raises <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">BudgetExhausted</code>. No more calls.</li>
            <li><strong className="text-foreground">Loop detection</strong> — Catches repeated calls before they drain the budget. Raises <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">LoopDetected</code>.</li>
          </ul>

          {/* Cost Report */}
          <h2 id="cost-report" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Cost Report
          </h2>
          <CodeBlock lang="json">{`{
    "session_id": "sess_abc123",
    "budget": 5.00,
    "total_spent": 3.42,
    "remaining": 1.58,
    "breakdown": {
        "llm": {"total": 3.12, "calls": 8, "by_model": {"gpt-4o": 2.80}},
        "tools": {"total": 0.30, "calls": 6, "by_tool": {"serp_api": 0.05}}
    },
    "duration_seconds": 34.2,
    "terminated_by": null,
    "events": [...]
}`}</CodeBlock>

          {/* Streaming */}
          <h2 id="streaming" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Streaming Support
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Streaming responses (<code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">stream=True</code>) are fully tracked. Cost is recorded after the stream is exhausted — every chunk passes through to your code unchanged.
          </p>
          <CodeBlock>{`agentbudget.init("$5.00")
client = openai.OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this report"}],
    stream=True,
    stream_options={"include_usage": True},  # required for OpenAI
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

print(agentbudget.spent())  # cost recorded after stream exhausted`}</CodeBlock>
          <div className="mt-4 border-l-2 border-amber-500 bg-amber-500/5 px-4 py-3 text-[13px] text-muted-foreground">
            <strong className="text-foreground">OpenAI note:</strong> You must pass{" "}
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">{'stream_options={"include_usage": True}'}</code>{" "}
            for token counts to appear on the final chunk. Without it, streaming calls are silently tracked as{" "}
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">$0.00</code> — no error, just no cost.
            Anthropic streams always include usage automatically.
          </div>
          <p className="mt-4 text-[14px] text-muted-foreground">
            Both for-loop and context-manager patterns are supported, sync and async:
          </p>
          <CodeBlock>{`# async for
async for chunk in await client.chat.completions.create(
    stream=True, stream_options={"include_usage": True}, ...
):
    process(chunk)

# context manager
with client.chat.completions.create(stream=True, ...) as stream:
    for chunk in stream:
        process(chunk)`}</CodeBlock>

          {/* Per-Client Tracking */}
          <h2 id="per-client" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Per-Client Tracking
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            By default, <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">agentbudget.init()</code> patches all OpenAI/Anthropic calls globally. For finer control — multiple budgets, isolated clients, or production apps where global side effects are undesirable — use <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">wrap_client()</code>:
          </p>
          <CodeBlock>{`import agentbudget
from agentbudget import AgentBudget
import openai

budget = AgentBudget(max_spend="$5.00")
with budget.session() as session:
    # Only this instance is tracked
    client = agentbudget.wrap_client(openai.OpenAI(), session)
    response = client.chat.completions.create(...)  # tracked

    other = openai.OpenAI()
    other.chat.completions.create(...)              # NOT tracked`}</CodeBlock>
          <p className="mt-4 text-[14px] text-muted-foreground">
            Works with <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">openai.OpenAI</code>, <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">openai.AsyncOpenAI</code>, <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">anthropic.Anthropic</code>, and <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">anthropic.AsyncAnthropic</code>.
            Global patching via <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">init()</code> is unchanged — both approaches coexist.
          </p>

          {/* Finalization Reserve */}
          <h2 id="finalization-reserve" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Finalization Reserve
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Prevent your agent from being cut off mid-task. Reserve a fraction of the budget exclusively for the final response step — the hard limit fires early, keeping that slice free.
          </p>
          <CodeBlock>{`budget = AgentBudget(
    max_spend="$1.00",
    finalization_reserve=0.05,  # hard limit at $0.95, $0.05 reserved for final call
)`}</CodeBlock>
          <p className="mt-4 mb-3 text-[14px] text-muted-foreground">
            For manual control, check before the final call with <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">session.would_exceed()</code>:
          </p>
          <CodeBlock>{`with budget.session() as session:
    # ... do work ...

    if session.would_exceed(estimated_final_cost):
        return "Budget nearly exhausted — here is what was completed: ..."

    # Safe to proceed
    response = session.wrap(client.chat.completions.create(...))`}</CodeBlock>
          <div className="mt-4 border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13px] text-muted-foreground">
            <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">would_exceed(cost)</code> checks against the remaining budget without recording anything. Use it as a pre-flight check before expensive final steps.
          </div>

          {/* Async */}
          <h2 id="async" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Async Support
          </h2>
          <CodeBlock>{`from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$5.00")

async with budget.async_session() as session:
    response = await session.wrap_async(
        client.chat.completions.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
    )

    @session.track_tool(cost=0.01)
    async def async_search(query):
        return await api.search(query)`}</CodeBlock>

          {/* Nested Budgets */}
          <h2 id="nested-budgets" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Nested Budgets
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Parent sessions allocate sub-budgets to child tasks. When the child finishes, its total spend is charged to the parent.
          </p>
          <CodeBlock>{`with budget.session() as parent:
    child = parent.child_session(max_spend=2.0)
    with child:
        child.track("result", cost=1.50, tool_name="sub_task")

    print(parent.spent)      # 1.50
    print(parent.remaining)  # 8.50`}</CodeBlock>
          <div className="mt-4 border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13px] text-muted-foreground">
            The child budget is automatically capped at the lesser of <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">max_spend</code> and the parent&apos;s remaining balance.
          </div>

          {/* Webhooks */}
          <h2 id="webhooks" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Webhooks
          </h2>
          <CodeBlock>{`budget = AgentBudget(
    max_spend="$5.00",
    webhook_url="https://your-app.com/api/budget-events",
)`}</CodeBlock>
          <p className="mt-4 text-[14px] text-muted-foreground">
            Events are sent as JSON POST requests with <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">event_type</code> ({'"'}soft_limit{'"'}, {'"'}hard_limit{'"'}, {'"'}loop_detected{'"'}) and the full cost report. Failures are logged but never raise.
          </p>

          {/* Callbacks */}
          <h2 id="callbacks" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Event Callbacks
          </h2>
          <CodeBlock>{`budget = AgentBudget(
    max_spend="$5.00",
    on_soft_limit=lambda r: logger.warning(f"90% used: {r}"),
    on_hard_limit=lambda r: alert_ops_team(r),
    on_loop_detected=lambda r: logger.error(f"Loop: {r}"),
)`}</CodeBlock>
          <p className="mt-4 text-[14px] text-muted-foreground">
            When <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">webhook_url</code> is also set, both your callback and the webhook fire.
          </p>

          {/* LangChain */}
          <h2 id="langchain" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            LangChain Integration
            <span className="ml-3 bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent-bright align-middle">Python only</span>
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Go and TypeScript integrations are coming in a future release.
          </p>
          <CodeBlock lang="bash">{`pip install agentbudget[langchain]`}</CodeBlock>
          <CodeBlock>{`from agentbudget.integrations.langchain import LangChainBudgetCallback

callback = LangChainBudgetCallback(budget="$5.00")

agent.run(
    "Research competitors in the CRM space",
    callbacks=[callback]
)

print(callback.get_report())`}</CodeBlock>

          {/* CrewAI */}
          <h2 id="crewai" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            CrewAI Integration
            <span className="ml-3 bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent-bright align-middle">Python only</span>
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            Go and TypeScript integrations are coming in a future release.
          </p>
          <CodeBlock>{`from agentbudget.integrations.crewai import CrewAIBudgetMiddleware

with CrewAIBudgetMiddleware(budget="$3.00") as middleware:
    result = middleware.track(
        crew.kickoff(),
        cost=0.50,
        tool_name="crew_run"
    )

print(middleware.get_report())`}</CodeBlock>

          {/* API Reference */}
          <h2 id="api-reference" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            API Reference
          </h2>

          <h3 className="mb-3 mt-6 text-base font-semibold">AgentBudget</h3>
          <CodeBlock>{`AgentBudget(
    max_spend: str | float | int,
    soft_limit: float = 0.9,
    max_repeated_calls: int = 10,
    loop_window_seconds: float = 60.0,
    on_soft_limit: Callable = None,
    on_hard_limit: Callable = None,
    on_loop_detected: Callable = None,
    webhook_url: str = None,
    finalization_reserve: float = 0.0,  # fraction of budget reserved for final step
)`}</CodeBlock>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Method</th><th className="py-2 pr-4 text-left font-semibold">Returns</th><th className="py-2 text-left font-semibold">Description</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.session()</code></td><td className="py-2 pr-4">BudgetSession</td><td className="py-2">Create a sync budget session</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.async_session()</code></td><td className="py-2 pr-4">AsyncBudgetSession</td><td className="py-2">Create an async budget session</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.max_spend</code></td><td className="py-2 pr-4">float</td><td className="py-2">The configured budget amount</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="mb-3 mt-8 text-base font-semibold">BudgetSession</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Method / Property</th><th className="py-2 text-left font-semibold">Description</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.wrap(response)</code></td><td className="py-2">Extract model/tokens from LLM response and record cost. Returns response.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.track(result, cost, tool_name)</code></td><td className="py-2">Record a tool call cost. Returns the result.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.track_tool(cost, tool_name)</code></td><td className="py-2">Decorator that tracks cost on every call.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.child_session(max_spend)</code></td><td className="py-2">Create child session with sub-budget. Costs roll up.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.would_exceed(cost)</code></td><td className="py-2">Returns True if cost would exceed the remaining budget. Does not record anything.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.report()</code></td><td className="py-2">Full cost report as a dict.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.spent</code></td><td className="py-2">Total dollars spent (float).</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">.remaining</code></td><td className="py-2">Dollars remaining (float).</td></tr>
              </tbody>
            </table>
          </div>

          {/* Supported Models */}
          <h2 id="supported-models" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Supported Models
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">Built-in pricing for 50+ models. Updated February 2026.</p>

          <h3 className="mb-2 mt-6 text-sm font-semibold text-accent-bright">OpenAI</h3>
          <div className="overflow-x-auto">
            <table className="mb-6 w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Model</th><th className="py-2 pr-4 text-left font-semibold">Input / 1M</th><th className="py-2 text-left font-semibold">Output / 1M</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gpt-4.1</td><td className="py-1.5 pr-4">$2.00</td><td className="py-1.5">$8.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gpt-4.1-mini</td><td className="py-1.5 pr-4">$0.40</td><td className="py-1.5">$1.60</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gpt-4.1-nano</td><td className="py-1.5 pr-4">$0.10</td><td className="py-1.5">$0.40</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gpt-4o</td><td className="py-1.5 pr-4">$2.50</td><td className="py-1.5">$10.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gpt-4o-mini</td><td className="py-1.5 pr-4">$0.15</td><td className="py-1.5">$0.60</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">o3</td><td className="py-1.5 pr-4">$2.00</td><td className="py-1.5">$8.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">o3-mini</td><td className="py-1.5 pr-4">$1.10</td><td className="py-1.5">$4.40</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">o4-mini</td><td className="py-1.5 pr-4">$1.10</td><td className="py-1.5">$4.40</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">o1</td><td className="py-1.5 pr-4">$15.00</td><td className="py-1.5">$60.00</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="mb-2 text-sm font-semibold text-accent-bright">Anthropic</h3>
          <div className="overflow-x-auto">
            <table className="mb-6 w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Model</th><th className="py-2 pr-4 text-left font-semibold">Input / 1M</th><th className="py-2 text-left font-semibold">Output / 1M</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-1.5 pr-4">claude-opus-4-6</td><td className="py-1.5 pr-4">$5.00</td><td className="py-1.5">$25.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">claude-sonnet-4.5</td><td className="py-1.5 pr-4">$3.00</td><td className="py-1.5">$15.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">claude-haiku-4.5</td><td className="py-1.5 pr-4">$1.00</td><td className="py-1.5">$5.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">claude-3.5-sonnet</td><td className="py-1.5 pr-4">$3.00</td><td className="py-1.5">$15.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">claude-3.5-haiku</td><td className="py-1.5 pr-4">$0.80</td><td className="py-1.5">$4.00</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="mb-2 text-sm font-semibold text-accent-bright">Google Gemini</h3>
          <div className="overflow-x-auto">
            <table className="mb-6 w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Model</th><th className="py-2 pr-4 text-left font-semibold">Input / 1M</th><th className="py-2 text-left font-semibold">Output / 1M</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gemini-2.5-pro</td><td className="py-1.5 pr-4">$1.25</td><td className="py-1.5">$10.00</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gemini-2.5-flash</td><td className="py-1.5 pr-4">$0.30</td><td className="py-1.5">$2.50</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gemini-2.0-flash</td><td className="py-1.5 pr-4">$0.10</td><td className="py-1.5">$0.40</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">gemini-1.5-pro</td><td className="py-1.5 pr-4">$1.25</td><td className="py-1.5">$5.00</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="mb-2 text-sm font-semibold text-accent-bright">Mistral &amp; Cohere</h3>
          <div className="overflow-x-auto">
            <table className="mb-6 w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Model</th><th className="py-2 pr-4 text-left font-semibold">Input / 1M</th><th className="py-2 text-left font-semibold">Output / 1M</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-1.5 pr-4">mistral-large</td><td className="py-1.5 pr-4">$0.50</td><td className="py-1.5">$1.50</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">mistral-small</td><td className="py-1.5 pr-4">$0.03</td><td className="py-1.5">$0.11</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">codestral</td><td className="py-1.5 pr-4">$0.30</td><td className="py-1.5">$0.90</td></tr>
                <tr className="border-b border-border"><td className="py-1.5 pr-4">command-r-plus</td><td className="py-1.5 pr-4">$2.50</td><td className="py-1.5">$10.00</td></tr>
              </tbody>
            </table>
          </div>

          <div className="border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13px] text-muted-foreground">
            Missing a model? Register it at runtime with <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">register_model()</code> or submit a PR to <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">pricing.json</code> and run <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">python scripts/generate_pricing.py</code>.
          </div>

          {/* Custom Model Pricing */}
          <h2 id="custom-pricing" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Custom Model Pricing
          </h2>
          <p className="mb-4 text-[14px] text-muted-foreground">
            New model just launched? Don{"'"}t wait for a release — register pricing at runtime.
          </p>

          <h3 className="mb-2 mt-6 text-sm font-semibold">Single model</h3>
          <MultiLangCode
            python={`import agentbudget

agentbudget.register_model(
    "gpt-5",
    input_price_per_million=5.00,
    output_price_per_million=20.00,
)`}
            go={`import agentbudget "github.com/AgentBudget/agentbudget/sdks/go"

agentbudget.RegisterModel("gpt-5", 5.00, 20.00)`}
            typescript={`import { registerModel } from "@agentbudget/agentbudget";

registerModel("gpt-5", 5.00, 20.00);`}
          />

          <h3 className="mb-2 mt-6 text-sm font-semibold">Batch register</h3>
          <MultiLangCode
            python={`agentbudget.register_models({
    "gpt-5": (5.00, 20.00),
    "gpt-5-mini": (0.50, 2.00),
})`}
            go={`agentbudget.RegisterModel("gpt-5", 5.00, 20.00)
agentbudget.RegisterModel("gpt-5-mini", 0.50, 2.00)`}
            typescript={`import { registerModel } from "@agentbudget/agentbudget";

registerModel("gpt-5", 5.00, 20.00);
registerModel("gpt-5-mini", 0.50, 2.00);`}
          />

          <h3 className="mb-2 mt-6 text-sm font-semibold">Fuzzy matching</h3>
          <p className="mb-3 text-[14px] text-muted-foreground">
            Dated model variants are automatically matched to their base model. For example, <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">gpt-4o-2025-06-15</code> automatically uses <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">gpt-4o</code> pricing.
          </p>

          <div className="border-l-2 border-accent bg-accent/5 px-4 py-3 text-[13px] text-muted-foreground">
            <strong className="text-foreground">Resolution order:</strong> Custom pricing (via <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">register_model</code>) → Built-in table → Fuzzy match (strip date suffixes) → OpenRouter prefix strip (<code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">"openai/gpt-4o"</code> → <code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">"gpt-4o"</code>).
          </div>

          {/* Exceptions */}
          <h2 id="exceptions" className="mb-4 mt-16 border-t border-border pt-8 text-xl font-semibold">
            Exceptions
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead><tr className="border-b-2 border-border"><th className="py-2 pr-4 text-left font-semibold">Exception</th><th className="py-2 text-left font-semibold">When</th></tr></thead>
              <tbody className="text-muted-foreground">
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">BudgetExhausted</code></td><td className="py-2">Session exceeded its dollar budget (hard limit).</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">LoopDetected</code></td><td className="py-2">Repeated calls to the same tool/model detected.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">InvalidBudget</code></td><td className="py-2">Budget string couldn&apos;t be parsed.</td></tr>
                <tr className="border-b border-border"><td className="py-2 pr-4"><code className="bg-code-bg px-1.5 py-0.5 text-[12px] text-accent-bright">AgentBudgetError</code></td><td className="py-2">Base exception for all AgentBudget errors.</td></tr>
              </tbody>
            </table>
          </div>

          <div className="h-24" />
        </main>
      </div>
    </div>
  );
}
