"use client";

import { useState } from "react";

const tabs = [
  { label: "Drop-in Mode", accent: "text-accent-bright", borderColor: "border-accent" },
  { label: "Streaming", accent: "text-accent-bright", borderColor: "border-accent" },
  { label: "Manual Session", accent: "text-accent-bright", borderColor: "border-accent" },
  { label: "Async Agent", accent: "text-accent-bright", borderColor: "border-accent" },
  { label: "Nested Budgets", accent: "text-accent-bright", borderColor: "border-accent" },
];

const codeBlocks = [
  // Drop-in Mode
  `import agentbudget
import openai

agentbudget.init("$5.00")

# Your existing code. Zero changes needed.
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze this market..."}]
)
# ^ Automatically tracked. Budget enforced.

print(agentbudget.spent())       # 0.0035
print(agentbudget.remaining())   # 4.9965
print(agentbudget.report())      # Full cost breakdown

agentbudget.teardown()`,

  // Streaming
  `import agentbudget
import openai

agentbudget.init("$5.00")
client = openai.OpenAI()

# stream=True is fully tracked — chunks pass through unchanged
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this report"}],
    stream=True,
    stream_options={"include_usage": True},  # required for OpenAI
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")

# Cost recorded after stream exhausted
print(agentbudget.spent())   # e.g. 0.0042
print(agentbudget.remaining())`,

  // Manual Session
  `from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$5.00")

with budget.session() as session:
    # Auto-cost LLM responses
    response = session.wrap(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
    )

    # Track tool calls with known costs
    data = session.track(call_api(query), cost=0.01, tool_name="serp")

    # Decorate functions for auto-tracking
    @session.track_tool(cost=0.02, tool_name="search")
    def my_search(query):
        return api.search(query)

print(session.report())`,

  // Async Agent
  `from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$5.00")

async with budget.async_session() as session:
    # Await and auto-cost in one call
    response = await session.wrap_async(
        client.chat.completions.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Research this topic"}]
        )
    )

    # Works with both sync and async functions
    @session.track_tool(cost=0.01)
    async def async_search(query):
        return await api.search(query)

    result = await async_search("AI market trends")`,

  // Nested Budgets
  `from agentbudget import AgentBudget

budget = AgentBudget(max_spend="$10.00")

with budget.session() as parent:
    # Allocate $2 for a research sub-task
    research = parent.child_session(max_spend=2.0)
    with research:
        research.wrap(client.chat.completions.create(...))
        research.track(serp_api(), cost=0.05, tool_name="search")

    # Allocate $3 for an analysis sub-task
    analysis = parent.child_session(max_spend=3.0)
    with analysis:
        analysis.wrap(client.chat.completions.create(...))

    # Both child costs roll up to parent automatically
    print(parent.remaining)  # Budget minus all child spend`,
];

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightPython(code: string): string {
  const keywords = [
    "import", "from", "def", "async", "await", "with", "as", "return",
    "class", "for", "in", "if", "else", "elif", "try", "except", "raise",
    "while", "True", "False", "None", "not", "and", "or", "is", "lambda",
    "print",
  ];

  return code
    .split("\n")
    .map((line) => {
      // Separate comment from code
      let mainPart = escapeHtml(line);
      let commentPart = "";

      const hashIdx = line.indexOf("#");
      if (hashIdx !== -1) {
        const before = line.substring(0, hashIdx);
        const singleQ = (before.match(/'/g) || []).length;
        const doubleQ = (before.match(/"/g) || []).length;
        if (singleQ % 2 === 0 && doubleQ % 2 === 0) {
          mainPart = escapeHtml(line.substring(0, hashIdx));
          commentPart = `<span style="color:#484f58">${escapeHtml(line.substring(hashIdx))}</span>`;
        }
      }

      // Strings (double and single quoted)
      mainPart = mainPart.replace(
        /(&quot;[^&]*?&quot;|&#x27;[^&]*?&#x27;|"[^"]*?"|'[^']*?')/g,
        '<span style="color:#a5d6ff">$1</span>'
      );

      // Decorators
      mainPart = mainPart.replace(
        /(@\w+)/g,
        '<span style="color:#d2a8ff">$1</span>'
      );

      // Keywords
      for (const kw of keywords) {
        const re = new RegExp(`\\b(${kw})\\b`, "g");
        mainPart = mainPart.replace(re, '<span style="color:#ff7b72">$1</span>');
      }

      // Numbers (standalone)
      mainPart = mainPart.replace(
        /\b(\d+\.?\d*)\b/g,
        '<span style="color:#ffa657">$1</span>'
      );

      // Function calls (word followed by parenthesis)
      mainPart = mainPart.replace(
        /(\w+)\(/g,
        '<span style="color:#d2a8ff">$1</span>('
      );

      // Built-in functions
      mainPart = mainPart.replace(
        /\b(print|len|range|str|int|float|dict|list|type)\b/g,
        '<span style="color:#79c0ff">$1</span>'
      );

      return mainPart + commentPart;
    })
    .join("\n");
}

export function CodeExamples() {
  const [active, setActive] = useState(0);

  return (
    <section className="border-x border-t border-border">
      <div className="mx-auto max-w-[1200px] px-6 py-20">
        <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-widest text-accent-bright">
          Code Examples
        </p>
        <h2 className="mb-3 text-3xl font-bold tracking-tight sm:text-4xl">
          A few lines. Full control.
        </h2>
        <p className="mb-10 max-w-md text-[15px] text-muted-foreground">
          Drop-in auto-tracking, manual wrapping, async agents, and nested budgets.
        </p>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {tabs.map((tab, i) => (
            <button
              key={tab.label}
              onClick={() => setActive(i)}
              className={`border-b-2 px-5 py-3 font-mono text-[12px] font-medium transition-colors ${
                active === i
                  ? `${tab.accent} ${tab.borderColor}`
                  : "border-transparent text-muted hover:text-muted-foreground"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Code block */}
        <div className="overflow-x-auto border-x border-b border-border bg-code-bg p-6">
          <pre className="font-mono text-[13px] leading-7">
            <code
              dangerouslySetInnerHTML={{
                __html: highlightPython(codeBlocks[active]),
              }}
            />
          </pre>
        </div>
      </div>
    </section>
  );
}
