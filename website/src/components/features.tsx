import Link from "next/link";

const features = [
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
    title: "Drop-in Auto-Tracking",
    description: "One line patches OpenAI and Anthropic SDKs. Your existing code stays completely untouched.",
    href: "/docs#drop-in",
    accent: "green",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="12" y1="1" x2="12" y2="23" />
        <path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" />
      </svg>
    ),
    title: "Dollar-Denominated",
    description: "Tracks real dollars, not tokens. Across LLM calls, tool calls, and external APIs in one balance.",
    href: "/docs#cost-sources",
    accent: "green",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    ),
    title: "Circuit Breaker",
    description: "Soft limit warnings, hard limit enforcement, and automatic loop detection kills runaway sessions.",
    href: "/docs#circuit-breaker",
    accent: "orange",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </svg>
    ),
    title: "Multi-Provider",
    description: "Built-in pricing for 40+ models across OpenAI, Anthropic, Google, Mistral, and Cohere. OpenRouter supported.",
    href: "/docs#supported-models",
    accent: "orange",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="16 3 21 3 21 8" />
        <line x1="4" y1="20" x2="21" y2="3" />
        <polyline points="21 16 21 21 16 21" />
        <line x1="15" y1="15" x2="21" y2="21" />
        <line x1="4" y1="4" x2="9" y2="9" />
      </svg>
    ),
    title: "Async Native",
    description: "Full async/await support with AsyncBudgetSession. Works with async agents and concurrent sessions.",
    href: "/docs#async",
    accent: "pink",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
      </svg>
    ),
    title: "Nested Budgets",
    description: "Parent sessions allocate sub-budgets to child agents. Costs roll up to the parent automatically.",
    href: "/docs#nested-budgets",
    accent: "pink",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    ),
    title: "Framework Integrations",
    description: "LangChain callback handler, CrewAI middleware. Drop into your existing agent framework.",
    href: "/docs#langchain",
    accent: "yellow",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
    title: "Webhooks",
    description: "Stream budget events to Slack, Datadog, or your billing system via HTTP webhooks.",
    href: "/docs#webhooks",
    accent: "yellow",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
        <line x1="4" y1="22" x2="4" y2="15" />
      </svg>
    ),
    title: "Zero Infrastructure",
    description: "No Redis. No servers. No cloud account. Pure Python library that runs in your process.",
    href: "/docs",
    accent: "blue",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
        <polyline points="17 6 23 6 23 12" />
      </svg>
    ),
    title: "Streaming Support",
    description: "stream=True calls are fully tracked. Cost recorded after iterator exhaustion — chunks pass through unchanged.",
    href: "/docs#streaming",
    accent: "green",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    ),
    title: "Finalization Reserve",
    description: "Reserve a budget fraction for the final response. Agents finish their last step instead of being cut off mid-task.",
    href: "/docs#finalization-reserve",
    accent: "orange",
  },
];

const accentGradient = "from-accent to-accent-bright";

export function Features() {
  return (
    <section className="border-x border-t border-border">
      <div className="mx-auto max-w-[1200px] px-6 pt-20 pb-0">
        <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-widest text-accent-bright">
          Features
        </p>
        <h2 className="mb-3 text-3xl font-bold tracking-tight sm:text-4xl">
          Everything you need.{" "}
          <span className="text-muted-foreground">Nothing you don&apos;t.</span>
        </h2>
        <p className="mb-12 max-w-md text-[15px] text-muted-foreground">
          A single library. No infrastructure. Works with your existing code.
        </p>
      </div>

      {/* Grid with border-based cells (Greptile style) */}
      <div className="mx-auto max-w-[1200px]">
        <div className="grid grid-cols-1 border-t border-l border-border sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="group border-b border-r border-border p-7 transition-colors hover:bg-surface"
            >
              <div
                className={`mb-4 inline-flex h-10 w-10 items-center justify-center bg-gradient-to-br text-white ${accentGradient}`}
              >
                {f.icon}
              </div>
              <h3 className="mb-1.5 text-[15px] font-semibold">{f.title}</h3>
              <p className="mb-3 text-[13px] leading-relaxed text-muted-foreground">
                {f.description}
              </p>
              <Link
                href={f.href}
                className="inline-flex items-center gap-1 text-[13px] font-medium text-accent-bright transition-all group-hover:gap-2 hover:no-underline"
              >
                Learn more
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
