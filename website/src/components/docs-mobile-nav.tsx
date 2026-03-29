"use client";

import { useState } from "react";

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

export function DocsMobileNav() {
  const [mobileNav, setMobileNav] = useState(false);

  return (
    <>
      <button
        className="fixed bottom-6 right-6 z-50 flex h-10 w-10 items-center justify-center border border-border bg-surface text-muted-foreground md:hidden"
        onClick={() => setMobileNav(!mobileNav)}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {mobileNav && (
        <div className="fixed inset-0 z-40 bg-background/90 backdrop-blur-sm md:hidden" onClick={() => setMobileNav(false)}>
          <div className="mt-14 w-[260px] border-r border-border bg-background p-6" onClick={(e) => e.stopPropagation()}>
            {sidebarSections.map((section) => (
              <div key={section.title} className="mb-5">
                <h4 className="mb-2 font-mono text-[10px] font-semibold uppercase tracking-widest text-muted">
                  {section.title}
                </h4>
                {section.items.map((item) => (
                  <a
                    key={item.id}
                    href={`#${item.id}`}
                    className="block py-1.5 text-[13px] text-muted-foreground hover:text-foreground"
                    onClick={() => setMobileNav(false)}
                  >
                    {item.label}
                  </a>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
