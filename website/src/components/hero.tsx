"use client";

import Link from "next/link";
import { useState } from "react";
import { GitHubStars, PyPIDownloads } from "@/components/github-stars";

export function Hero() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText("pip install agentbudget");
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <section className="border-x border-border">
      <div className="mx-auto max-w-[1200px] px-6 pb-16 pt-24 md:pt-32">
        {/* Badge + Stats */}
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <div className="inline-flex items-center gap-2 border border-accent/20 bg-accent/5 px-4 py-1.5 font-mono text-[12px] text-muted-foreground">
            <span
              className="h-1.5 w-1.5 rounded-full bg-accent"
              style={{ animation: "pulse-dot 2s ease-in-out infinite" }}
            />
            OPEN SOURCE &middot; PYTHON SDK
          </div>
          <GitHubStars />
          <PyPIDownloads />
          <a
            href="https://www.producthunt.com/products/agentbudget?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-agentbudget"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              alt="AgentBudget - Real-time budget enforcement for AI agents | Product Hunt"
              width="250"
              height="54"
              src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1095898&theme=light&t=1773282939570"
            />
          </a>
        </div>

        {/* Heading */}
        <h1 className="mb-6 max-w-3xl text-4xl font-bold leading-[1.08] tracking-tight sm:text-5xl md:text-6xl lg:text-[68px]">
          <span className="text-gradient-hero-animated">REAL-TIME</span>
          <br />
          <span className="text-gradient-hero-animated" style={{ animationDelay: "-1.5s" }}>
            COST ENFORCEMENT
          </span>
          <br />
          <span className="text-muted-foreground">FOR AI AGENTS</span>
        </h1>

        {/* Subtitle */}
        <p className="mb-10 max-w-lg text-[16px] leading-relaxed text-muted-foreground">
          Set a hard dollar limit on any AI agent session with one line of code.
          Automatic tracking, circuit breaking, and cost reports across every
          LLM provider.
        </p>

        {/* CTAs */}
        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="https://github.com/sahiljagtap08/agentbudget"
            className="inline-flex items-center gap-2 bg-gradient-accent px-6 py-2.5 text-[14px] font-semibold text-white transition-opacity hover:opacity-90 hover:no-underline"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="4 17 10 11 4 5" />
              <line x1="12" y1="19" x2="20" y2="19" />
            </svg>
            Get Started
          </Link>
          <Link
            href="/docs"
            className="inline-flex items-center gap-2 border border-border-bright px-6 py-2.5 text-[14px] font-medium text-foreground transition-colors hover:bg-surface hover:no-underline"
          >
            Read the Docs
          </Link>
          <Link
            href="/whitepaper"
            className="inline-flex items-center gap-2 border border-border-bright px-6 py-2.5 text-[14px] font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground hover:no-underline"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            Whitepaper
          </Link>
        </div>

        {/* Install command with copy */}
        <button
          onClick={handleCopy}
          className="mt-8 inline-flex cursor-pointer items-center gap-3 border border-border bg-code-bg px-4 py-2.5 font-mono text-[13px] transition-colors hover:border-border-bright"
        >
          <span className="text-muted">$</span>
          <span className="text-accent-bright">pip install</span>
          <span className="text-muted-foreground">agentbudget</span>
          <span className="ml-2 text-muted transition-colors hover:text-muted-foreground">
            {copied ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 6L9 17l-5-5" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
            )}
          </span>
        </button>
      </div>
    </section>
  );
}
