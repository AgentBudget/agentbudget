"use client";

import Link from "next/link";
import { useState } from "react";
import { GitHubStars } from "@/components/github-stars";

export function Nav() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 text-foreground no-underline">
          <span className="text-[17px] font-bold tracking-tight">
            Agent<span className="text-accent-bright">Budget</span>
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden items-center gap-7 md:flex">
          <span className="inline-flex items-center gap-1.5 border border-accent/20 bg-accent/10 px-2.5 py-0.5 font-mono text-[11px] font-medium text-accent-bright">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" style={{ animation: "pulse-dot 2s ease-in-out infinite" }} />
            v0.4.0
          </span>
          <Link href="/docs" className="text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground hover:no-underline">
            Docs
          </Link>
          <Link href="https://pypi.org/project/agentbudget/" className="text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground hover:no-underline">
            PyPI
          </Link>
          <Link href="/whitepaper" className="text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground hover:no-underline">
            Whitepaper
          </Link>
          <GitHubStars />
          <Link
            href="https://github.com/AgentBudget/agentbudget"
            className="inline-flex items-center gap-1.5 bg-gradient-accent px-3.5 py-1.5 text-[13px] font-semibold text-white transition-opacity hover:opacity-90 hover:no-underline"
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            Get Started
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          className="flex items-center justify-center md:hidden text-muted-foreground"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileOpen ? (
              <path d="M18 6L6 18M6 6l12 12" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="border-t border-border px-6 py-4 md:hidden">
          <div className="flex flex-col gap-3">
            <Link href="/docs" className="text-sm text-muted-foreground hover:text-foreground" onClick={() => setMobileOpen(false)}>Docs</Link>
            <Link href="https://pypi.org/project/agentbudget/" className="text-sm text-muted-foreground hover:text-foreground">PyPI</Link>
            <Link href="/whitepaper" className="text-sm text-muted-foreground hover:text-foreground" onClick={() => setMobileOpen(false)}>Whitepaper</Link>
            <Link href="https://github.com/AgentBudget/agentbudget" className="text-sm text-muted-foreground hover:text-foreground">GitHub</Link>
          </div>
        </div>
      )}
    </nav>
  );
}
