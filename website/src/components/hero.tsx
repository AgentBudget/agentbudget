"use client";

import { GitHubStars, PyPIDownloads, NpmDownloads, GoClones } from "@/components/github-stars";
import Link from "next/link";
import { useState } from "react";

const INSTALL_COMMANDS = [
  { lang: "Python", cmd: "pip install agentbudget" },
  { lang: "Go", cmd: "go get github.com/AgentBudget/agentbudget/sdks/go" },
  { lang: "TypeScript", cmd: "npm install @agentbudget/agentbudget" },
] as const;

const LangIcon = ({ lang }: { lang: Lang }) => {
  if (lang === "Python") return (
    <svg width="13" height="13" viewBox="0 0 256 255" xmlns="http://www.w3.org/2000/svg">
      <path d="M126.916.072c-64.832 0-60.784 28.115-60.784 28.115l.072 29.128h61.868v8.745H41.631S.145 61.355.145 126.77c0 65.417 36.21 63.097 36.21 63.097h21.61v-30.356s-1.165-36.21 35.632-36.21h61.362s34.475.557 34.475-33.319V33.97S194.67.072 126.916.072zM92.802 19.66a11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13 11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.13z" fill="#387EB8"/>
      <path d="M128.757 254.126c64.832 0 60.784-28.115 60.784-28.115l-.072-29.127H127.6v-8.745h86.441s41.486 4.705 41.486-60.712c0-65.416-36.21-63.096-36.21-63.096h-21.61v30.355s1.165 36.21-35.632 36.21h-61.362s-34.475-.557-34.475 33.32v56.013s-5.235 33.897 62.518 33.897zm34.114-19.586a11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.131 11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13z" fill="#FFE052"/>
    </svg>
  );
  if (lang === "Go") return (
    <svg width="15" height="11" viewBox="0 0 207 78" xmlns="http://www.w3.org/2000/svg">
      <path d="M16.2 24.1c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h35.7c.4 0 .5.3.3.6l-1.7 2.6c-.2.3-.7.6-1 .6l-36.2-.1zM1 33.3c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h45.6c.4 0 .6.3.5.6l-.8 2.4c-.1.4-.5.6-.9.6L1 33.3zM25.3 42.5c-.4 0-.5-.3-.3-.6l1.4-2.5c.2-.3.6-.6 1-.6h20c.4 0 .6.3.6.7l-.2 2.4c0 .4-.4.7-.7.7l-21.8-.1z" fill="#00ACD7"/>
      <path d="M153.1 22.4c-6.3 1.6-10.6 2.8-16.8 4.4-1.5.4-1.6.5-2.9-1-1.5-1.7-2.6-2.8-4.7-3.8-6.3-3.1-12.4-2.2-18.1 1.5-6.8 4.4-10.3 10.9-10.2 19 .1 8 5.6 14.6 13.5 15.7 6.8.9 12.5-1.5 17-6.6.9-1.1 1.7-2.3 2.7-3.7H117c-2.1 0-2.6-1.3-1.9-3 1.3-3.1 3.7-8.3 5.1-10.9.3-.6 1-1.6 2.5-1.6h36.4c-.2 2.7-.2 5.4-.6 8.1-1.1 7.2-3.8 13.8-8.2 19.6-7.2 9.5-16.6 15.4-28.5 17-9.8 1.3-18.9-.6-26.9-6.6-7.4-5.6-11.6-13-12.7-22.2-1.3-10.9 1.9-20.7 8.5-29.3C97.7 9.8 107 4 118.2 1.7c9.1-1.9 17.9-1 26 4.1 5.4 3.3 9.3 7.9 11.8 13.8.5.9.1 1.4-2.9 2.8z" fill="#00ACD7"/>
      <path d="M186.2 64.6c-9.1-.2-17.4-2.8-24.4-8.8-5.9-5.1-9.6-11.6-10.8-19.3-1.8-11.3 1.3-21.3 8.1-30.2 7.3-9.6 16.1-14.6 28-16.7 10.2-1.8 19.8-.8 28.5 5.1 7.9 5.4 12.8 12.7 14.1 22.3 1.7 13.5-2.2 24.5-11.5 33.4-6.6 6.3-14.7 10.1-23.8 11.8-2.7.5-5.4.6-8.2.4zm23.2-34.4c-.1-1.3-.1-2.3-.3-3.3-1.8-9.9-10.9-15.5-20.4-13.3-9.3 2.1-15.3 8-17.5 17.4-1.8 7.8 2 15.7 9.2 18.9 5.5 2.4 11 2.1 16.3-.6 7.9-4.1 12.2-10.5 12.7-19.1z" fill="#00ACD7"/>
    </svg>
  );
  // TypeScript
  return (
    <svg width="13" height="13" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
      <rect width="400" height="400" rx="50" fill="#3178C6"/>
      <path d="M87.7 200.7V217h52v148h36.9V217h52v-16c0-9 0-16.3-.4-16.5-.3-.3-31.7-.4-70-.4l-69.7.3v16.3zM321.4 184c10.2 2.4 18 7 25 14.3 3.7 4 9.2 11 9.6 12.8.1.5-17.3 12.3-27.8 18.8-.4.3-2-1.4-3.6-4-5.2-7.4-10.5-10.6-18.8-11.2-12.1-.8-20 5.5-20 16 0 3.2.5 5 1.8 7.6 2.7 5.5 7.7 8.8 23.2 15.6 28.6 12.3 40.9 20.4 48.5 32 8.5 13 10.4 33.4 4.7 48.7-6.4 16.7-22.2 28-44.3 31.6-6.8 1.2-23 1-30.5-.3-16-3-31.3-11-40.7-21.3-3.7-4-10.8-14.7-10.4-15.4l3.8-2.4 15-8.7 11.3-6.6 2.6 3.5c3.3 5.2 10.7 12.2 15.2 14.6 13 6.7 30.4 5.8 39.1-2 3.7-3.4 5.3-6.9 5.3-12 0-4.6-.7-6.7-3-10.2-3.2-4.4-9.6-8-27.6-16-20.7-8.9-29.5-14.4-37.7-23-4.7-5.2-9-13.3-10.8-20-1.5-5.8-1.9-20.4-.6-26.1 4.9-23 23.1-38.5 47.9-41 8.1-.7 27 .3 34.9 2.1z" fill="#fff"/>
    </svg>
  );
};

type Lang = (typeof INSTALL_COMMANDS)[number]["lang"];

export function Hero() {
  const [activeLang, setActiveLang] = useState<Lang>("Python");
  const [copied, setCopied] = useState(false);

  const activeCmd = INSTALL_COMMANDS.find((c) => c.lang === activeLang)!.cmd;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCmd);
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
            OPEN SOURCE &middot; PYTHON &middot; GO &middot; TYPESCRIPT
          </div>
          <GitHubStars />
          <PyPIDownloads />
          <NpmDownloads />
          <GoClones />
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
            href="https://github.com/AgentBudget/agentbudget"
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

        {/* Install command with language tab switcher */}
        <div className="mt-8 inline-flex flex-col">
          {/* Tab row */}
          <div className="flex border border-b-0 border-border">
            {INSTALL_COMMANDS.map(({ lang }) => (
              <button
                key={lang}
                onClick={() => setActiveLang(lang)}
                className={`inline-flex items-center gap-1.5 px-4 py-1.5 font-mono text-[11px] transition-colors ${
                  activeLang === lang
                    ? "bg-code-bg text-accent-bright"
                    : "bg-transparent text-muted hover:text-muted-foreground"
                }`}
              >
                <LangIcon lang={lang} />
                {lang}
              </button>
            ))}
          </div>
          {/* Command row */}
          <button
            onClick={handleCopy}
            className="inline-flex cursor-pointer items-center gap-3 border border-border bg-code-bg px-4 py-2.5 font-mono text-[13px] transition-colors hover:border-border-bright"
          >
            <span className="text-muted">$</span>
            <span className="text-accent-bright">
              {activeLang === "Python" ? "pip install" : activeLang === "Go" ? "go get" : "npm install"}
            </span>
            <span className="text-muted-foreground">
              {activeLang === "Python"
                ? "agentbudget"
                : activeLang === "Go"
                ? "github.com/AgentBudget/agentbudget/sdks/go"
                : "@agentbudget/agentbudget"}
            </span>
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
      </div>
    </section>
  );
}
