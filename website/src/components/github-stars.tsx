"use client";

import { useEffect, useState } from "react";

export function GitHubStars() {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    fetch("https://api.github.com/repos/AgentBudget/agentbudget")
      .then((res) => res.json())
      .then((data) => {
        if (typeof data.stargazers_count === "number") {
          setStars(data.stargazers_count);
        }
      })
      .catch(() => {});
  }, []);

  return (
    <a
      href="https://github.com/AgentBudget/agentbudget"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 border border-border bg-surface px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:border-border-bright hover:text-foreground hover:no-underline"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
      </svg>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" className="text-accent-bright">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
      </svg>
      {stars !== null ? (
        <span className="font-mono">{stars}</span>
      ) : (
        <span className="font-mono text-muted">--</span>
      )}
    </a>
  );
}

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

const DownloadIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
  </svg>
);

export function PyPIDownloads() {
  const [downloads, setDownloads] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/pypi-stats")
      .then((res) => res.json())
      .then((data) => {
        if (typeof data?.downloads === "number") setDownloads(data.downloads);
      })
      .catch(() => {});
  }, []);

  return (
    <a
      href="https://pepy.tech/projects/agentbudget"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 border border-border bg-surface px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:border-border-bright hover:text-foreground hover:no-underline"
    >
      <DownloadIcon />
      <span className="font-mono text-[10px] text-muted">py</span>
      <span className="font-mono">{downloads !== null ? fmt(downloads) : "--"}</span>
      <span className="text-muted">installs</span>
    </a>
  );
}

export function NpmDownloads() {
  const [downloads, setDownloads] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/npm-stats")
      .then((res) => res.json())
      .then((data) => {
        if (typeof data?.downloads === "number") setDownloads(data.downloads);
      })
      .catch(() => {});
  }, []);

  return (
    <a
      href="https://www.npmjs.com/package/@agentbudget/agentbudget"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 border border-border bg-surface px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:border-border-bright hover:text-foreground hover:no-underline"
    >
      <DownloadIcon />
      <span className="font-mono text-[10px] text-muted">npm</span>
      <span className="font-mono">{downloads !== null ? fmt(downloads) : "--"}</span>
      <span className="text-muted">installs</span>
    </a>
  );
}

export function GoClones() {
  const [clones, setClones] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/go-stats")
      .then((r) => r.json())
      .then((d) => { if (typeof d?.clones === "number") setClones(d.clones); })
      .catch(() => {});
  }, []);

  return (
    <a
      href="https://pkg.go.dev/github.com/AgentBudget/agentbudget/sdks/go"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 border border-border bg-surface px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:border-border-bright hover:text-foreground hover:no-underline"
    >
      <DownloadIcon />
      <span className="font-mono text-[10px] text-muted">go</span>
      <span className="font-mono">{clones !== null ? fmt(clones) : "--"}</span>
      <span className="text-muted">clones</span>
    </a>
  );
}

export function TotalInstalls() {
  const [pypi, setPypi] = useState<number | null>(null);
  const [npm, setNpm] = useState<number | null>(null);
  const [go, setGo] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/pypi-stats")
      .then((r) => r.json())
      .then((d) => { if (typeof d?.downloads === "number") setPypi(d.downloads); })
      .catch(() => {});
    fetch("/api/npm-stats")
      .then((r) => r.json())
      .then((d) => { if (typeof d?.downloads === "number") setNpm(d.downloads); })
      .catch(() => {});
    fetch("/api/go-stats")
      .then((r) => r.json())
      .then((d) => { if (typeof d?.clones === "number") setGo(d.clones); })
      .catch(() => {});
  }, []);

  const total = pypi !== null || npm !== null || go !== null
    ? (pypi ?? 0) + (npm ?? 0) + (go ?? 0)
    : null;

  return (
    <a
      href="https://pepy.tech/projects/agentbudget"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 border border-border bg-surface px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:border-border-bright hover:text-foreground hover:no-underline"
    >
      <DownloadIcon />
      <span className="font-mono">{total !== null ? fmt(total) : "--"}</span>
      <span className="text-muted">total installs</span>
    </a>
  );
}
