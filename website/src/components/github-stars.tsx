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

export const PythonLogo = () => (
  <svg width="13" height="13" viewBox="0 0 256 255" xmlns="http://www.w3.org/2000/svg">
    <path d="M126.916.072c-64.832 0-60.784 28.115-60.784 28.115l.072 29.128h61.868v8.745H41.631S.145 61.355.145 126.77c0 65.417 36.21 63.097 36.21 63.097h21.61v-30.356s-1.165-36.21 35.632-36.21h61.362s34.475.557 34.475-33.319V33.97S194.67.072 126.916.072zM92.802 19.66a11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13 11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.13z" fill="#387EB8"/>
    <path d="M128.757 254.126c64.832 0 60.784-28.115 60.784-28.115l-.072-29.127H127.6v-8.745h86.441s41.486 4.705 41.486-60.712c0-65.416-36.21-63.096-36.21-63.096h-21.61v30.355s1.165 36.21-35.632 36.21h-61.362s-34.475-.557-34.475 33.32v56.013s-5.235 33.897 62.518 33.897zm34.114-19.586a11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.131 11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13z" fill="#FFE052"/>
  </svg>
);

export const GoLogo = () => (
  <svg width="16" height="12" viewBox="0 0 207 78" xmlns="http://www.w3.org/2000/svg">
    <path d="M16.2 24.1c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h35.7c.4 0 .5.3.3.6l-1.7 2.6c-.2.3-.7.6-1 .6l-36.2-.1zM1 33.3c-.4 0-.5-.2-.3-.5l2.1-2.7c.2-.3.7-.5 1.1-.5h45.6c.4 0 .6.3.5.6l-.8 2.4c-.1.4-.5.6-.9.6L1 33.3zM25.3 42.5c-.4 0-.5-.3-.3-.6l1.4-2.5c.2-.3.6-.6 1-.6h20c.4 0 .6.3.6.7l-.2 2.4c0 .4-.4.7-.7.7l-21.8-.1zM153.1 22.4c-6.3 1.6-10.6 2.8-16.8 4.4-1.5.4-1.6.5-2.9-1-1.5-1.7-2.6-2.8-4.7-3.8-6.3-3.1-12.4-2.2-18.1 1.5-6.8 4.4-10.3 10.9-10.2 19 .1 8 5.6 14.6 13.5 15.7 6.8.9 12.5-1.5 17-6.6.9-1.1 1.7-2.3 2.7-3.7H117c-2.1 0-2.6-1.3-1.9-3 1.3-3.1 3.7-8.3 5.1-10.9.3-.6 1-1.6 2.5-1.6h36.4c-.2 2.7-.2 5.4-.6 8.1-1.1 7.2-3.8 13.8-8.2 19.6-7.2 9.5-16.6 15.4-28.5 17-9.8 1.3-18.9-.6-26.9-6.6-7.4-5.6-11.6-13-12.7-22.2-1.3-10.9 1.9-20.7 8.5-29.3C97.7 9.8 107 4 118.2 1.7c9.1-1.9 17.9-1 26 4.1 5.4 3.3 9.3 7.9 11.8 13.8.5.9.1 1.4-2.9 2.8zM186.2 64.6c-9.1-.2-17.4-2.8-24.4-8.8-5.9-5.1-9.6-11.6-10.8-19.3-1.8-11.3 1.3-21.3 8.1-30.2 7.3-9.6 16.1-14.6 28-16.7 10.2-1.8 19.8-.8 28.5 5.1 7.9 5.4 12.8 12.7 14.1 22.3 1.7 13.5-2.2 24.5-11.5 33.4-6.6 6.3-14.7 10.1-23.8 11.8-2.7.5-5.4.6-8.2.4zm23.2-34.4c-.1-1.3-.1-2.3-.3-3.3-1.8-9.9-10.9-15.5-20.4-13.3-9.3 2.1-15.3 8-17.5 17.4-1.8 7.8 2 15.7 9.2 18.9 5.5 2.4 11 2.1 16.3-.6 7.9-4.1 12.2-10.5 12.7-19.1z" fill="#00ACD7"/>
  </svg>
);

export const TSLogo = () => (
  <svg width="13" height="13" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
    <rect width="400" height="400" rx="50" fill="#3178C6"/>
    <path d="M87.7 200.7V217h52v148h36.9V217h52v-16c0-9 0-16.3-.4-16.5-.3-.3-31.7-.4-70-.4l-69.7.3v16.3zM321.4 184c10.2 2.4 18 7 25 14.3 3.7 4 9.2 11 9.6 12.8.1.5-17.3 12.3-27.8 18.8-.4.3-2-1.4-3.6-4-5.2-7.4-10.5-10.6-18.8-11.2-12.1-.8-20 5.5-20 16 0 3.2.5 5 1.8 7.6 2.7 5.5 7.7 8.8 23.2 15.6 28.6 12.3 40.9 20.4 48.5 32 8.5 13 10.4 33.4 4.7 48.7-6.4 16.7-22.2 28-44.3 31.6-6.8 1.2-23 1-30.5-.3-16-3-31.3-11-40.7-21.3-3.7-4-10.8-14.7-10.4-15.4l3.8-2.4 15-8.7 11.3-6.6 2.6 3.5c3.3 5.2 10.7 12.2 15.2 14.6 13 6.7 30.4 5.8 39.1-2 3.7-3.4 5.3-6.9 5.3-12 0-4.6-.7-6.7-3-10.2-3.2-4.4-9.6-8-27.6-16-20.7-8.9-29.5-14.4-37.7-23-4.7-5.2-9-13.3-10.8-20-1.5-5.8-1.9-20.4-.6-26.1 4.9-23 23.1-38.5 47.9-41 8.1-.7 27 .3 34.9 2.1z" fill="#fff"/>
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
      <PythonLogo />
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
      <TSLogo />
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
      <GoLogo />
      <span className="font-mono">{clones !== null ? fmt(clones) : "--"}</span>
      <span className="text-muted">installs</span>
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
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
      </svg>
      <span className="font-mono">{total !== null ? fmt(total) : "--"}</span>
      <span className="text-muted">total installs</span>
    </a>
  );
}
