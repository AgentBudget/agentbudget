import Link from "next/link";

const columns = [
  {
    title: "Product",
    links: [
      { label: "Documentation", href: "/docs" },
      { label: "PyPI Package", href: "https://pypi.org/project/agentbudget/" },
      { label: "PyPI Stats", href: "https://pypistats.org/packages/agentbudget" },
      { label: "Changelog", href: "https://github.com/AgentBudget/agentbudget/releases" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Getting Started", href: "/docs#installation" },
      { label: "API Reference", href: "/docs#api-reference" },
      { label: "Supported Models", href: "/docs#supported-models" },
      { label: "Whitepaper", href: "/whitepaper" },
      { label: "Published Paper (DOI)", href: "https://doi.org/10.5281/zenodo.18734145" },
    ],
  },
  {
    title: "Community",
    links: [
      { label: "GitHub", href: "https://github.com/AgentBudget/agentbudget" },
      { label: "Issues", href: "https://github.com/AgentBudget/agentbudget/issues" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-x border-t border-border">
      <div className="mx-auto max-w-[1200px] px-6 py-16">
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          {/* Brand column */}
          <div>
            <span className="text-[15px] font-bold tracking-tight">
              Agent<span className="text-accent-bright">Budget</span>
            </span>
            <p className="mt-3 text-[13px] leading-relaxed text-muted">
              Real-time cost enforcement
              <br />
              for AI agent sessions.
            </p>
            <div className="mt-4 flex gap-3">
              <Link
                href="https://github.com/AgentBudget/agentbudget"
                className="text-muted transition-colors hover:text-muted-foreground"
                aria-label="GitHub"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
                </svg>
              </Link>
              <Link
                href="https://twitter.com/agentbudgetsdk"
                className="text-muted transition-colors hover:text-muted-foreground"
                aria-label="Twitter / X"
                target="_blank"
                rel="noopener noreferrer"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.746l7.73-8.835L1.254 2.25H8.08l4.253 5.622 5.911-5.622zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </Link>
              <Link
                href="https://linkedin.com/company/agentbudget"
                className="text-muted transition-colors hover:text-muted-foreground"
                aria-label="LinkedIn"
                target="_blank"
                rel="noopener noreferrer"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </Link>
            </div>
          </div>

          {/* Link columns */}
          {columns.map((col) => (
            <div key={col.title}>
              <h4 className="mb-3 text-[12px] font-semibold uppercase tracking-wider text-muted-foreground">
                {col.title}
              </h4>
              <ul className="space-y-2">
                {col.links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-[13px] text-muted transition-colors hover:text-muted-foreground hover:no-underline"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 border-t border-border pt-6">
          <p className="text-[12px] text-muted">
            &copy; {new Date().getFullYear()} AgentBudget &middot; Apache 2.0 &middot; Built by{" "}
            <Link href="https://github.com/AgentBudget" className="text-muted-foreground hover:text-foreground">
              Sahil Jagtap
            </Link>
          </p>
        </div>
      </div>
    </footer>
  );
}
