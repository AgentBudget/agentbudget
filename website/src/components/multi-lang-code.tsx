"use client";

import { useState } from "react";
import { CodeBlock } from "@/components/code-block";

export type DocLang = "Python" | "Go" | "TypeScript";

interface MultiLangCodeProps {
  python: string;
  go?: string;
  typescript?: string;
  defaultLang?: DocLang;
  pythonLang?: string;
  goLang?: string;
  tsLang?: string;
  /** If true, show all tabs even if some are missing (displays a "coming soon" placeholder) */
  showAll?: boolean;
}

const COMING_SOON = "# Coming soon";

export function MultiLangCode({
  python,
  go,
  typescript,
  defaultLang = "Python",
  pythonLang = "python",
  goLang = "go",
  tsLang = "typescript",
  showAll = false,
}: MultiLangCodeProps) {
  const [active, setActive] = useState<DocLang>(defaultLang);

  const allTabs: { lang: DocLang; code: string; syntaxLang: string; available: boolean }[] = [
    { lang: "Python", code: python, syntaxLang: pythonLang, available: true },
    { lang: "Go", code: go ?? COMING_SOON, syntaxLang: goLang, available: !!go },
    { lang: "TypeScript", code: typescript ?? COMING_SOON, syntaxLang: tsLang, available: !!typescript },
  ];
  const tabs = allTabs.filter((t) => showAll || t.available);

  if (tabs.length === 1) {
    return <CodeBlock lang={tabs[0]!.syntaxLang}>{tabs[0]!.code}</CodeBlock>;
  }

  const activeTab = tabs.find((t) => t.lang === active) ?? tabs[0]!;

  return (
    <div className="mt-4">
      <div className="flex border border-b-0 border-border">
        {tabs.map((tab) => (
          <button
            key={tab.lang}
            onClick={() => setActive(tab.lang)}
            className={`px-3 py-1.5 font-mono text-[11px] transition-colors ${
              activeTab.lang === tab.lang
                ? "bg-code-bg text-accent-bright"
                : "bg-transparent text-muted hover:text-muted-foreground"
            }`}
          >
            {tab.lang}
          </button>
        ))}
      </div>
      <CodeBlock lang={activeTab.syntaxLang}>{activeTab.code}</CodeBlock>
    </div>
  );
}
