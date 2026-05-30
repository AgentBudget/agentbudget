import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { PostHogProvider } from "@/components/posthog-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "AgentBudget - Real-time cost enforcement for AI agents",
    template: "%s | AgentBudget",
  },
  description:
    "Open-source Python SDK that puts a hard dollar limit on any AI agent session. One line to set a budget. Zero infrastructure to manage.",
  metadataBase: new URL("https://agentbudget.dev"),
  keywords: [
    "agentbudget",
    "AgentBudget",
    "AI agent budget",
    "LLM cost tracking",
    "AI cost control",
    "OpenAI budget limit",
    "agent cost enforcement",
    "LLM spend limit",
    "AI agent cost management",
    "python AI budget",
    "GPT cost tracker",
    "anthropic budget",
    "runaway agent prevention",
    "AI agent loop detection",
    "pip install agentbudget",
    "AI agent cost tracking Python",
  ],
  authors: [{ name: "Sahil Jagtap" }],
  creator: "Sahil Jagtap",
  openGraph: {
    title: "AgentBudget - The ulimit for AI agents",
    description:
      "Real-time cost enforcement for AI agent sessions. Stop runaway LLM spend with one line of code.",
    url: "https://agentbudget.dev",
    siteName: "AgentBudget",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "AgentBudget - The ulimit for AI agents",
    description:
      "Real-time cost enforcement for AI agent sessions. Stop runaway LLM spend with one line of code.",
    creator: "@twtofsahil",
  },
  alternates: {
    canonical: "https://agentbudget.dev",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "AgentBudget",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Any",
  description:
    "Open-source Python SDK that puts a hard dollar limit on any AI agent session. Real-time cost enforcement with circuit breaking and loop detection.",
  url: "https://agentbudget.dev",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  author: {
    "@type": "Person",
    name: "Sahil Jagtap",
  },
  programmingLanguage: "Python",
  softwareRequirements: "Python 3.9+",
  license: "https://opensource.org/licenses/Apache-2.0",
  codeRepository: "https://github.com/AgentBudget/agentbudget",
};

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: [
    {
      "@type": "Question",
      name: "What is AgentBudget?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "AgentBudget is an open-source Python SDK that enforces real-time cost limits on AI agent sessions. With one line of code — agentbudget.init('$5.00') — you set a hard dollar cap on any agent using OpenAI, Anthropic, or Gemini. When the budget is hit, the agent stops automatically.",
      },
    },
    {
      "@type": "Question",
      name: "How do I install AgentBudget?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Install AgentBudget via pip: pip install agentbudget. It requires Python 3.9+ and has no external dependencies.",
      },
    },
    {
      "@type": "Question",
      name: "What AI models does AgentBudget support?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "AgentBudget supports 60+ models including all OpenAI models (GPT-4o, GPT-4.1, o1, o3, o4-mini), all Anthropic Claude models (Opus, Sonnet, Haiku), Google Gemini (1.5, 2.0, 2.5), Mistral, and Cohere. Custom model pricing can be registered at runtime.",
      },
    },
    {
      "@type": "Question",
      name: "How does AgentBudget stop runaway AI agents?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "AgentBudget uses three layers of protection: a hard dollar limit that raises BudgetExhausted when reached, a soft limit callback at 90% to allow graceful shutdown, and a circuit breaker that detects repeated API calls in a loop and raises LoopDetected before the budget is drained.",
      },
    },
    {
      "@type": "Question",
      name: "Does AgentBudget work with LangChain, LangGraph, and CrewAI?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Yes. AgentBudget has native integrations for LangChain and LangGraph via LangChainBudgetCallback (which tracks chat-model usage and tool costs), CrewAI via CrewAIBudgetMiddleware, and AutoGen. Install with pip install agentbudget[langchain].",
      },
    },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <PostHogProvider>{children}</PostHogProvider>
      </body>
    </html>
  );
}
