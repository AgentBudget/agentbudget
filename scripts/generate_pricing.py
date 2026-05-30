#!/usr/bin/env python3
"""Generate SDK pricing files from the canonical pricing.json.

Usage:
    python scripts/generate_pricing.py          # generate all
    python scripts/generate_pricing.py --check  # verify files are up to date (for CI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICING_JSON = ROOT / "pricing.json"

# Output paths
PYTHON_OUT = ROOT / "agentbudget" / "pricing.py"
GO_OUT = ROOT / "sdks" / "go" / "pricing.go"
TS_OUT = ROOT / "sdks" / "typescript" / "src" / "pricing.ts"


def load_pricing() -> dict:
    with open(PRICING_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


def fmt_price(price: float) -> str:
    """Format a price for code generation, keeping it readable."""
    # Use up to 3 decimal places, strip trailing zeros
    s = f"{price:.3f}".rstrip("0").rstrip(".")
    return s


# ── Python ──────────────────────────────────────────────────────────────────


PYTHON_HEADER = '''\
"""Model pricing data for LLM cost calculation.

Prices are per token in USD. Generated from pricing.json — do not edit manually.
"""

from __future__ import annotations

from typing import Optional

# Mapping of model name -> (input_price_per_token, output_price_per_token)
MODEL_PRICING: dict[str, tuple[float, float]] = {
'''

PYTHON_FOOTER = '''\
}


_custom_pricing: dict[str, tuple[float, float]] = {}


def register_model(
    model: str,
    input_price_per_million: float,
    output_price_per_million: float,
) -> None:
    """Register custom pricing for a model.

    Use this when a new model launches before AgentBudget ships an update,
    or to override built-in pricing.

    Args:
        model: Model name exactly as passed to the provider SDK.
        input_price_per_million: Cost in USD per 1M input tokens.
        output_price_per_million: Cost in USD per 1M output tokens.

    Example::

        agentbudget.register_model("gpt-5", input_price_per_million=5.00, output_price_per_million=15.00)
    """
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError(
            f"Prices must be non-negative, got input={input_price_per_million}, "
            f"output={output_price_per_million}"
        )
    _custom_pricing[model] = (
        input_price_per_million / 1_000_000,
        output_price_per_million / 1_000_000,
    )


def register_models(models: dict[str, tuple[float, float]]) -> None:
    """Register pricing for multiple models at once.

    Args:
        models: Dict of model name -> (input_price_per_million, output_price_per_million).

    Example::

        agentbudget.register_models({
            "gpt-5": (5.00, 15.00),
            "gpt-5-mini": (0.50, 1.50),
        })
    """
    for model, (inp, out) in models.items():
        register_model(model, inp, out)


def _fuzzy_match(model: str) -> Optional[tuple[float, float]]:
    """Try to match a dated model variant to its base model.

    For example, \'gpt-4o-2025-03-01\' matches \'gpt-4o\'.
    """
    # Try progressively shorter prefixes by stripping trailing segments
    parts = model.rsplit("-", 1)
    while len(parts) == 2:
        prefix = parts[0]
        # Check custom first, then built-in
        if prefix in _custom_pricing:
            return _custom_pricing[prefix]
        if prefix in MODEL_PRICING:
            return MODEL_PRICING[prefix]
        parts = prefix.rsplit("-", 1)
    return None


def get_model_pricing(model: str) -> Optional[tuple[float, float]]:
    """Look up per-token pricing for a model.

    Resolution order:
    1. Custom pricing (registered via register_model)
    2. Built-in pricing table
    3. Fuzzy match (strip date suffixes to find base model)
    4. Retry steps 1-3 after stripping OpenRouter-style "provider/model" prefix

    Returns (input_price_per_token, output_price_per_token) or None if unknown.
    """
    # 1. Custom pricing takes priority
    if model in _custom_pricing:
        return _custom_pricing[model]
    # 2. Built-in table
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # 3. Fuzzy match dated variants
    result = _fuzzy_match(model)
    if result is not None:
        return result
    # 4. Strip OpenRouter-style "provider/model" prefix and retry
    if "/" in model:
        bare = model.split("/", 1)[1]
        if bare in _custom_pricing:
            return _custom_pricing[bare]
        if bare in MODEL_PRICING:
            return MODEL_PRICING[bare]
        return _fuzzy_match(bare)
    return None


def calculate_llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:
    """Calculate the cost of an LLM call in USD.

    Returns None if model pricing is not found.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens * input_price) + (output_tokens * output_price)
'''

PROVIDER_COMMENTS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
    "mistral": "Mistral",
    "cohere": "Cohere",
}


def generate_python(models: dict) -> str:
    lines = [PYTHON_HEADER]
    for provider, provider_models in models.items():
        comment = PROVIDER_COMMENTS.get(provider, provider.title())
        lines.append(f"    # ── {comment} {'─' * (55 - len(comment))}")
        for model, prices in provider_models.items():
            inp = fmt_price(prices["input"])
            out = fmt_price(prices["output"])
            lines.append(
                f'    "{model}": ({inp} / 1_000_000, {out} / 1_000_000),'
            )
    lines.append(PYTHON_FOOTER)
    return "\n".join(lines)


# ── Go ──────────────────────────────────────────────────────────────────────


GO_HEADER = '''\
package agentbudget

// Code generated by scripts/generate_pricing.py from pricing.json — DO NOT EDIT.

import (
\t"strings"
\t"sync"
\t"unicode"
)

// priceEntry stores per-token prices (not per-million).
type priceEntry struct {
\tinputPerToken  float64
\toutputPerToken float64
}

// builtinPricing holds the default pricing table.
// All prices are stored as cost-per-token (i.e. price-per-million / 1_000_000).
var builtinPricing = map[string]priceEntry{
'''

GO_FOOTER = '''\
}

var (
\tcustomPricingMu sync.RWMutex
\tcustomPricing   = map[string]priceEntry{}
)

// RegisterModel adds or overrides pricing for a model at runtime.
// Prices are in USD per million tokens.
func RegisterModel(model string, inputPerMillion, outputPerMillion float64) error {
\tif inputPerMillion < 0 || outputPerMillion < 0 {
\t\treturn &InvalidBudget{Value: model + ": prices must be non-negative"}
\t}
\tcustomPricingMu.Lock()
\tdefer customPricingMu.Unlock()
\tcustomPricing[model] = priceEntry{
\t\tinputPerToken:  inputPerMillion / 1e6,
\t\toutputPerToken: outputPerMillion / 1e6,
\t}
\treturn nil
}

// getModelPricing returns the pricing entry for a model, or nil if not found.
// Lookup order: custom → builtin → fuzzy (strip date suffix) → OpenRouter prefix.
func getModelPricing(model string) *priceEntry {
\tcustomPricingMu.RLock()
\tdefer customPricingMu.RUnlock()

\tif e, ok := customPricing[model]; ok {
\t\treturn &e
\t}
\tif e, ok := builtinPricing[model]; ok {
\t\treturn &e
\t}

\t// Strip date suffix: "gpt-4o-2025-06-15" → "gpt-4o"
\tif base := stripDateSuffix(model); base != model {
\t\tif e, ok := customPricing[base]; ok {
\t\t\treturn &e
\t\t}
\t\tif e, ok := builtinPricing[base]; ok {
\t\t\treturn &e
\t\t}
\t}

\t// OpenRouter prefix: "openai/gpt-4o" → "gpt-4o"
\tif idx := strings.Index(model, "/"); idx != -1 {
\t\tstripped := model[idx+1:]
\t\treturn getModelPricingNoLock(stripped)
\t}

\treturn nil
}

// getModelPricingNoLock is used recursively; caller must hold customPricingMu.RLock.
func getModelPricingNoLock(model string) *priceEntry {
\tif e, ok := customPricing[model]; ok {
\t\treturn &e
\t}
\tif e, ok := builtinPricing[model]; ok {
\t\treturn &e
\t}
\tif base := stripDateSuffix(model); base != model {
\t\tif e, ok := customPricing[base]; ok {
\t\t\treturn &e
\t\t}
\t\tif e, ok := builtinPricing[base]; ok {
\t\t\treturn &e
\t\t}
\t}
\treturn nil
}

// stripDateSuffix removes trailing date-like segments from model names.
// e.g. "gpt-4o-2025-06-15" → "gpt-4o"
func stripDateSuffix(model string) string {
\tparts := strings.Split(model, "-")
\tfor i := len(parts) - 1; i >= 1; i-- {
\t\tseg := parts[i]
\t\tif len(seg) == 4 && allDigits(seg) {
\t\t\treturn strings.Join(parts[:i], "-")
\t\t}
\t}
\treturn model
}

func allDigits(s string) bool {
\tfor _, r := range s {
\t\tif !unicode.IsDigit(r) {
\t\t\treturn false
\t\t}
\t}
\treturn true
}

// CalculateCost returns the cost in USD for the given model and token counts.
// Returns (cost, true) if the model is found, (0, false) otherwise.
func CalculateCost(model string, inputTokens, outputTokens int64) (float64, bool) {
\tp := getModelPricing(model)
\tif p == nil {
\t\treturn 0, false
\t}
\tcost := float64(inputTokens)*p.inputPerToken + float64(outputTokens)*p.outputPerToken
\treturn cost, true
}
'''


def generate_go(models: dict) -> str:
    lines = [GO_HEADER]
    for provider, provider_models in models.items():
        comment = PROVIDER_COMMENTS.get(provider, provider.title())
        lines.append(f"\t// {comment}")
        entries = list(provider_models.items())
        for i, (model, prices) in enumerate(entries):
            inp = fmt_price(prices["input"])
            out = fmt_price(prices["output"])
            # Calculate padding for alignment
            key = f'"{model}":'
            padding = " " * max(1, 28 - len(key))
            lines.append(f"\t{key}{padding}{{{inp} / 1e6, {out} / 1e6}},")
        # Blank line between providers (except after last)
    lines.append(GO_FOOTER)
    return "\n".join(lines)


# ── TypeScript ──────────────────────────────────────────────────────────────


TS_HEADER = '''\
// Code generated by scripts/generate_pricing.py from pricing.json — DO NOT EDIT.

interface PriceEntry {
  inputPerToken: number;
  outputPerToken: number;
}

// Built-in pricing: cost per token (i.e. price per million / 1_000_000)
const BUILTIN_PRICING: Record<string, PriceEntry> = {
'''

TS_FOOTER = '''\
};

const customPricing: Record<string, PriceEntry> = {};

/** Register custom pricing for a model. Prices are per million tokens. */
export function registerModel(
  model: string,
  inputPricePerMillion: number,
  outputPricePerMillion: number
): void {
  if (inputPricePerMillion < 0 || outputPricePerMillion < 0) {
    throw new Error(`Prices must be non-negative for model "${model}"`);
  }
  customPricing[model] = {
    inputPerToken: inputPricePerMillion / 1e6,
    outputPerToken: outputPricePerMillion / 1e6,
  };
}

/** Strip trailing date suffix from model names (e.g. "gpt-4o-2025-06-15" → "gpt-4o"). */
function stripDateSuffix(model: string): string {
  const parts = model.split("-");
  for (let i = parts.length - 1; i >= 1; i--) {
    const seg = parts[i] ?? "";
    if (seg.length === 4 && /^\\d{4}$/.test(seg)) {
      return parts.slice(0, i).join("-");
    }
  }
  return model;
}

function lookupPricing(model: string): PriceEntry | undefined {
  return customPricing[model] ?? BUILTIN_PRICING[model];
}

function getModelPricing(model: string): PriceEntry | undefined {
  // Exact match
  const exact = lookupPricing(model);
  if (exact) return exact;

  // Fuzzy: strip date suffix
  const base = stripDateSuffix(model);
  if (base !== model) {
    const fuzzy = lookupPricing(base);
    if (fuzzy) return fuzzy;
  }

  // OpenRouter prefix: "openai/gpt-4o" → "gpt-4o"
  const slashIdx = model.indexOf("/");
  if (slashIdx !== -1) {
    return getModelPricing(model.slice(slashIdx + 1));
  }

  return undefined;
}

/** Calculate USD cost for an LLM call. Returns undefined if model is not found. */
export function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number
): number | undefined {
  const pricing = getModelPricing(model);
  if (!pricing) return undefined;
  return inputTokens * pricing.inputPerToken + outputTokens * pricing.outputPerToken;
}
'''


def generate_ts(models: dict) -> str:
    lines = [TS_HEADER]
    for provider, provider_models in models.items():
        comment = PROVIDER_COMMENTS.get(provider, provider.title())
        lines.append(f"  // {comment}")
        for model, prices in provider_models.items():
            inp = fmt_price(prices["input"])
            out = fmt_price(prices["output"])
            # Calculate padding for alignment
            key = f'"{model}":'
            padding = " " * max(1, 28 - len(key))
            lines.append(
                f"  {key}{padding}{{ inputPerToken: {inp} / 1e6,"
                f"  outputPerToken: {out} / 1e6 }},"
            )
    lines.append(TS_FOOTER)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    check_mode = "--check" in sys.argv
    models = load_pricing()

    generators = [
        ("Python", PYTHON_OUT, generate_python),
        ("Go", GO_OUT, generate_go),
        ("TypeScript", TS_OUT, generate_ts),
    ]

    any_diff = False
    for name, path, gen_fn in generators:
        expected = gen_fn(models)
        if check_mode:
            try:
                actual = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"FAIL: {path.relative_to(ROOT)} does not exist")
                any_diff = True
                continue
            if actual != expected:
                print(f"FAIL: {path.relative_to(ROOT)} is out of sync with pricing.json")
                any_diff = True
            else:
                print(f"  OK: {path.relative_to(ROOT)}")
        else:
            path.write_text(expected, encoding="utf-8", newline="\n")
            print(f"Generated {path.relative_to(ROOT)}")

    if check_mode and any_diff:
        print(
            "\nPricing files are out of sync. "
            "Run `python scripts/generate_pricing.py` to regenerate."
        )
        return 1

    if not check_mode:
        print("\nAll pricing files generated from pricing.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
