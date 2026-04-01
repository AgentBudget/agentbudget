interface PriceEntry {
  inputPerToken: number;
  outputPerToken: number;
}

// Built-in pricing: cost per token (i.e. price per million / 1_000_000)
const BUILTIN_PRICING: Record<string, PriceEntry> = {
  // OpenAI
  "gpt-4.1":       { inputPerToken: 5.00 / 1e6,  outputPerToken: 15.00 / 1e6 },
  "gpt-4.1-mini":  { inputPerToken: 0.40 / 1e6,  outputPerToken: 1.60 / 1e6 },
  "gpt-4.1-nano":  { inputPerToken: 0.10 / 1e6,  outputPerToken: 0.40 / 1e6 },
  "gpt-4o":        { inputPerToken: 2.50 / 1e6,  outputPerToken: 10.00 / 1e6 },
  "gpt-4o-mini":   { inputPerToken: 0.15 / 1e6,  outputPerToken: 0.60 / 1e6 },
  "gpt-4-turbo":   { inputPerToken: 10.00 / 1e6, outputPerToken: 30.00 / 1e6 },
  "gpt-4":         { inputPerToken: 30.00 / 1e6, outputPerToken: 60.00 / 1e6 },
  "gpt-3.5-turbo": { inputPerToken: 0.50 / 1e6,  outputPerToken: 1.50 / 1e6 },
  "o1":            { inputPerToken: 15.00 / 1e6, outputPerToken: 60.00 / 1e6 },
  "o1-mini":       { inputPerToken: 3.00 / 1e6,  outputPerToken: 12.00 / 1e6 },
  "o3":            { inputPerToken: 10.00 / 1e6, outputPerToken: 40.00 / 1e6 },
  "o3-pro":        { inputPerToken: 20.00 / 1e6, outputPerToken: 80.00 / 1e6 },
  "o4-mini":       { inputPerToken: 1.10 / 1e6,  outputPerToken: 4.40 / 1e6 },
  // Anthropic
  "claude-opus-4-6":   { inputPerToken: 15.00 / 1e6, outputPerToken: 75.00 / 1e6 },
  "claude-opus-4-5":   { inputPerToken: 15.00 / 1e6, outputPerToken: 75.00 / 1e6 },
  "claude-sonnet-4-5": { inputPerToken: 3.00 / 1e6,  outputPerToken: 15.00 / 1e6 },
  "claude-sonnet-4":   { inputPerToken: 3.00 / 1e6,  outputPerToken: 15.00 / 1e6 },
  "claude-haiku-4-5":  { inputPerToken: 0.80 / 1e6,  outputPerToken: 4.00 / 1e6 },
  "claude-3-opus":     { inputPerToken: 15.00 / 1e6, outputPerToken: 75.00 / 1e6 },
  "claude-3-sonnet":   { inputPerToken: 3.00 / 1e6,  outputPerToken: 15.00 / 1e6 },
  "claude-3-haiku":    { inputPerToken: 0.25 / 1e6,  outputPerToken: 1.25 / 1e6 },
  // Google
  "gemini-2.5-pro":        { inputPerToken: 1.25 / 1e6,   outputPerToken: 10.00 / 1e6 },
  "gemini-2.5-flash":      { inputPerToken: 0.075 / 1e6,  outputPerToken: 0.30 / 1e6 },
  "gemini-2.5-flash-lite": { inputPerToken: 0.01 / 1e6,   outputPerToken: 0.04 / 1e6 },
  "gemini-2.0-flash":      { inputPerToken: 0.10 / 1e6,   outputPerToken: 0.40 / 1e6 },
  "gemini-1.5-pro":        { inputPerToken: 1.25 / 1e6,   outputPerToken: 5.00 / 1e6 },
  "gemini-1.5-flash":      { inputPerToken: 0.075 / 1e6,  outputPerToken: 0.30 / 1e6 },
  // Mistral
  "mistral-large":     { inputPerToken: 2.00 / 1e6, outputPerToken: 6.00 / 1e6 },
  "mistral-medium":    { inputPerToken: 2.70 / 1e6, outputPerToken: 8.10 / 1e6 },
  "mistral-small":     { inputPerToken: 0.20 / 1e6, outputPerToken: 0.60 / 1e6 },
  "codestral":         { inputPerToken: 0.20 / 1e6, outputPerToken: 0.60 / 1e6 },
  "open-mistral-nemo": { inputPerToken: 0.15 / 1e6, outputPerToken: 0.15 / 1e6 },
  // Cohere
  "command-r-plus": { inputPerToken: 3.00 / 1e6, outputPerToken: 15.00 / 1e6 },
  "command-r":      { inputPerToken: 0.50 / 1e6, outputPerToken: 1.50 / 1e6 },
  "command":        { inputPerToken: 1.00 / 1e6, outputPerToken: 2.00 / 1e6 },
  "command-light":  { inputPerToken: 0.30 / 1e6, outputPerToken: 0.60 / 1e6 },
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
    if (seg.length === 4 && /^\d{4}$/.test(seg)) {
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
