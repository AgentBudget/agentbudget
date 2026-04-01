package agentbudget

import (
	"strings"
	"sync"
	"unicode"
)

// priceEntry stores per-token prices (not per-million).
type priceEntry struct {
	inputPerToken  float64
	outputPerToken float64
}

// builtinPricing holds the default pricing table.
// All prices are stored as cost-per-token (i.e. price-per-million / 1_000_000).
var builtinPricing = map[string]priceEntry{
	// OpenAI
	"gpt-4.1":          {5.00 / 1e6, 15.00 / 1e6},
	"gpt-4.1-mini":     {0.40 / 1e6, 1.60 / 1e6},
	"gpt-4.1-nano":     {0.10 / 1e6, 0.40 / 1e6},
	"gpt-4o":           {2.50 / 1e6, 10.00 / 1e6},
	"gpt-4o-mini":      {0.15 / 1e6, 0.60 / 1e6},
	"gpt-4-turbo":      {10.00 / 1e6, 30.00 / 1e6},
	"gpt-4":            {30.00 / 1e6, 60.00 / 1e6},
	"gpt-3.5-turbo":    {0.50 / 1e6, 1.50 / 1e6},
	"o1":               {15.00 / 1e6, 60.00 / 1e6},
	"o1-mini":          {3.00 / 1e6, 12.00 / 1e6},
	"o3":               {10.00 / 1e6, 40.00 / 1e6},
	"o3-pro":           {20.00 / 1e6, 80.00 / 1e6},
	"o4-mini":          {1.10 / 1e6, 4.40 / 1e6},
	// Anthropic
	"claude-opus-4-6":     {15.00 / 1e6, 75.00 / 1e6},
	"claude-opus-4-5":     {15.00 / 1e6, 75.00 / 1e6},
	"claude-sonnet-4-5":   {3.00 / 1e6, 15.00 / 1e6},
	"claude-sonnet-4":     {3.00 / 1e6, 15.00 / 1e6},
	"claude-haiku-4-5":    {0.80 / 1e6, 4.00 / 1e6},
	"claude-3-opus":       {15.00 / 1e6, 75.00 / 1e6},
	"claude-3-sonnet":     {3.00 / 1e6, 15.00 / 1e6},
	"claude-3-haiku":      {0.25 / 1e6, 1.25 / 1e6},
	// Google
	"gemini-2.5-pro":        {1.25 / 1e6, 10.00 / 1e6},
	"gemini-2.5-flash":      {0.075 / 1e6, 0.30 / 1e6},
	"gemini-2.5-flash-lite": {0.01 / 1e6, 0.04 / 1e6},
	"gemini-2.0-flash":      {0.10 / 1e6, 0.40 / 1e6},
	"gemini-1.5-pro":        {1.25 / 1e6, 5.00 / 1e6},
	"gemini-1.5-flash":      {0.075 / 1e6, 0.30 / 1e6},
	// Mistral
	"mistral-large":      {2.00 / 1e6, 6.00 / 1e6},
	"mistral-medium":     {2.70 / 1e6, 8.10 / 1e6},
	"mistral-small":      {0.20 / 1e6, 0.60 / 1e6},
	"codestral":          {0.20 / 1e6, 0.60 / 1e6},
	"open-mistral-nemo":  {0.15 / 1e6, 0.15 / 1e6},
	// Cohere
	"command-r-plus": {3.00 / 1e6, 15.00 / 1e6},
	"command-r":      {0.50 / 1e6, 1.50 / 1e6},
	"command":        {1.00 / 1e6, 2.00 / 1e6},
	"command-light":  {0.30 / 1e6, 0.60 / 1e6},
}

var (
	customPricingMu sync.RWMutex
	customPricing   = map[string]priceEntry{}
)

// RegisterModel adds or overrides pricing for a model at runtime.
// Prices are in USD per million tokens.
func RegisterModel(model string, inputPerMillion, outputPerMillion float64) error {
	if inputPerMillion < 0 || outputPerMillion < 0 {
		return &InvalidBudget{Value: model + ": prices must be non-negative"}
	}
	customPricingMu.Lock()
	defer customPricingMu.Unlock()
	customPricing[model] = priceEntry{
		inputPerToken:  inputPerMillion / 1e6,
		outputPerToken: outputPerMillion / 1e6,
	}
	return nil
}

// getModelPricing returns the pricing entry for a model, or nil if not found.
// Lookup order: custom → builtin → fuzzy (strip date suffix) → OpenRouter prefix.
func getModelPricing(model string) *priceEntry {
	customPricingMu.RLock()
	defer customPricingMu.RUnlock()

	if e, ok := customPricing[model]; ok {
		return &e
	}
	if e, ok := builtinPricing[model]; ok {
		return &e
	}

	// Strip date suffix: "gpt-4o-2025-06-15" → "gpt-4o"
	if base := stripDateSuffix(model); base != model {
		if e, ok := customPricing[base]; ok {
			return &e
		}
		if e, ok := builtinPricing[base]; ok {
			return &e
		}
	}

	// OpenRouter prefix: "openai/gpt-4o" → "gpt-4o"
	if idx := strings.Index(model, "/"); idx != -1 {
		stripped := model[idx+1:]
		return getModelPricingNoLock(stripped)
	}

	return nil
}

// getModelPricingNoLock is used recursively; caller must hold customPricingMu.RLock.
func getModelPricingNoLock(model string) *priceEntry {
	if e, ok := customPricing[model]; ok {
		return &e
	}
	if e, ok := builtinPricing[model]; ok {
		return &e
	}
	if base := stripDateSuffix(model); base != model {
		if e, ok := customPricing[base]; ok {
			return &e
		}
		if e, ok := builtinPricing[base]; ok {
			return &e
		}
	}
	return nil
}

// stripDateSuffix removes trailing date-like segments from model names.
// e.g. "gpt-4o-2025-06-15" → "gpt-4o"
func stripDateSuffix(model string) string {
	parts := strings.Split(model, "-")
	for i := len(parts) - 1; i >= 1; i-- {
		seg := parts[i]
		if len(seg) == 4 && allDigits(seg) {
			return strings.Join(parts[:i], "-")
		}
	}
	return model
}

func allDigits(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// CalculateCost returns the cost in USD for the given model and token counts.
// Returns (cost, true) if the model is found, (0, false) otherwise.
func CalculateCost(model string, inputTokens, outputTokens int64) (float64, bool) {
	p := getModelPricing(model)
	if p == nil {
		return 0, false
	}
	cost := float64(inputTokens)*p.inputPerToken + float64(outputTokens)*p.outputPerToken
	return cost, true
}
