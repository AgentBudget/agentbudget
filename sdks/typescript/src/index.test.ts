/**
 * Unit tests for the AgentBudget TypeScript SDK.
 * No network calls, no API keys required.
 */

import { AgentBudget, registerModel, calculateCost, AgentBudgetError, BudgetExhausted, LoopDetected, InvalidBudget, type BudgetSession } from "./index.js";
import { jest } from "@jest/globals";

// ---------------------------------------------------------------------------
// Mock response fixtures
// ---------------------------------------------------------------------------

const mockOpenAIResp = {
  model: "gpt-4o",
  usage: { prompt_tokens: 100, completion_tokens: 50 },
  choices: [{ message: { content: "hello" } }],
};

const mockOpenAIRespNoUsage = {
  model: "gpt-4o",
  usage: null,
  choices: [{ message: { content: "hello" } }],
};

const mockAnthropicResp = {
  model: "claude-3-haiku",
  usage: { input_tokens: 200, output_tokens: 100 },
  content: [{ type: "text", text: "world" }],
};

// gpt-4o: $2.50/M input, $10.00/M output
// 100 input tokens  → 100 * 2.50e-6 = 0.00025
// 50  output tokens → 50  * 10.0e-6 = 0.0005
// total             → 0.00075
const GPT4O_100_50_COST = 0.00075;

// claude-3-haiku: $0.25/M input, $1.25/M output
// 200 * 0.25e-6 = 0.00005
// 100 * 1.25e-6 = 0.000125
// total          = 0.000175
const HAIKU_200_100_COST = 0.000175;

// ---------------------------------------------------------------------------
// Helper: session with a large budget so we can test without hitting limits
// ---------------------------------------------------------------------------
function bigSession(): BudgetSession {
  return new AgentBudget(100).newSession();
}

// ---------------------------------------------------------------------------
// 1. Budget parsing
// ---------------------------------------------------------------------------

describe("AgentBudget – budget parsing", () => {
  test('AgentBudget("$5.00") parses correctly', () => {
    const b = new AgentBudget("$5.00");
    expect(b.maxSpend).toBeCloseTo(5.0, 10);
  });

  test("AgentBudget(5) parses correctly", () => {
    const b = new AgentBudget(5);
    expect(b.maxSpend).toBe(5);
  });

  test('AgentBudget("5.00") parses correctly (no $ prefix)', () => {
    const b = new AgentBudget("5.00");
    expect(b.maxSpend).toBeCloseTo(5.0, 10);
  });

  test('AgentBudget("  $  5.00  ") tolerates surrounding whitespace', () => {
    const b = new AgentBudget("  $  5.00  ");
    expect(b.maxSpend).toBeCloseTo(5.0, 10);
  });

  test("AgentBudget(0) throws InvalidBudget", () => {
    expect(() => new AgentBudget(0)).toThrow(InvalidBudget);
  });

  test("AgentBudget(-1) throws InvalidBudget", () => {
    expect(() => new AgentBudget(-1)).toThrow(InvalidBudget);
  });

  test('AgentBudget("abc") throws InvalidBudget', () => {
    expect(() => new AgentBudget("abc")).toThrow(InvalidBudget);
  });

  test('AgentBudget("$-5") throws InvalidBudget', () => {
    expect(() => new AgentBudget("$-5")).toThrow(InvalidBudget);
  });

  test('AgentBudget("$0") throws InvalidBudget', () => {
    expect(() => new AgentBudget("$0")).toThrow(InvalidBudget);
  });
});

// ---------------------------------------------------------------------------
// 2. finalizationReserve
// ---------------------------------------------------------------------------

describe("AgentBudget – finalizationReserve", () => {
  test("reduces effective session budget", () => {
    const b = new AgentBudget(10, { finalizationReserve: 0.1 });
    // maxSpend is still the full amount
    expect(b.maxSpend).toBeCloseTo(10, 10);
    const s = b.newSession();
    // remaining should be 10 * (1 - 0.1) = 9
    expect(s.remaining).toBeCloseTo(9, 10);
  });

  test("finalizationReserve of 0 leaves budget unchanged", () => {
    const b = new AgentBudget(10, { finalizationReserve: 0 });
    expect(b.newSession().remaining).toBeCloseTo(10, 10);
  });

  test("finalizationReserve >= 1 throws InvalidBudget", () => {
    expect(() => new AgentBudget(10, { finalizationReserve: 1 })).toThrow(InvalidBudget);
  });

  test("finalizationReserve < 0 throws InvalidBudget", () => {
    expect(() => new AgentBudget(10, { finalizationReserve: -0.1 })).toThrow(InvalidBudget);
  });
});

// ---------------------------------------------------------------------------
// 3. wrapOpenAI
// ---------------------------------------------------------------------------

describe("BudgetSession.wrapOpenAI()", () => {
  test("records cost and returns response unchanged", () => {
    const session = bigSession();
    const result = session.wrapOpenAI(mockOpenAIResp);
    expect(result).toBe(mockOpenAIResp); // same reference
    expect(session.spent).toBeCloseTo(GPT4O_100_50_COST, 8);
  });

  test("preserves all fields on the returned object", () => {
    const session = bigSession();
    const result = session.wrapOpenAI(mockOpenAIResp);
    expect(result.choices[0]?.message.content).toBe("hello");
  });

  test("skips recording when usage is null", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIRespNoUsage);
    expect(session.spent).toBe(0);
  });

  test("skips recording for unknown models (no error)", () => {
    const session = bigSession();
    session.wrapOpenAI({ model: "unknown-model-xyz", usage: { prompt_tokens: 100, completion_tokens: 50 } });
    expect(session.spent).toBe(0);
  });

  test("accumulates cost across multiple calls", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.wrapOpenAI(mockOpenAIResp);
    expect(session.spent).toBeCloseTo(GPT4O_100_50_COST * 2, 8);
  });
});

// ---------------------------------------------------------------------------
// 4. wrapAnthropic
// ---------------------------------------------------------------------------

describe("BudgetSession.wrapAnthropic()", () => {
  test("records cost and returns response unchanged", () => {
    const session = bigSession();
    const result = session.wrapAnthropic(mockAnthropicResp);
    expect(result).toBe(mockAnthropicResp);
    expect(session.spent).toBeCloseTo(HAIKU_200_100_COST, 8);
  });

  test("preserves all fields on the returned object", () => {
    const session = bigSession();
    const result = session.wrapAnthropic(mockAnthropicResp);
    expect(result.content[0]?.text).toBe("world");
  });

  test("accumulates cost across multiple calls", () => {
    const session = bigSession();
    session.wrapAnthropic(mockAnthropicResp);
    session.wrapAnthropic(mockAnthropicResp);
    expect(session.spent).toBeCloseTo(HAIKU_200_100_COST * 2, 8);
  });
});

// ---------------------------------------------------------------------------
// 5. wrapUsage
// ---------------------------------------------------------------------------

describe("BudgetSession.wrapUsage()", () => {
  test("records cost for gpt-4o", () => {
    const session = bigSession();
    session.wrapUsage("gpt-4o", 100, 50);
    expect(session.spent).toBeCloseTo(GPT4O_100_50_COST, 8);
  });

  test("records cost for claude-3-haiku", () => {
    const session = bigSession();
    session.wrapUsage("claude-3-haiku", 200, 100);
    expect(session.spent).toBeCloseTo(HAIKU_200_100_COST, 8);
  });

  test("silently skips unknown model (spent stays 0)", () => {
    const session = bigSession();
    session.wrapUsage("does-not-exist", 1000, 1000);
    expect(session.spent).toBe(0);
  });

  test("strips date suffix from model name", () => {
    // "gpt-4o-2025-06-15" should resolve to "gpt-4o"
    const session = bigSession();
    session.wrapUsage("gpt-4o-2025-06-15", 100, 50);
    expect(session.spent).toBeCloseTo(GPT4O_100_50_COST, 8);
  });

  test("handles OpenRouter-prefixed model names", () => {
    // "openai/gpt-4o" → "gpt-4o"
    const session = bigSession();
    session.wrapUsage("openai/gpt-4o", 100, 50);
    expect(session.spent).toBeCloseTo(GPT4O_100_50_COST, 8);
  });
});

// ---------------------------------------------------------------------------
// 6. BudgetExhausted
// ---------------------------------------------------------------------------

describe("BudgetExhausted", () => {
  test("thrown when spending exceeds the budget", () => {
    const session = new AgentBudget(0.0001).newSession();
    expect(() => session.wrapOpenAI(mockOpenAIResp)).toThrow(BudgetExhausted);
  });

  test("BudgetExhausted is an AgentBudgetError", () => {
    const session = new AgentBudget(0.0001).newSession();
    try {
      session.wrapOpenAI(mockOpenAIResp);
    } catch (err) {
      expect(err).toBeInstanceOf(AgentBudgetError);
    }
  });

  test("BudgetExhausted carries budget and spent properties", () => {
    const session = new AgentBudget(0.0001).newSession();
    try {
      session.wrapOpenAI(mockOpenAIResp);
      fail("Expected BudgetExhausted to be thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(BudgetExhausted);
      const be = err as BudgetExhausted;
      expect(typeof be.budget).toBe("number");
      expect(typeof be.spent).toBe("number");
      expect(be.spent).toBeGreaterThan(be.budget);
    }
  });

  test("onHardLimit callback is called when budget is exhausted", () => {
    const onHardLimit = jest.fn();
    const session = new AgentBudget(0.0001, { onHardLimit }).newSession();
    try {
      session.wrapOpenAI(mockOpenAIResp);
    } catch {
      // expected
    }
    expect(onHardLimit).toHaveBeenCalledTimes(1);
  });

  test("report().terminated_by is 'budget_exhausted' after hard limit", () => {
    const session = new AgentBudget(0.0001).newSession();
    try {
      session.wrapOpenAI(mockOpenAIResp);
    } catch {
      // expected
    }
    expect(session.report().terminated_by).toBe("budget_exhausted");
  });

  test("track() also throws BudgetExhausted", () => {
    const session = new AgentBudget(0.0001).newSession();
    expect(() => session.track("result", 999, "expensive_tool")).toThrow(BudgetExhausted);
  });
});

// ---------------------------------------------------------------------------
// 7. Soft limit callback
// ---------------------------------------------------------------------------

describe("Soft limit callback", () => {
  test("fires once when spending crosses the soft limit threshold", () => {
    const onSoftLimit = jest.fn();
    // budget = $1, softLimit = 0.9 → fires when spent >= $0.90
    const session = new AgentBudget(1, { softLimit: 0.9, onSoftLimit }).newSession();

    // $2/M input, $8/M output — 100k input = $0.20 per call
    registerModel("__test_soft__", 2.0, 8.0);
    const callResp = { model: "__test_soft__", usage: { prompt_tokens: 100_000, completion_tokens: 0 } };
    // Calls 1-4: $0.20 each → $0.80 total (below $0.90 threshold)
    for (let i = 0; i < 4; i++) {
      session.wrapOpenAI(callResp);
    }
    expect(onSoftLimit).not.toHaveBeenCalled();

    // 5th call: $1.00 total — crosses $0.90 soft threshold (exactly $1.00 is NOT > $1.00, so no hard limit)
    session.wrapOpenAI(callResp);
    expect(onSoftLimit).toHaveBeenCalledTimes(1);
  });

  test("soft limit fires once then never again", () => {
    const onSoftLimit = jest.fn();
    registerModel("__test_soft2__", 0.30, 0); // $0.30/M input, $0 output
    // 1M tokens × $0.30/M = $0.30 per call
    // Budget $1, soft at 0.5 → fires when spent >= $0.50
    // Call 1: $0.30 (30% — no fire)
    // Call 2: $0.60 (60% — fires once)
    // Call 3: $0.90 (90% — should NOT fire again)
    const session = new AgentBudget(1, { softLimit: 0.5, onSoftLimit }).newSession();
    const callResp = { model: "__test_soft2__", usage: { prompt_tokens: 1_000_000, completion_tokens: 0 } };

    session.wrapOpenAI(callResp); // $0.30
    expect(onSoftLimit).not.toHaveBeenCalled();

    session.wrapOpenAI(callResp); // $0.60
    expect(onSoftLimit).toHaveBeenCalledTimes(1);

    session.wrapOpenAI(callResp); // $0.90
    expect(onSoftLimit).toHaveBeenCalledTimes(1); // still 1
  });

  test("onSoftLimit callback receives a report object", () => {
    let capturedReport: unknown = null;
    const onSoftLimit = jest.fn((r) => { capturedReport = r; });

    registerModel("__test_soft3__", 0.30, 0);
    const session = new AgentBudget(1, { softLimit: 0.5, onSoftLimit }).newSession();
    const callResp = { model: "__test_soft3__", usage: { prompt_tokens: 1_000_000, completion_tokens: 0 } };

    session.wrapOpenAI(callResp);
    session.wrapOpenAI(callResp); // crosses 0.5

    expect(capturedReport).not.toBeNull();
    const r = capturedReport as Record<string, unknown>;
    expect(typeof r["session_id"]).toBe("string");
    expect(typeof r["total_spent"]).toBe("number");
  });
});

// ---------------------------------------------------------------------------
// 8. Loop detection
// ---------------------------------------------------------------------------

describe("LoopDetected", () => {
  test("thrown after N repeated calls within the window", () => {
    // loopMaxCalls=3 means on the 3rd call with the same key → LoopDetected
    const session = new AgentBudget(100, { loopMaxCalls: 3, loopWindowSeconds: 60 }).newSession();
    session.track("r1", 0.01, "search");
    session.track("r2", 0.01, "search");
    expect(() => session.track("r3", 0.01, "search")).toThrow(LoopDetected);
  });

  test("LoopDetected carries the key that was repeated", () => {
    const session = new AgentBudget(100, { loopMaxCalls: 3, loopWindowSeconds: 60 }).newSession();
    session.track("r1", 0.01, "my_tool");
    session.track("r2", 0.01, "my_tool");
    try {
      session.track("r3", 0.01, "my_tool");
      fail("Expected LoopDetected");
    } catch (err) {
      expect(err).toBeInstanceOf(LoopDetected);
      expect((err as LoopDetected).key).toBe("my_tool");
    }
  });

  test("LoopDetected is an AgentBudgetError", () => {
    const session = new AgentBudget(100, { loopMaxCalls: 2 }).newSession();
    session.track("r1", 0.01, "loop_key");
    expect(() => session.track("r2", 0.01, "loop_key")).toThrow(AgentBudgetError);
  });

  test("different tool names do not interfere with each other", () => {
    const session = new AgentBudget(100, { loopMaxCalls: 3 }).newSession();
    // 2 calls each for two distinct tools — should not throw
    session.track("r", 0.01, "tool_a");
    session.track("r", 0.01, "tool_b");
    session.track("r", 0.01, "tool_a");
    session.track("r", 0.01, "tool_b");
    // no exception so far
    expect(true).toBe(true);
  });

  test("report().terminated_by is 'loop_detected' after loop", () => {
    const session = new AgentBudget(100, { loopMaxCalls: 2 }).newSession();
    session.track("r1", 0.01, "k");
    try {
      session.track("r2", 0.01, "k");
    } catch {
      // expected
    }
    expect(session.report().terminated_by).toBe("loop_detected");
  });

  test("onLoopDetected callback fires when loop is detected", () => {
    const onLoopDetected = jest.fn();
    const session = new AgentBudget(100, { loopMaxCalls: 2, onLoopDetected }).newSession();
    session.track("r1", 0.01, "loopy");
    try {
      session.track("r2", 0.01, "loopy");
    } catch {
      // expected
    }
    expect(onLoopDetected).toHaveBeenCalledTimes(1);
  });

  test("wrapOpenAI also participates in loop detection", () => {
    const session = new AgentBudget(100, { loopMaxCalls: 3 }).newSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.wrapOpenAI(mockOpenAIResp);
    expect(() => session.wrapOpenAI(mockOpenAIResp)).toThrow(LoopDetected);
  });
});

// ---------------------------------------------------------------------------
// 9. track()
// ---------------------------------------------------------------------------

describe("BudgetSession.track()", () => {
  test("returns the result unchanged", () => {
    const session = bigSession();
    const obj = { data: 42 };
    const result = session.track(obj, 0.05, "my_api");
    expect(result).toBe(obj);
  });

  test("records cost correctly", () => {
    const session = bigSession();
    session.track("result", 0.123, "serp_api");
    expect(session.spent).toBeCloseTo(0.123, 8);
  });

  test("works without a toolName", () => {
    const session = bigSession();
    session.track("result", 0.05);
    expect(session.spent).toBeCloseTo(0.05, 8);
  });

  test("accumulates cost across multiple tool calls", () => {
    const session = bigSession();
    session.track("a", 0.10, "tool1");
    session.track("b", 0.20, "tool2");
    expect(session.spent).toBeCloseTo(0.30, 8);
  });

  test("null/undefined results are returned as-is", () => {
    const session = bigSession();
    expect(session.track(null, 0.01)).toBeNull();
    expect(session.track(undefined, 0.01)).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 10. wouldExceed()
// ---------------------------------------------------------------------------

describe("BudgetSession.wouldExceed()", () => {
  test("returns false when cost is within remaining budget", () => {
    const session = new AgentBudget(1).newSession(); // $1 budget
    expect(session.wouldExceed(0.50)).toBe(false);
  });

  test("returns true when cost exceeds remaining budget", () => {
    const session = new AgentBudget(1).newSession();
    expect(session.wouldExceed(1.01)).toBe(true);
  });

  test("returns false for zero cost", () => {
    const session = new AgentBudget(1).newSession();
    expect(session.wouldExceed(0)).toBe(false);
  });

  test("returns true after some spend has occurred", () => {
    const session = new AgentBudget(1).newSession();
    session.track("x", 0.60, "step1");
    expect(session.wouldExceed(0.50)).toBe(true);
  });

  test("does not mutate spent when called", () => {
    const session = new AgentBudget(1).newSession();
    session.wouldExceed(0.50);
    session.wouldExceed(0.99);
    expect(session.spent).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 11. childSession()
// ---------------------------------------------------------------------------

describe("BudgetSession.childSession()", () => {
  test("caps to the requested maxSpend when parent has enough remaining", () => {
    const parent = new AgentBudget(10).newSession();
    const child = parent.childSession(2);
    expect(child.remaining).toBeCloseTo(2, 10);
  });

  test("caps to parent remaining when maxSpend exceeds parent remaining", () => {
    const parent = new AgentBudget(1).newSession();
    parent.track("x", 0.80, "step");
    // remaining = 0.20, requested = 2.00 → child gets 0.20
    const child = parent.childSession(2);
    expect(child.remaining).toBeCloseTo(0.20, 8);
  });

  test("child session is fully independent — parent spent is unchanged", () => {
    const parent = new AgentBudget(10).newSession();
    const child = parent.childSession(1);
    child.track("r", 0.50, "step");
    // Parent spent is NOT automatically updated; caller must roll up manually
    expect(parent.spent).toBe(0);
  });

  test("child session has its own id", () => {
    const parent = new AgentBudget(10).newSession();
    const child = parent.childSession(1);
    expect(child.id).not.toBe(parent.id);
  });
});

// ---------------------------------------------------------------------------
// 12. report()
// ---------------------------------------------------------------------------

describe("BudgetSession.report()", () => {
  test("returns expected structure with correct fields", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.track("result", 0.05, "serp_api");
    const r = session.report();

    expect(typeof r.session_id).toBe("string");
    expect(r.session_id).toMatch(/^sess_/);
    expect(typeof r.budget).toBe("number");
    expect(typeof r.total_spent).toBe("number");
    expect(typeof r.remaining).toBe("number");
    expect(typeof r.breakdown).toBe("object");
    expect(typeof r.duration_seconds).toBe("number");
    expect(r.terminated_by).toBeNull();
    expect(typeof r.event_count).toBe("number");
  });

  test("total_spent + remaining ≈ budget", () => {
    const session = new AgentBudget(5).newSession();
    session.track("x", 1.23, "step");
    const r = session.report();
    expect(r.total_spent + r.remaining).toBeCloseTo(r.budget, 6);
  });

  test("breakdown has llm and tools sub-objects", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.track("x", 0.01, "my_tool");
    const r = session.report();

    const bd = r.breakdown as Record<string, unknown>;
    expect(bd["llm"]).toBeDefined();
    expect(bd["tools"]).toBeDefined();

    const llm = bd["llm"] as Record<string, unknown>;
    expect(llm["calls"]).toBe(1);
    expect((llm["total"] as number)).toBeCloseTo(GPT4O_100_50_COST, 8);

    const tools = bd["tools"] as Record<string, unknown>;
    expect(tools["calls"]).toBe(1);
    expect((tools["total"] as number)).toBeCloseTo(0.01, 8);
  });

  test("event_count matches the number of recorded events", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.wrapAnthropic(mockAnthropicResp);
    session.track("x", 0.01, "t1");
    expect(session.report().event_count).toBe(3);
  });

  test("session_id matches id from opts when provided", () => {
    const b = new AgentBudget(5);
    const session = b.newSession({ id: "my-custom-id" });
    expect(session.report().session_id).toBe("my-custom-id");
  });

  test("duration_seconds is non-negative", () => {
    const session = bigSession();
    const r = session.report();
    expect(r.duration_seconds).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// 13. registerModel() + calculateCost()
// ---------------------------------------------------------------------------

describe("registerModel() and calculateCost()", () => {
  test("calculateCost() returns correct value for built-in gpt-4o", () => {
    // gpt-4o: $2.50/M input, $10.00/M output
    const cost = calculateCost("gpt-4o", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(12.5, 6);
  });

  test("calculateCost() returns undefined for unknown model", () => {
    expect(calculateCost("totally-unknown-model-zzz", 100, 100)).toBeUndefined();
  });

  test("registerModel() + calculateCost() work for custom model", () => {
    registerModel("custom-model-v1", 4.0, 16.0); // $4/M input, $16/M output
    const cost = calculateCost("custom-model-v1", 500_000, 250_000); // 0.5M input + 0.25M output
    // 0.5 * 4 + 0.25 * 16 = 2.0 + 4.0 = 6.0
    expect(cost).toBeCloseTo(6.0, 6);
  });

  test("registerModel() overrides built-in pricing for same model name", () => {
    registerModel("gpt-4o", 1.0, 2.0); // override with custom pricing
    const cost = calculateCost("gpt-4o", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(3.0, 6); // 1.0 + 2.0

    // Restore original pricing to avoid polluting other tests
    registerModel("gpt-4o", 2.5, 10.0);
  });

  test("registerModel() throws for negative prices", () => {
    expect(() => registerModel("bad-model", -1, 0)).toThrow();
    expect(() => registerModel("bad-model", 0, -1)).toThrow();
  });

  test("calculateCost() for zero tokens returns 0", () => {
    expect(calculateCost("gpt-4o", 0, 0)).toBeCloseTo(0, 10);
  });

  test("wrapUsage uses custom-registered model pricing in a session", () => {
    registerModel("my-custom-llm", 10.0, 20.0); // $10/M input, $20/M output
    const session = bigSession();
    // 1M input + 1M output → $10 + $20 = $30
    session.wrapUsage("my-custom-llm", 1_000_000, 1_000_000);
    expect(session.spent).toBeCloseTo(30.0, 6);
  });
});

// ---------------------------------------------------------------------------
// 14. close()
// ---------------------------------------------------------------------------

describe("BudgetSession.close()", () => {
  test("marks the session as done (duration_seconds is stable after close)", () => {
    const session = bigSession();
    session.wrapOpenAI(mockOpenAIResp);
    session.close();
    const r1 = session.report();
    // Simulate passage of time by calling report() again later
    const r2 = session.report();
    // duration should be the same since endTime is frozen on first close()
    expect(r1.duration_seconds).toBe(r2.duration_seconds);
  });

  test("calling close() twice is idempotent", () => {
    const session = bigSession();
    session.close();
    expect(() => session.close()).not.toThrow();
  });

  test("session can still be read after close()", () => {
    const session = bigSession();
    session.track("x", 0.01, "t");
    session.close();
    expect(session.spent).toBeCloseTo(0.01, 8);
    expect(session.report().event_count).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 15. Error class hierarchy
// ---------------------------------------------------------------------------

describe("Error class hierarchy", () => {
  test("BudgetExhausted is instanceof AgentBudgetError and Error", () => {
    const err = new BudgetExhausted(1, 2);
    expect(err).toBeInstanceOf(AgentBudgetError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("BudgetExhausted");
  });

  test("LoopDetected is instanceof AgentBudgetError and Error", () => {
    const err = new LoopDetected("some_key");
    expect(err).toBeInstanceOf(AgentBudgetError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("LoopDetected");
    expect(err.key).toBe("some_key");
  });

  test("InvalidBudget is instanceof AgentBudgetError and Error", () => {
    const err = new InvalidBudget("bad");
    expect(err).toBeInstanceOf(AgentBudgetError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("InvalidBudget");
    expect(err.value).toBe("bad");
  });

  test("error messages are human-readable strings", () => {
    expect(new BudgetExhausted(5, 6).message).toContain("Budget exhausted");
    expect(new LoopDetected("tool_x").message).toContain("Loop detected");
    expect(new InvalidBudget("abc").message).toContain("Invalid budget");
  });
});

// ---------------------------------------------------------------------------
// 16. spent / remaining accessors
// ---------------------------------------------------------------------------

describe("spent and remaining accessors", () => {
  test("spent starts at 0", () => {
    expect(bigSession().spent).toBe(0);
  });

  test("remaining equals budget before any spending", () => {
    const session = new AgentBudget(5).newSession();
    expect(session.remaining).toBeCloseTo(5, 10);
  });

  test("remaining decreases after spending", () => {
    const session = new AgentBudget(5).newSession();
    session.track("x", 1.0, "step");
    expect(session.remaining).toBeCloseTo(4.0, 8);
    expect(session.spent).toBeCloseTo(1.0, 8);
  });

  test("remaining floors at 0 (never goes negative)", () => {
    // If somehow spent overshoots budget (shouldn't happen normally)
    // Ledger clamps remaining at Math.max(0, budget - spent)
    const session = new AgentBudget(1).newSession();
    // Force a track just below the limit
    session.track("x", 0.9999, "step");
    expect(session.remaining).toBeGreaterThanOrEqual(0);
  });
});
