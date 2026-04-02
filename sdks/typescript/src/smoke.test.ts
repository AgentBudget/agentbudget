/**
 * Integration smoke tests for the AgentBudget TypeScript SDK.
 *
 * These tests make REAL network calls to the OpenAI API and require a valid
 * OPENAI_API_KEY environment variable. They are automatically skipped when the
 * key is absent so that CI pipelines without credentials can still run the
 * full test suite without failure.
 *
 * Run only these tests with:
 *   npm run test:smoke
 *
 * Or together with the unit tests:
 *   npm test
 */

import * as https from "https";

import { AgentBudget } from "./budget.js";
import { calculateCost } from "./pricing.js";
import { BudgetExhausted } from "./errors.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal HTTPS POST — avoids importing a heavy dep like node-fetch. */
function httpsPost(
  hostname: string,
  path: string,
  headers: Record<string, string>,
  body: string
): Promise<{ status: number; data: string }> {
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname,
        path,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
          ...headers,
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk: Buffer) => { data += chunk.toString(); });
        res.on("end", () => resolve({ status: res.statusCode ?? 0, data }));
      }
    );
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Pre-flight: detect API key
// ---------------------------------------------------------------------------

const OPENAI_API_KEY = process.env["OPENAI_API_KEY"] ?? "";
const hasOpenAIKey = OPENAI_API_KEY.length > 0;

// Cheap model — keep costs as low as possible during smoke tests
const CHEAP_MODEL = "gpt-4o-mini";

// ---------------------------------------------------------------------------
// Smoke test group
// ---------------------------------------------------------------------------

describe("smoke – AgentBudget integration (requires OPENAI_API_KEY)", () => {
  beforeAll(() => {
    if (!hasOpenAIKey) {
      console.warn(
        "\n[smoke] OPENAI_API_KEY not set — all smoke tests will be skipped.\n"
      );
    }
  });

  // -------------------------------------------------------------------------
  // Test 1: Real OpenAI call → wrapUsage() records a positive spend
  // -------------------------------------------------------------------------
  test("real OpenAI call: wrapUsage() records spent > 0", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown; // skip gracefully at runtime
    }

    const budget = new AgentBudget("$1.00");
    const session = budget.newSession();

    const requestBody = JSON.stringify({
      model: CHEAP_MODEL,
      messages: [{ role: "user", content: "Say exactly the word: pong" }],
      max_tokens: 5,
    });

    const { status, data } = await httpsPost(
      "api.openai.com",
      "/v1/chat/completions",
      { Authorization: `Bearer ${OPENAI_API_KEY}` },
      requestBody
    );

    expect(status).toBe(200);

    const json = JSON.parse(data) as {
      model: string;
      usage: { prompt_tokens: number; completion_tokens: number };
      choices: { message: { content: string } }[];
    };

    expect(json.usage).toBeDefined();
    expect(json.usage.prompt_tokens).toBeGreaterThan(0);

    // Record usage with the SDK
    session.wrapUsage(json.model, json.usage.prompt_tokens, json.usage.completion_tokens);

    expect(session.spent).toBeGreaterThan(0);
    expect(session.remaining).toBeLessThan(1.0);

    session.close();
    const report = session.report();

    expect(report.total_spent).toBeGreaterThan(0);
    expect(report.event_count).toBe(1);
    expect(report.terminated_by).toBeNull();
  }, 15_000 /* 15 s timeout for network */);

  // -------------------------------------------------------------------------
  // Test 2: wrapOpenAI() on a real response object
  // -------------------------------------------------------------------------
  test("real OpenAI call: wrapOpenAI() returns the response unchanged and records cost", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown;
    }

    const budget = new AgentBudget("$1.00");
    const session = budget.newSession();

    const requestBody = JSON.stringify({
      model: CHEAP_MODEL,
      messages: [{ role: "user", content: "Reply with a single digit: 7" }],
      max_tokens: 5,
    });

    const { status, data } = await httpsPost(
      "api.openai.com",
      "/v1/chat/completions",
      { Authorization: `Bearer ${OPENAI_API_KEY}` },
      requestBody
    );

    expect(status).toBe(200);

    const json = JSON.parse(data) as {
      model: string;
      usage: { prompt_tokens: number; completion_tokens: number };
      choices: { message: { content: string } }[];
    };

    // wrapOpenAI should return the exact same object reference
    const returned = session.wrapOpenAI(json);
    expect(returned).toBe(json);

    // The content should still be accessible
    expect(returned.choices.length).toBeGreaterThan(0);

    // Cost should have been recorded
    expect(session.spent).toBeGreaterThan(0);

    session.close();
  }, 15_000);

  // -------------------------------------------------------------------------
  // Test 3: Spending is consistent with calculateCost()
  // -------------------------------------------------------------------------
  test("real call: session.spent matches calculateCost() for the same token counts", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown;
    }

    const budget = new AgentBudget("$1.00");
    const session = budget.newSession();

    const requestBody = JSON.stringify({
      model: CHEAP_MODEL,
      messages: [{ role: "user", content: "Write the number 42." }],
      max_tokens: 5,
    });

    const { data } = await httpsPost(
      "api.openai.com",
      "/v1/chat/completions",
      { Authorization: `Bearer ${OPENAI_API_KEY}` },
      requestBody
    );

    const json = JSON.parse(data) as {
      model: string;
      usage: { prompt_tokens: number; completion_tokens: number };
    };

    const expectedCost = calculateCost(
      json.model,
      json.usage.prompt_tokens,
      json.usage.completion_tokens
    );

    session.wrapUsage(json.model, json.usage.prompt_tokens, json.usage.completion_tokens);

    // Allow a tiny floating point epsilon
    expect(session.spent).toBeCloseTo(expectedCost ?? 0, 8);
  }, 15_000);

  // -------------------------------------------------------------------------
  // Test 4: BudgetExhausted is thrown when the budget is too small for the call
  // -------------------------------------------------------------------------
  test("real call: BudgetExhausted thrown if budget is smaller than cost", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown;
    }

    // A real call to gpt-4o-mini with even tiny token counts will cost more than $0.000001
    const budget = new AgentBudget(0.000001 /* $0.000001 */);
    const session = budget.newSession();

    const requestBody = JSON.stringify({
      model: CHEAP_MODEL,
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 3,
    });

    const { data } = await httpsPost(
      "api.openai.com",
      "/v1/chat/completions",
      { Authorization: `Bearer ${OPENAI_API_KEY}` },
      requestBody
    );

    const json = JSON.parse(data) as {
      model: string;
      usage: { prompt_tokens: number; completion_tokens: number };
    };

    expect(() =>
      session.wrapUsage(json.model, json.usage.prompt_tokens, json.usage.completion_tokens)
    ).toThrow(BudgetExhausted);
  }, 15_000);

  // -------------------------------------------------------------------------
  // Test 5: Multiple sequential calls accumulate cost
  // -------------------------------------------------------------------------
  test("real calls: cost accumulates across multiple wrapUsage() invocations", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown;
    }

    const budget = new AgentBudget("$5.00");
    const session = budget.newSession();

    const makeCall = async (content: string) => {
      const requestBody = JSON.stringify({
        model: CHEAP_MODEL,
        messages: [{ role: "user", content }],
        max_tokens: 5,
      });
      const { data } = await httpsPost(
        "api.openai.com",
        "/v1/chat/completions",
        { Authorization: `Bearer ${OPENAI_API_KEY}` },
        requestBody
      );
      return JSON.parse(data) as {
        model: string;
        usage: { prompt_tokens: number; completion_tokens: number };
      };
    };

    const resp1 = await makeCall("Say: alpha");
    session.wrapUsage(resp1.model, resp1.usage.prompt_tokens, resp1.usage.completion_tokens);
    const spentAfter1 = session.spent;
    expect(spentAfter1).toBeGreaterThan(0);

    const resp2 = await makeCall("Say: beta");
    session.wrapUsage(resp2.model, resp2.usage.prompt_tokens, resp2.usage.completion_tokens);
    const spentAfter2 = session.spent;

    expect(spentAfter2).toBeGreaterThan(spentAfter1);
    expect(session.report().event_count).toBe(2);

    session.close();
  }, 30_000 /* 30 s for two network calls */);

  // -------------------------------------------------------------------------
  // Test 6: report() structure is valid after a real call
  // -------------------------------------------------------------------------
  test("real call: report() returns a well-formed Report object", async () => {
    if (!hasOpenAIKey) {
      return test.skip as unknown;
    }

    const budget = new AgentBudget("$1.00");
    const session = budget.newSession();

    const requestBody = JSON.stringify({
      model: CHEAP_MODEL,
      messages: [{ role: "user", content: "One word: yes" }],
      max_tokens: 3,
    });

    const { data } = await httpsPost(
      "api.openai.com",
      "/v1/chat/completions",
      { Authorization: `Bearer ${OPENAI_API_KEY}` },
      requestBody
    );

    const json = JSON.parse(data) as {
      model: string;
      usage: { prompt_tokens: number; completion_tokens: number };
    };

    session.wrapUsage(json.model, json.usage.prompt_tokens, json.usage.completion_tokens);
    session.close();

    const r = session.report();

    // All required fields present
    expect(typeof r.session_id).toBe("string");
    expect(r.session_id.length).toBeGreaterThan(0);
    expect(r.budget).toBeCloseTo(1.0, 6);
    expect(r.total_spent).toBeGreaterThan(0);
    expect(r.remaining).toBeGreaterThanOrEqual(0);
    expect(r.total_spent + r.remaining).toBeCloseTo(r.budget, 6);
    expect(r.event_count).toBe(1);
    expect(r.terminated_by).toBeNull();
    expect(r.duration_seconds).toBeGreaterThanOrEqual(0);

    const bd = r.breakdown as Record<string, unknown>;
    expect(bd["llm"]).toBeDefined();
    const llm = bd["llm"] as Record<string, unknown>;
    expect(llm["calls"]).toBe(1);
    expect((llm["total"] as number)).toBeCloseTo(r.total_spent, 6);
  }, 15_000);
});
