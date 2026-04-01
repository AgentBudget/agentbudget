import { BudgetExhausted, LoopDetected } from "./errors.js";
import { Ledger, CostEvent } from "./ledger.js";
import { CircuitBreaker } from "./circuit_breaker.js";
import { calculateCost } from "./pricing.js";

export interface Report {
  session_id: string;
  budget: number;
  total_spent: number;
  remaining: number;
  breakdown: Record<string, unknown>;
  duration_seconds: number | null;
  terminated_by: "budget_exhausted" | "loop_detected" | null;
  event_count: number;
}

/** Minimal shape we need from an OpenAI chat completion response. */
export interface OpenAICompletionLike {
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
  } | null;
}

/** Minimal shape we need from an Anthropic message response. */
export interface AnthropicMessageLike {
  model: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export interface SessionOptions {
  id?: string | undefined;
  onSoftLimit?: ((report: Report) => void) | undefined;
  onHardLimit?: ((report: Report) => void) | undefined;
  onLoopDetected?: ((report: Report) => void) | undefined;
}

let _counter = 0;

function generateId(): string {
  return `sess_${Date.now()}_${++_counter}`;
}

/**
 * BudgetSession tracks costs for a single agent run.
 * Create via `Budget.newSession()`.
 */
export class BudgetSession {
  private readonly _id: string;
  private readonly _ledger: Ledger;
  private readonly _breaker: CircuitBreaker;
  private readonly _onSoftLimit?: ((r: Report) => void) | undefined;
  private readonly _onHardLimit?: ((r: Report) => void) | undefined;
  private readonly _onLoopDetected?: ((r: Report) => void) | undefined;
  private readonly _startTime: number;
  private _endTime: number | null = null;
  private _terminatedBy: Report["terminated_by"] = null;

  constructor(ledger: Ledger, breaker: CircuitBreaker, opts: SessionOptions = {}) {
    this._id = opts.id ?? generateId();
    this._ledger = ledger;
    this._breaker = breaker;
    this._onSoftLimit = opts.onSoftLimit;
    this._onHardLimit = opts.onHardLimit;
    this._onLoopDetected = opts.onLoopDetected;
    this._startTime = Date.now();
  }

  get id(): string { return this._id; }
  get spent(): number { return this._ledger.spent; }
  get remaining(): number { return this._ledger.remaining; }

  /** Returns true if the given cost would exceed the remaining budget. Does not record. */
  wouldExceed(cost: number): boolean {
    return this._ledger.wouldExceed(cost);
  }

  /**
   * Record the cost of an OpenAI chat completion response.
   * Throws BudgetExhausted or LoopDetected if limits are reached.
   */
  wrapOpenAI<T extends OpenAICompletionLike>(response: T): T {
    if (!response.usage) return response;
    const { prompt_tokens, completion_tokens } = response.usage;
    this._recordLLM(response.model, prompt_tokens, completion_tokens);
    return response;
  }

  /**
   * Record the cost of an Anthropic message response.
   * Throws BudgetExhausted or LoopDetected if limits are reached.
   */
  wrapAnthropic<T extends AnthropicMessageLike>(response: T): T {
    this._recordLLM(
      response.model,
      response.usage.input_tokens,
      response.usage.output_tokens
    );
    return response;
  }

  /**
   * Record the cost of any LLM call given the model name and token counts directly.
   * Throws BudgetExhausted or LoopDetected if limits are reached.
   */
  wrapUsage(model: string, inputTokens: number, outputTokens: number): void {
    this._recordLLM(model, inputTokens, outputTokens);
  }

  /**
   * Track a fixed-cost action (tool call, API call, etc.).
   * Returns the result unchanged for easy chaining.
   * Throws BudgetExhausted or LoopDetected if limits are reached.
   */
  track<T>(result: T, cost: number, toolName?: string, metadata?: Record<string, unknown>): T {
    const event: CostEvent = {
      cost,
      type: "tool",
      timestamp: Date.now(),
      toolName,
      metadata,
    };
    this._record(event, toolName);
    return result;
  }

  /**
   * Create a child session with a sub-budget.
   * When the child is done, roll up its spend with:
   *   parent.track(null, child.spent, "child:" + child.id)
   */
  childSession(maxSpend: number, opts?: SessionOptions): BudgetSession {
    const cap = Math.min(maxSpend, this.remaining);
    const childLedger = new Ledger(cap);
    const childBreaker = new CircuitBreaker(
      this._breaker["softLimitFraction"],
      this._breaker["loopMaxCalls"],
      this._breaker["loopWindowMs"] / 1000
    );
    return new BudgetSession(childLedger, childBreaker, opts ?? {});
  }

  /** Mark the session as ended. */
  close(): void {
    if (this._endTime === null) {
      this._endTime = Date.now();
    }
  }

  /** Structured cost report. */
  report(): Report {
    const endMs = this._endTime ?? Date.now();
    const durationSecs = Math.round((endMs - this._startTime) / 10) / 100;

    return {
      session_id: this._id,
      budget: this._ledger.budget,
      total_spent: round6(this._ledger.spent),
      remaining: round6(this._ledger.remaining),
      breakdown: this._ledger.breakdown(),
      duration_seconds: durationSecs,
      terminated_by: this._terminatedBy,
      event_count: this._ledger.events().length,
    };
  }

  private _recordLLM(model: string, inputTokens: number, outputTokens: number): void {
    const cost = calculateCost(model, inputTokens, outputTokens);
    if (cost === undefined) return; // unknown model — skip silently

    const event: CostEvent = {
      cost,
      type: "llm",
      timestamp: Date.now(),
      model,
      inputTokens,
      outputTokens,
    };
    this._record(event, model);
  }

  private _record(event: CostEvent, loopKey?: string): void {
    try {
      this._ledger.record(event);
    } catch (err) {
      if (err instanceof BudgetExhausted) {
        this._terminatedBy = "budget_exhausted";
        this._endTime ??= Date.now();
        this._onHardLimit?.(this.report());
      }
      throw err;
    }

    // Soft limit check (fires once)
    const warning = this._breaker.checkBudget(this._ledger.spent, this._ledger.budget);
    if (warning) {
      this._onSoftLimit?.(this.report());
    }

    // Loop detection
    if (loopKey && this._breaker.checkLoop(loopKey)) {
      this._terminatedBy = "loop_detected";
      this._endTime ??= Date.now();
      this._onLoopDetected?.(this.report());
      throw new LoopDetected(loopKey);
    }
  }
}

function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}
