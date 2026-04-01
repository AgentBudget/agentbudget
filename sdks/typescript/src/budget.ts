import { InvalidBudget } from "./errors.js";
import { Ledger } from "./ledger.js";
import { CircuitBreaker } from "./circuit_breaker.js";
import { BudgetSession, SessionOptions, Report } from "./session.js";

export interface BudgetOptions {
  /** Fraction of budget at which the soft-limit callback fires. Default: 0.9 */
  softLimit?: number;
  /** Max repeated calls in the loop window before LoopDetected is thrown. Default: 10 */
  loopMaxCalls?: number;
  /** Loop detection window in seconds. Default: 60 */
  loopWindowSeconds?: number;
  /** Fraction of budget to hold back from hard limit enforcement. Default: 0 */
  finalizationReserve?: number;
  /** Called once when spending exceeds the soft limit. */
  onSoftLimit?: ((report: Report) => void) | undefined;
  /** Called when BudgetExhausted is thrown. */
  onHardLimit?: ((report: Report) => void) | undefined;
  /** Called when LoopDetected is thrown. */
  onLoopDetected?: ((report: Report) => void) | undefined;
}

/**
 * AgentBudget is the entry point. Create one per agent type, then call
 * `newSession()` for each agent run.
 *
 * @example
 * ```ts
 * import { AgentBudget } from "agentbudget";
 *
 * const budget = new AgentBudget("$5.00");
 * const session = budget.newSession();
 *
 * const resp = await openai.chat.completions.create({ model: "gpt-4o", ... });
 * session.wrapOpenAI(resp);
 *
 * console.log(session.spent, session.remaining);
 * session.close();
 * ```
 */
export class AgentBudget {
  private readonly _budget: number;
  private readonly _reserveBudget: number;
  private readonly _softLimit: number;
  private readonly _loopMaxCalls: number;
  private readonly _loopWindowSeconds: number;
  private readonly _onSoftLimit?: ((r: Report) => void) | undefined;
  private readonly _onHardLimit?: ((r: Report) => void) | undefined;
  private readonly _onLoopDetected?: ((r: Report) => void) | undefined;

  constructor(maxSpend: string | number, opts: BudgetOptions = {}) {
    const raw = parseBudget(maxSpend);

    const finReserve = opts.finalizationReserve ?? 0;
    if (finReserve < 0 || finReserve >= 1) {
      throw new InvalidBudget(`finalizationReserve must be in [0, 1), got ${finReserve}`);
    }

    this._reserveBudget = raw;
    this._budget = raw * (1 - finReserve);
    this._softLimit = opts.softLimit ?? 0.9;
    this._loopMaxCalls = opts.loopMaxCalls ?? 10;
    this._loopWindowSeconds = opts.loopWindowSeconds ?? 60;
    this._onSoftLimit = opts.onSoftLimit;
    this._onHardLimit = opts.onHardLimit;
    this._onLoopDetected = opts.onLoopDetected;
  }

  /** Total configured budget including any finalization reserve. */
  get maxSpend(): number { return this._reserveBudget; }

  /** Create a new budget-enforced session for an agent run. */
  newSession(opts?: SessionOptions): BudgetSession {
    const ledger = new Ledger(this._budget);
    const breaker = new CircuitBreaker(
      this._softLimit,
      this._loopMaxCalls,
      this._loopWindowSeconds
    );
    return new BudgetSession(ledger, breaker, {
      onSoftLimit: this._onSoftLimit,
      onHardLimit: this._onHardLimit,
      onLoopDetected: this._onLoopDetected,
      ...opts,
    });
  }
}

function parseBudget(value: string | number): number {
  if (typeof value === "number") {
    if (value <= 0) throw new InvalidBudget(String(value));
    return value;
  }
  const cleaned = value.trim().replace(/^\$/, "").trim();
  const amount = parseFloat(cleaned);
  if (isNaN(amount) || amount <= 0) throw new InvalidBudget(value);
  return amount;
}
