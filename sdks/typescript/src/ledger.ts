import { BudgetExhausted } from "./errors.js";

export type CostType = "llm" | "tool";

export interface CostEvent {
  cost: number;
  type: CostType;
  timestamp: number; // Unix ms
  model?: string | undefined;
  inputTokens?: number | undefined;
  outputTokens?: number | undefined;
  toolName?: string | undefined;
  metadata?: Record<string, unknown> | undefined;
}

/** Thread-safe-equivalent running cost tracker (JS is single-threaded). */
export class Ledger {
  private readonly _budget: number;
  private _spent = 0;
  private readonly _events: CostEvent[] = [];

  constructor(budget: number) {
    this._budget = budget;
  }

  get budget(): number { return this._budget; }
  get spent(): number { return this._spent; }
  get remaining(): number { return Math.max(0, this._budget - this._spent); }

  wouldExceed(cost: number): boolean {
    return this._spent + cost > this._budget;
  }

  /** Record a cost event. Throws BudgetExhausted if the budget is exceeded. */
  record(event: CostEvent): void {
    const newTotal = this._spent + event.cost;
    if (newTotal > this._budget) {
      throw new BudgetExhausted(this._budget, newTotal);
    }
    this._spent = newTotal;
    this._events.push(event);
  }

  events(): CostEvent[] {
    return [...this._events];
  }

  breakdown(): Record<string, unknown> {
    let llmTotal = 0;
    let llmCalls = 0;
    const byModel: Record<string, number> = {};

    let toolTotal = 0;
    let toolCalls = 0;
    const byTool: Record<string, number> = {};

    for (const e of this._events) {
      if (e.type === "llm") {
        llmTotal += e.cost;
        llmCalls++;
        if (e.model) byModel[e.model] = (byModel[e.model] ?? 0) + e.cost;
      } else {
        toolTotal += e.cost;
        toolCalls++;
        if (e.toolName) byTool[e.toolName] = (byTool[e.toolName] ?? 0) + e.cost;
      }
    }

    return {
      llm: { total: round6(llmTotal), calls: llmCalls, by_model: byModel },
      tools: { total: round6(toolTotal), calls: toolCalls, by_tool: byTool },
    };
  }
}

function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}
