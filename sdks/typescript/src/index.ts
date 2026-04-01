/**
 * agentbudget — Real-time cost enforcement for AI agent sessions.
 *
 * @example
 * ```ts
 * import { AgentBudget } from "agentbudget";
 *
 * const budget = new AgentBudget("$5.00");
 * const session = budget.newSession();
 *
 * // After your OpenAI call:
 * session.wrapOpenAI(response);
 *
 * // After your Anthropic call:
 * session.wrapAnthropic(response);
 *
 * // After a tool/API call with a known cost:
 * session.track(result, 0.05, "serp_api");
 *
 * console.log(session.spent, session.remaining);
 * session.close();
 * ```
 */

export { AgentBudget } from "./budget.js";
export type { BudgetOptions } from "./budget.js";

export { BudgetSession } from "./session.js";
export type {
  Report,
  SessionOptions,
  OpenAICompletionLike,
  AnthropicMessageLike,
} from "./session.js";

export { wrapClient } from "./patch.js";

export { registerModel, calculateCost } from "./pricing.js";

export {
  AgentBudgetError,
  BudgetExhausted,
  LoopDetected,
  InvalidBudget,
} from "./errors.js";
