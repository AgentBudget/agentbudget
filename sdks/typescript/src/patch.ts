/**
 * SDK patching utilities — attach a BudgetSession to an OpenAI or Anthropic
 * client instance so every call is tracked automatically.
 *
 * Unlike Python's global monkey-patching, this wraps a specific client
 * instance so other clients of the same type are unaffected.
 */

import { BudgetSession } from "./session.js";

type AnyFunction = (...args: unknown[]) => unknown;

/** Wrap the `.create` method on an OpenAI chat completions object. */
function patchOpenAICompletions(completions: Record<string, AnyFunction>, session: BudgetSession): void {
  const original = completions["create"] as AnyFunction;
  if (!original || (original as { _agentbudget?: boolean })._agentbudget) return;

  const patched = async function (this: unknown, ...args: unknown[]) {
    const resp = await (original.apply(this, args) as Promise<unknown>);
    // Only wrap non-streaming responses (streaming requires iteration)
    if (resp && typeof resp === "object" && "model" in resp && "usage" in resp) {
      session.wrapOpenAI(resp as Parameters<typeof session.wrapOpenAI>[0]);
    }
    return resp;
  };
  (patched as { _agentbudget?: boolean })._agentbudget = true;
  completions["create"] = patched as AnyFunction;
}

/** Wrap the `.create` method on an Anthropic messages object. */
function patchAnthropicMessages(messages: Record<string, AnyFunction>, session: BudgetSession): void {
  const original = messages["create"] as AnyFunction;
  if (!original || (original as { _agentbudget?: boolean })._agentbudget) return;

  const patched = async function (this: unknown, ...args: unknown[]) {
    const resp = await (original.apply(this, args) as Promise<unknown>);
    if (resp && typeof resp === "object" && "model" in resp && "usage" in resp) {
      session.wrapAnthropic(resp as Parameters<typeof session.wrapAnthropic>[0]);
    }
    return resp;
  };
  (patched as { _agentbudget?: boolean })._agentbudget = true;
  messages["create"] = patched as AnyFunction;
}

/**
 * Attach a BudgetSession to a specific OpenAI or Anthropic client instance.
 * Only that instance's API calls will be tracked.
 *
 * @example
 * ```ts
 * import OpenAI from "openai";
 * import { AgentBudget, wrapClient } from "agentbudget";
 *
 * const budget = new AgentBudget("$5.00");
 * const session = budget.newSession();
 * const client = wrapClient(new OpenAI(), session);
 *
 * // All calls on `client` are now tracked:
 * const resp = await client.chat.completions.create({ ... });
 * // Other OpenAI instances are NOT tracked.
 * ```
 */
export function wrapClient<T extends object>(client: T, session: BudgetSession): T {
  const c = client as Record<string, unknown>;

  // OpenAI client: client.chat.completions.create
  if (
    c["chat"] &&
    typeof c["chat"] === "object" &&
    (c["chat"] as Record<string, unknown>)["completions"]
  ) {
    const completions = (c["chat"] as Record<string, unknown>)[
      "completions"
    ] as Record<string, AnyFunction>;
    patchOpenAICompletions(completions, session);
  }

  // Anthropic client: client.messages.create
  if (c["messages"] && typeof c["messages"] === "object") {
    const messages = c["messages"] as Record<string, AnyFunction>;
    patchAnthropicMessages(messages, session);
  }

  return client;
}
