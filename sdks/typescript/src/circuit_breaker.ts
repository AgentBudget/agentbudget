export class CircuitBreaker {
  private readonly softLimitFraction: number;
  private softLimitTriggered = false;
  private readonly loopMaxCalls: number;
  private readonly loopWindowMs: number;
  private readonly callLog = new Map<string, number[]>();

  constructor(
    softLimitFraction = 0.9,
    loopMaxCalls = 10,
    loopWindowSeconds = 60
  ) {
    this.softLimitFraction = softLimitFraction;
    this.loopMaxCalls = loopMaxCalls;
    this.loopWindowMs = loopWindowSeconds * 1000;
  }

  /**
   * Returns a warning message if the soft limit is reached (fires once).
   * Returns undefined otherwise.
   */
  checkBudget(spent: number, budget: number): string | undefined {
    if (budget <= 0) return undefined;
    const fraction = spent / budget;
    if (fraction >= this.softLimitFraction && !this.softLimitTriggered) {
      this.softLimitTriggered = true;
      return `Soft limit reached: ${Math.round(fraction * 100)}% of budget used ($${spent.toFixed(4)} / $${budget.toFixed(2)})`;
    }
    return undefined;
  }

  /** Returns true if a loop is detected for the given key. */
  checkLoop(key: string): boolean {
    if (this.loopMaxCalls <= 0) return false;

    const now = Date.now();
    const cutoff = now - this.loopWindowMs;

    const existing = (this.callLog.get(key) ?? []).filter((t) => t > cutoff);
    existing.push(now);
    this.callLog.set(key, existing);

    return existing.length >= this.loopMaxCalls;
  }
}
