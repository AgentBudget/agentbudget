export class AgentBudgetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentBudgetError";
  }
}

export class BudgetExhausted extends AgentBudgetError {
  readonly budget: number;
  readonly spent: number;

  constructor(budget: number, spent: number) {
    super(
      `Budget exhausted: spent $${spent.toFixed(4)} of $${budget.toFixed(2)} budget`
    );
    this.name = "BudgetExhausted";
    this.budget = budget;
    this.spent = spent;
  }
}

export class LoopDetected extends AgentBudgetError {
  readonly key: string;

  constructor(key: string) {
    super(`Loop detected: repeated calls to "${key}"`);
    this.name = "LoopDetected";
    this.key = key;
  }
}

export class InvalidBudget extends AgentBudgetError {
  readonly value: string;

  constructor(value: string) {
    super(`Invalid budget value: "${value}"`);
    this.name = "InvalidBudget";
    this.value = value;
  }
}
