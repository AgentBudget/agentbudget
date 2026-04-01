// Package agentbudget provides real-time cost enforcement for AI agent sessions.
//
// Set a hard dollar limit on any LLM session. Wrap API responses, track tool
// costs, and get automatic circuit breaking — no surprise bills.
//
// # Quickstart
//
//	budget, err := agentbudget.New("$5.00")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	session := budget.NewSession()
//	defer session.Close()
//
//	// After calling your LLM, record usage:
//	err = session.WrapUsage("gpt-4o", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
//	if err != nil {
//	    // *agentbudget.BudgetExhausted or *agentbudget.LoopDetected
//	}
//
//	fmt.Printf("spent: $%.4f  remaining: $%.4f\n", session.Spent(), session.Remaining())
//
// # OpenAI Example
//
//	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
//	budget, _ := agentbudget.New(5.0)
//	session := budget.NewSession()
//	defer session.Close()
//
//	resp, _ := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
//	    Model:    openai.GPT4o,
//	    Messages: messages,
//	})
//	if err := session.WrapUsage(
//	    resp.Model,
//	    int64(resp.Usage.PromptTokens),
//	    int64(resp.Usage.CompletionTokens),
//	); err != nil {
//	    log.Fatal(err)
//	}
//
// # Anthropic Example
//
//	client, _ := anthropic.NewClient(os.Getenv("ANTHROPIC_API_KEY"))
//	resp, _ := client.Messages.New(ctx, anthropic.MessageNewParams{...})
//	if err := session.WrapUsage(
//	    string(resp.Model),
//	    resp.Usage.InputTokens,
//	    resp.Usage.OutputTokens,
//	); err != nil {
//	    log.Fatal(err)
//	}
package agentbudget
