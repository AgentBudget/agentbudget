<!--
Thanks for contributing to AgentBudget! Please fill out the sections below.
See CONTRIBUTING.md for setup and the per-SDK test commands.
-->

## Summary

<!-- What does this PR change, and why? -->

## Related issue

<!-- e.g. "Closes #123". PRs that fix a bug should link the issue they resolve. -->

## Which SDK(s) does this touch?

- [ ] Python (`agentbudget`)
- [ ] Go (`sdks/go`)
- [ ] TypeScript (`@agentbudget/agentbudget`)
- [ ] Website / docs
- [ ] CI / tooling

## Checklist

- [ ] I ran the tests for the SDK(s) I changed (`pytest` / `go test ./...` / `npm test`).
- [ ] I added or updated tests covering the change.
- [ ] I did **not** hand-edit generated pricing files. If pricing changed, I edited
      `pricing.json` and ran `python scripts/generate_pricing.py` to regenerate
      `agentbudget/pricing.py`, `sdks/go/pricing.go`, and `sdks/typescript/src/pricing.ts`.
- [ ] I updated `CHANGELOG.md` (for user-facing changes) and any relevant docs/README.
