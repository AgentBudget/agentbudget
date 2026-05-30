# Contributing to AgentBudget

Thanks for your interest in contributing! AgentBudget is a small, dependency-free
toolkit for putting hard spend limits on AI agents. This guide covers how the
repo is laid out, how to run each SDK's tests, and the one workflow gotcha that
trips up most first PRs (generated pricing files).

## Repository layout

This is a monorepo with three SDKs that share one source of truth for pricing:

| Path | What it is |
| --- | --- |
| `agentbudget/` | The Python package (`agentbudget`) — the reference implementation. |
| `sdks/go/` | The Go SDK (`github.com/AgentBudget/agentbudget/sdks/go`). |
| `sdks/typescript/` | The TypeScript SDK (`@agentbudget/agentbudget`). |
| `website/` | The Next.js marketing + docs site (`agentbudget.dev`). |
| `pricing.json` | **Source of truth** for model pricing (see below). |
| `scripts/generate_pricing.py` | Generates the per-SDK pricing files from `pricing.json`. |
| `tests/` | Python test suite. |

## Development setup & running tests

Run the tests for whichever SDK(s) you touched.

### Python (`agentbudget/`)

Supports Python 3.9–3.13.

```bash
pip install -e ".[dev]"
pytest
```

### Go (`sdks/go/`)

Requires Go 1.21+. The SDK is standard-library only.

```bash
cd sdks/go
go test ./...
```

### TypeScript (`sdks/typescript/`)

```bash
cd sdks/typescript
npm install
npm run typecheck
npm test          # unit tests
npm run build     # bundle with tsup
```

### Website (`website/`)

```bash
cd website
npm install
npm run lint
npm run build
```

## ⚠️ Don't hand-edit generated pricing files

Model pricing lives in **`pricing.json`** at the repo root. The following files are
**generated** from it and must never be edited by hand:

- `agentbudget/pricing.py`
- `sdks/go/pricing.go`
- `sdks/typescript/src/pricing.ts`

To add or change a model's pricing:

```bash
# 1. Edit pricing.json
# 2. Regenerate the per-SDK files:
python scripts/generate_pricing.py
# 3. Commit pricing.json AND all regenerated files together.
```

CI runs `python scripts/generate_pricing.py --check` (the `pricing-sync` job) and
will fail if the generated files are out of sync with `pricing.json`. These files
are also guarded by `CODEOWNERS`, so a PR that edits them by hand will require
maintainer review and fail the sync check.

## Submitting a pull request

1. Fork the repo and create a branch (`fix/...`, `feat/...`, `docs/...`).
2. Make your change and add or update tests.
3. Run the relevant SDK's tests (see above).
4. Update `CHANGELOG.md` for any user-facing change, and the README/docs if behavior changed.
5. Open a PR using the template, and link the issue it resolves (e.g. `Closes #123`).

Small, focused PRs are easier to review and merge. If you're planning a larger
change, open an issue first to discuss the approach.

## Reporting bugs & requesting features

Use the issue templates (Bug report / Feature request). For bugs, please say which
SDK and version you're on — behavior can differ between the Python, Go, and
TypeScript implementations.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache-2.0 License](LICENSE) that covers this project.
