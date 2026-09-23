---
name: commit-style
description: This repo's commit message format. Use whenever writing a commit
  message, amending one, or drafting a PR title.
---

Conventional commits, imperative mood, 72-char subject limit:

  feat(eda): add eda for smard's generation data
  fix(risk-definition-notebook): treat label comparison as case-insensitive
  refactor(notebook): move the repeating style function into helper_function()
  docs(claude-md): document the metrics data pipeline
  chore(deps): add requests and python-dotenv as runtime deps

Rules:
- Types: feat, fix, refactor, test, docs, chore, perf
- Scope is the repo area touched — usually a notebook (eda, risk-definition-notebook), but also
  claude-md, specs, deps, data-pipeline, modeling for non-notebook changes
- Body explains WHY, never WHAT (the diff shows what)
- Reference the spec: "Implements .claude/specs/02-Risk-Definition.md"
- Never mention the tools used to write the code
