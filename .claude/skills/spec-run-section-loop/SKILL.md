---
name: spec-run-section-loop
description: How to run a spec from .claude/specs/ into its output notebook — one section at a
  time, with a review and an approved commit per section. Use whenever implementing or running a
  spec, or when asked to continue a spec run.
---

A spec is never implemented in one go. The notebook grows section by section, and every section
passes a review and a commit review before the next one starts.

## Before the first section

- Plan first: list the notebook sections in order, the files touched, and the open questions. Ask
  about anything ambiguous before writing any cell.
- Check the spec's status in CLAUDE.md. A spec that has already been run is never re-run, and its
  output notebook is never overwritten unless a team member explicitly asks and confirms.
- Work on a `feature/*` branch off `main`.

## The loop, per section

1. **Build**: append the section's code and markdown cells. Never touch cells from sections that
   are already approved, including any edits the team made to them. Cells that differ from the
   spec are team edits, not drift to repair.
2. **Run**: execute the whole notebook from a fresh kernel so the section is tested against
   everything before it:
   `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>`.
   Fix errors until it runs cleanly.
3. **Interpret**: read the actual outputs, then write the section's interpretation markdown from
   those numbers (see the `interpretation-style` skill). Run again so the outputs match the text.
4. **Review, then stop**: hand the executed notebook, outputs included, to the user with a short
   summary of what the section contains and any deviations from the spec. **Wait for approval.**
   Apply requested changes and repeat steps 2–4 until it is approved.
5. **Clear outputs**: `uv run jupyter nbconvert --clear-output --inplace <notebook>`. Committed
   notebooks carry no outputs. `--clear-output` keeps each cell's `metadata.execution`
   timestamps, which churn on every run, so strip those too, e.g. with nbformat:
   `for c in nb.cells: c.metadata.pop("execution", None)`.
6. **Commit**: one commit per section, message per the `commit-style` skill. Stage only the
   notebook.
7. **Commit review, then stop**: show the message and `git show --stat`. **Wait for approval.**
   Amend only if asked.
8. **Ask to proceed**: ask explicitly before starting the next section. Never roll on
   automatically.

## Rules across the whole run

- Sections that summarise the others (Findings, benchmark or conclusion sections) come last.
- Files the notebook writes into gitignored `data/` folders are side effects. They are never staged.
- Nothing is pushed unless the user asks.
- If a section needs something the spec rules out (a new dependency, an out-of-scope input), stop
  and raise it rather than working around it.
- After the last section, update the spec's status in CLAUDE.md (spec table, layout tree, data
  section) through the same review → commit → commit review steps, as a separate commit.
