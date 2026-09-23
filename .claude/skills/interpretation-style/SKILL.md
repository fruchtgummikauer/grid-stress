---
name: interpretation-style
description: How to write interpretation, findings and conclusion markdown in this repo's
  notebooks and in plans that describe them. Use whenever writing or revising the text that
  explains a table, plot or result.
---

Interpretation text is for team members who read the notebook without running or reading the
code. It must be short, but it must stand on its own.

## Rules

- **Concise, with enough context.** Say what the reader is looking at and why it matters, then
  stop. No restating the code, no filler.
- **Every finding carries an example from the data.** Quote a concrete number from the output
  directly above: a value, a date, an hour, a bin. It shows exactly what the claim means. A finding
  without a number is not finished.
- **Bullets for major findings, indented sub-bullets for the supporting evidence.**
- **A blank line between different finding topics.**
- **Units and windows on every number:** `MWh`, `%`, `h`, and where it matters, the window or
  `hour_count` the number is averaged over.
- **State results plainly, whichever way they come out.** Don't claim more than the data supports.
  If a trend is not established, say so instead of implying one.
- **Numbers come from the executed outputs, never from memory or the spec.** If a re-run changes
  them, update the text.

## Final Findings section

- A **short synthesis**, not a 1:1 copy of the section notes: one or two bullets per section,
  pointing back to the section for detail.
- Add something new only where sections connect, e.g. one result explaining another.
- If a section has nothing more to add, a one-line summary is enough.
- A reference section the spec asks for explicitly (e.g. a list of benchmark numbers) may stay a
  numeric list. It is not the same as the synthesis.

## Example

```markdown
- **The generation forecast is weakest around solar midday.**
  - MAE peaks at 13:00 (≈ 2,400 MWh) against ≈ 900 MWh at 03:00.
  - The bias at the peak is negative (−600 MWh), so midday solar is under-forecast.

- **Grid-load error shows no daily pattern worth modelling.**
  - MAE stays between 800 and 1,000 MWh across all 24 hours.
```

(Numbers in the example are illustrative only.)
