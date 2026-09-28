---
name: notebook-refactor
description: How to refactor an existing notebook in this repo cell by cell - a marked proposed
  cell below each original, reviewed and edited by the user, applied only on confirmation. Use
  whenever asked to refactor, simplify, restyle or clean up notebook cells.
---


A refactor never edits a cell directly. Each cell gets a *proposed cell* directly below it; the
user reviews and usually edits it in the IDE, and only their confirmation moves it into the
original cell. Behaviour must not change unless the user asks for it.

## Before the first cell

- Agree on the scope (which sections, VS Code cell range) and read CLAUDE.md's status for the
  notebook: spec output notebooks are never edited without asking.
- Map the cells in scope and list which of their names later cells use (`tokenize`/`ast` over the
  later code cells). That list decides what can be cut or renamed.
- Ask once, up front: may names used by later sections be renamed (then propose the follow-up
  edits)? How far should "beginner friendly" go? Formatting (black vs. the user's style)? Are the
  markdown cells in scope? Settled answers for this repo are under *Style*.

## The loop, per cell

1. **Show** the current cell (source and saved output) and what later cells use from it.
2. **Ask** about everything ambiguous before drafting (AskUserQuestion, recommended option
   first). Always ask **per demo output** (every print, table, example): some stay, some go.
   Ask whether to touch a cell at all when it is already short; "leave unchanged" is a valid answer.
3. **Draft** the proposal in the scratchpad (`p_cell<NN>.py` / `.md`).
4. **Verify** it before inserting (see *Verification*).
5. **Insert** it directly below the original (see *Proposing*). A cell split in two: the
   replacement goes below the original, a *new-cell* proposal below that.
6. **Report**: VS Code cell number of the proposal, checks run, what changed, what was removed,
   then ask to **confirm / reject / change**. Stop and wait.
7. **On confirm**: re-read the saved proposal (the user usually edited it), diff it against your
   draft and show those edits, re-test if code changed, back up the notebook, then apply the
   proposal's *current* source, so the user's version lands, not the draft (see *Proposing*).
   Check afterwards: same cell id, marker gone, compiles, no `proposed` cell left.
8. **On reject**: delete the proposal; the original stays untouched.

Paired cells (a markdown cell and the code it describes) may be proposed together; the user can
still confirm them separately.

## Never

- **Self-check cells** (any section titled "…self-check", incl. §9.1's closing self-check): never
  propose changes to their code; their markdown only if the user asks.
- **Checks inside working cells** (`assert`, `raise`): keep condition and message by default.
  A change to them is a separate question to the user, never part of a proposal unasked.
- Edit a cell before the user confirmed, re-run a spec, or "repair" cells back towards a spec.
- Change CLAUDE.md or the specs in passing; collect what they will need (e.g. a changed loading
  convention) and list it for the user, who updates CLAUDE.md later.

## Style

- **Beginner/intermediate friendly**: comprehensions stay, nested ones included. A deep chain
  becomes named steps: more than ~3 chained calls, or a chain that hides a filter, condition or
  lookup in the middle, e.g. `s.reindex(idx).where(mask).groupby(day).agg("max").to_numpy()`.
  Named intermediate values (`missing_columns`, `only_in_errors`) over inline arithmetic in messages.
- **Comments**: one short line per block, no "why" essays; the user trims anything longer. Keep the
  user's own comments in their wording.
- **Config dicts** (settings the team edits): aligned trailing comments, one entry per line,
  trailing commas. `make format` would collapse the alignment; that is accepted.
- **Other code**: black, lines ≤ 88. Black never wraps comments or strings, so shorten those by
  hand; when black breaks an `assert` mid-expression, compute the condition into a named value first.
- **Loop variables leak** into the notebook's globals: pick names nothing else uses
  (`group_name`, not `group`/`columns`), and run the leak check (see *Verification*).
- When a settings cell also holds helpers or display output, offer to split it (settings cell
  first, helpers/output below); short cells usually stay one.
- **Markdown text** follows the `interpretation-style` skill.

## Verification (before every insert)

Run everything in a scratch Python process from the notebook's folder (relative paths then
behave as in Jupyter), never in the user's kernel.

- **Compare old and new**: execute every code cell above the original in one namespace
  (`display` replaced by a no-op, output captured). Then run the original and the proposal each
  in its own copy of that namespace, with deep copies of DataFrames, dicts and lists, so a cell
  that changes data in place cannot affect the other run. Compare:
  - every name either side defines: identical (pandas via `.equals` plus equal dtypes)
  - names only one side defines: intended?
  - objects from earlier cells that either side changed in place (e.g. new columns): identical
  - the printed output
- **Leak check**: with `ast`, list the names the proposal assigns that the original does not.
  Flag every later code cell that reads one of them without assigning it itself: that cell
  would silently pick up the proposal's value.
- Functions: call old and new on a handful of edge inputs and compare (e.g. `describe` on 20
  durations/offsets/non-durations).
- Checks: build the failure inputs (missing file, renamed column, stale export, broken configs) and
  confirm old and new stop on the same ones.
- Downstream: run the cells up to the next self-check with the proposal and confirm it still passes.
- `uv run black --check` and a line-length scan (`awk 'length > 88'`); config cells are exempt
  (their aligned comments are intended).

## Finish

- When the scope is done, ask the user to run the notebook top to bottom and report the self-check
  output. Commit only when asked, per the `commit-style` and `git-workflow` skills.
- Hand over the collected CLAUDE.md / spec follow-ups (see *Never*).

## Proposing

Every notebook write is a small Python step with `nbformat` (read, change, `nbformat.write`),
after copying the notebook to the scratchpad. Cells are found by **id**, never by position.

- **Markers**: a proposal carries the tag `proposed` and starts with a marker line:
  - replacing a cell: code `# PROPOSED: replaces the cell above`, markdown
    `> **PROPOSED**: replaces the cell above` plus a blank line
  - a new cell (e.g. from a split): `# PROPOSED: new cell` / `> **PROPOSED**: new cell`
- **Insert**: create the cell with `nbformat.v4.new_code_cell` / `new_markdown_cell` (marker line +
  draft, tag `proposed`) and insert it right after the original. First check that the cell types
  match and that no proposal already sits below the original.
- **Apply** (on confirm): check the marker line is still there, copy the proposal's source minus
  the marker into the original cell (the original keeps its id, which matters for nbdime merges),
  then delete the proposal. A new-cell proposal instead loses its marker and its `proposed` tag
  and stays in place.
- **Reject**: delete the proposal cell.

## Mechanics

- VS Code numbers cells from **1**; `nb["cells"]` from 0. Always say "VS Code cell N (§x.y)". A
  proposal shifts every later number by one while it exists.
- Notebooks of this size are too big for Read/NotebookEdit: read them with `json`, write with
  `nbformat` (see *Proposing*).
- The user has the notebook open in VS Code: on a "file changed on disk" prompt they should reload,
  and their unsaved edits to a proposal are only seen once they save.
