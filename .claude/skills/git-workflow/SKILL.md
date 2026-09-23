---
name: git-workflow
description: This repo's git workflow — feature branches, commits, pushes, rebasing onto main,
  pull requests and merge conflicts. Use whenever branching, committing, pushing, rebasing,
  opening a PR or resolving a merge conflict.
---

All work happens on `feature/*` branches off `main`. `main` is only ever updated through merged
pull requests, never by a direct commit or push, and never force-pushed.

Always let the user confirm when you run this skill unprompted.
If the user does not confirm, do not proceed on your own but link the user to the repo wiki
(https://github.com/fruchtgummikauer/grid-stress/wiki/GIT-Workflow) and do no git work at all for
the current task.
Git steps that are already part of another skill the user started (e.g. the per-section commits
in `spec-run-section-loop`) count as prompted.

## 1. Start a feature branch

Before switching, check `git status`. If the working tree is dirty, stop and ask the user what to
do with the changes; never stash, discard or carry them over silently.

```shell
# Make sure you are on main and up to date
git switch main
git pull

# Create and switch to a new feature branch
git switch -c feature/your-feature-name

# Stage and commit
git add notebook.ipynb
git commit -m "feat: describe what you did"

# Push and set upstream tracking (first time after creating the branch)
git push -u origin feature/your-feature-name
```

- Only start a new branch for new work. On an existing `feature/*` branch, commit there. If you
  are on `main` when asked to commit, stop and start a feature branch first.
- Branch names `feature/<short-kebab-case-description>`; stage files by name; commit messages
  follow the `commit-style` skill.
- Ask before pushing; pushing is visible to the whole team. Use `-u` on the first push only, a
  plain `git push` after that.

## 2. Update the branch with the latest main (rebase)

**Only when the user asks for it**, never on your own initiative.

```shell
git switch feature/your-feature-name
git fetch origin
git rebase origin/main
git push --force-with-lease  # required after rebase; never --force or -f
```

- Conflicts come up one commit at a time: resolve, `git add` the resolved files,
  `git rebase --continue`, repeat. For `.ipynb` conflicts run `uv run nbdime mergetool`, never
  edit the JSON by hand; it opens a browser UI, so wait until the user has resolved it. If it
  fails, the clone needs `uv run nbdime config-git --enable` once (see CLAUDE.md).
- If the rebase goes wrong, `git rebase --abort` returns the branch to where it started. Offer
  this rather than improvising a fix.
- **The force push needs two separate confirmations from the user.** First ask whether to
  force-push, naming the branch and the number of rewritten commits. After a yes, ask a second
  time, stating that it overwrites the remote branch and that teammates who pulled it must
  re-sync. Only push after the second yes.

## 3. Pull requests

**Only when the user asks for it.**

```shell
# After the user merged the PR: update main
git switch main
git pull

# The user deletes the remote branch in the PR ("Delete branch"); Claude never does

# Delete local branch
git branch -d feature/your-branch-name

# Clean up stale remote-tracking references (optional but tidy)
git fetch --prune
```

- Open the PR against `main` (`gh pr create --base main`). The title follows the `commit-style`
  skill; the body explains why the change exists and which spec it implements, if any.
- Only the user merges PRs on GitHub. Claude never merges a PR (not via `gh pr merge` either).
- Clean up only after the user confirms the PR is merged. Delete the local branch with `-d`,
  never `-D`; if `-d` refuses (common after a squash merge), tell the user why and ask before
  forcing it.
