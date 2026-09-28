# 07 — Hybrid Linear Stage: Trend Extrapolation Diagnosis

- Status: **draft, parked**. A note for later, not yet run. Run only when a team member explicitly
  asks.
- Branch: `feature/*` off `main`
- Deliverable: changes to `notebooks/05_modeling/regression-models-claude.ipynb` (reference),
  applied cell by cell through the proposal-cell workflow (`notebook-refactor` skill). The notebook
  is a team-edited spec 06 output, so every change is proposed below the original cell and applied
  only on confirmation.
- Depends on: [06-regression-models.md](06-regression-models.md) Behaviour 17 (the hybrid) and the
  scoreboards of its §7.

## Goal

In the first full run of spec 06 (2026-09-29, test window 2025-09-07 .. 2026-09-06), the **direct
boosters beat the hybrids on overall MAE**, but the **hybrids beat the direct boosters in the
tails**. This spec finds out why, and whether a change to the hybrid's linear stage removes the
overall gap without losing the tail advantage.

The working hypothesis is that the cause is the linear stage's **trend term**, not the feature set.
Booster features are shared by both architectures, so adding features cannot explain or close a gap
between them.

## Evidence from the first run

Configuration: 13 booster features (all `USE_FEATURE` groups on), identical frozen booster configs
for direct and hybrid (`learning_rate 0.05`, `n_estimators 300`, `num_leaves 31` / `max_depth 4`).

**Scoreboard A, test MAE (MWh):**

| | static | rolling |
|---|---|---|
| xgb_direct | 2,408 (bias −110) | 2,398 (bias +133) |
| xgb_hybrid | 2,654 (bias **−1,393**) | 2,420 (bias −180) |
| lgbm_direct | 2,441 (bias −85) | 2,412 (bias +71) |
| lgbm_hybrid | 2,716 (bias **−1,467**) | 2,484 (bias −269) |
| smard | 2,790 (bias +414) | |

- The gap is large only under **static**. Rolling (refit every 30 days) shrinks it to 22–72 MWh.
- Static hybrid's bias is about −1,500 in the *ordinary* bin too (Scoreboard B), so it is a level
  shift over the whole year, not a tail effect.
- The validation year shows the same ranking (validation MAE: direct 2,185 / 2,205, hybrid
  2,330 / 2,328), so this is not a test-year accident.

**Scoreboard B, bottom 1 % by actual (≤ −7,844 MWh, 88 h):**

| | MAE | bias |
|---|---|---|
| xgb_direct static / rolling | 6,779 / 5,598 | +6,779 / +5,598 |
| lgbm_direct static / rolling | 6,622 / 4,741 | +6,622 / +4,741 |
| xgb_hybrid static / rolling | 3,029 / 3,441 | +2,442 / +3,011 |
| lgbm_hybrid static / rolling | **2,985** / 3,595 | +2,234 / +3,060 |
| smard | 3,952 | +3,322 |

- The direct boosters miss every extreme-low hour upward (bias = MAE). Binned by forecast, the
  static direct boosters put **0 hours** in the bottom bin: they are capped at the training floor.
  The test minimum is −15,562 MWh, the static fit's training minimum −8,443 MWh.
- Binned by forecast, the hybrids put 90–115 hours in the bottom bin with a bias of about −4,100:
  when they forecast an extreme low, they overshoot. For the low risk flag this means more false
  alarms (hybrid) against missed events (direct).

## Hypotheses

- **H1 — trend extrapolation (main suspect).** `HybridRegressor` adds `trend` = days since the
  first training day, fitted by OLS and extrapolated without limit. The static fit is trained on
  2023-09-07 .. 2025-09-05 and then extrapolated up to 365 days past it. Rolling runs at most
  ~30 days past its training window, which fits the bias vanishing under rolling. The booster cannot
  correct the drift: none of its features grows with the trend's day count.
- **H2 — OLS slope on `fc_gen_wind_solar` steeper than −1.** This would also explain the hybrids'
  overshoot on forecast-bottom hours. Expected coefficients if the stage is well calibrated: about
  +1 for `fc_grid_load`, about −1 for `fc_gen_wind_solar`, trend near 0.
- **H3 — stage 2 under-regularised.** The hybrid's booster fits smaller, noisier residuals but
  shares the direct grid (4 configs), and both chose the same config. Least likely; tested last.

## Scope

### IN

1. **Diagnostic, no model change:** print the linear stage's coefficients (`linear_.coef_`,
   `linear_.intercept_`) for the static fit and for the first and last rolling fits, per hybrid
   model. Report the trend's total contribution at the end of the test year
   (`coef_trend × days`) next to static hybrid's bias. This tests H1 and H2 before anything is
   changed.
2. **Trend switch:** a `trend` parameter on `HybridRegressor` (default `True`, today's behaviour),
   and two new registry entries `lgbm_hybrid_notrend` / `xgb_hybrid_notrend` with `trend=False`.
   Separate entries rather than a global toggle, so the scoreboard shows both variants side by side
   on identical hours without re-running.
3. **Decision table** in the notebook's closing section, comparing hybrid (trend), hybrid
   (no trend) and direct on the criteria below.
4. Only if H1 is rejected by step 1: test H3 with a hybrid-only grid (more regularisation:
   `min_child_samples` / `min_child_weight`, fewer leaves). The rule "hybrid uses the direct grid"
   (spec 06 Behaviour, tuning) is then relaxed for the hybrids only, and the notebook says so.

### OUT

- **New booster features.** They would change direct and hybrid equally; feature work belongs to
  [05-feature-engineering.md](05-feature-engineering.md).
- More linear-stage inputs (`cap_total`, lags, calendar). Discussed and rejected: `cap_total` is
  collinear with the trend and `fc_gen_wind_solar`, lags are collinear with the SMARD forecasts,
  calendar effects are non-linear and belong to the booster.
- Replacing the trend with a bounded or damped trend. A possible follow-up only if "no trend"
  loses the tail advantage.
- Changing the static / rolling setup, the test window or the interval method.
- Editing §9's "Did we beat SMARD?" interpretation beyond adding the decision table. The team
  rewrites it after reading the results.

## Decision criteria

Measured on the common test hours, for both split methods:

| Criterion | Source | Aim |
|---|---|---|
| overall MAE | Scoreboard A | within ~1 % of the direct booster |
| bias, ordinary bin | Scoreboard B, by actual | near 0 under static (today about −1,500) |
| MAE, bottom 1 % by actual | Scoreboard B, by actual | stays clearly below direct (today about 3,000 vs 6,700 static) |
| hours and bias, bottom bin by forecast | Scoreboard B, by forecast | fewer false lows than today's hybrid, without dropping to 0 like direct |
| MAE of the day min | Scoreboard B, day extremes | no worse than today's hybrid |

Possible outcomes, to be reported as such, not smoothed over:

- **No-trend hybrid meets all rows:** it replaces the trend hybrid as the default hybrid; the
  trend entries stay in the registry, switched off.
- **Overall gap closes, tail advantage lost:** the trend was carrying the tail. Report it, keep
  both, and un-park the damped-trend follow-up.
- **Nothing changes:** H1 is rejected. Continue with step 4 (H3).

## Open questions

- Should the no-trend entries be temporary (removed once decided) or kept permanently in the
  registry?
- Is ~1 % the right tolerance for "overall MAE within reach of direct", given the hybrid's tail
  advantage is what matters for the risk flag?
- Should the choice between hybrid and direct eventually be made per risk direction, since the low
  tail favours the hybrid and ordinary hours favour direct?

## Acceptance criteria

- [ ] Not run automatically. The spec 06 notebook stays unchanged until a team member un-parks
      this spec.
- [ ] Step 1's coefficients are reported before any model change, with a statement of whether they
      support H1 and H2.
- [ ] `HybridRegressor(trend=True)` reproduces today's hybrid forecasts exactly (same seed, same
      config).
- [ ] The no-trend entries run under both split methods and appear in every scoreboard on the same
      common hours.
- [ ] The decision table covers every criterion above, and the outcome is named as one of the three
      listed.
- [ ] All existing self-checks and the leakage test still pass.
