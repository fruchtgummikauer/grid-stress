# AtlasFeature Library

An **optional, experimental** GridStress feature reference for the team to review. Open the [interactive HTML](index.html) to browse feature definitions, runnable recipe examples, proposed models, and descriptive correlation heatmaps.

## Open it in a browser

GitHub's file preview shows the HTML source. To use the interactive page **now**, download `index.html` from GitHub and open the downloaded file in a modern browser with JavaScript enabled. The interface works offline; the notebook links inside it need an internet connection.

If the repository owner chooses to publish this proposal, GitHub Pages can host it directly at `https://fruchtgummikauer.github.io/grid-stress/AtlasFeatureLibrary/` **after** this branch is merged and Pages is configured to serve `main` from `/docs`. That URL is a proposed future address, not a live site.

**Finding your way:** The **Feature table** is the overview of forecast candidate columns and recipes. **Variables & targets** explains inputs, observed calculations, and labels. **Models & tuning** lists recorded or proposed settings. The **Correlation map** measures pairwise co-movement in a stated sample; the **Relationship lab** illustrates forward-fold tests of added value and combinations. Its one measured pair example does not rank all features. **Team guide** explains how to contribute to each category. The same mapping is available under **What belongs in each Atlas section?** immediately below the page tabs.

## Add variables, features, and models

In the downloaded HTML, open **Team guide · add variables, targets, features & models**. It has five links: **Add source variable**, **Add derived observed variable**, **Add target / label**, **Add forecast feature**, and **Add model**. Each opens a separate, copyable contribution guide. These are instructions, not forms that save changes.

- **Source variable:** Define the input column's origin, units (for power series, average MW), UTC event time, cadence, missing values, and historical availability. Extend and validate the Python ingestion schema before using the column in a feature.
- **Derived observed variable:** Specify its formula and parents, implement the calculation, and record its units and missing-value rules. A value calculated from observations at the target time is descriptive; to use it in a forecast, create a separate recipe that only uses information available at the forecast origin.
- **Target / label:** Define an outcome using observations at the UTC target time, record its formula under `catalog/evidence.json` → `target_only_outcomes`, and keep it out of predictor columns. The existing names `stress_high`, `stress_low`, and `stress_negative` retain their current definitions. Freeze any fitted threshold on earlier eligible data before forward validation, then measure prevalence and model scores afresh. The current team-model registry accepts only `residual_load_mw` regression or `stress_any` classification; other outcomes need a reviewed code and evaluation extension.
- **Forecast feature:** Where its parents are already supported forecast-time feature columns, add a reviewed function to `gridstress_features/team_recipes.py` and a matching card to `catalog/team_features.json`. New IDs start as `PROPOSED_UNSCORED`. A new raw input requires ingestion work first.
- **Model:** Declare a proposal in `catalog/team_models.json`; fitting, tuning, and reporting scores require a separate recorded evaluation.

**Contribution status:** This pull request contains the read-only HTML snapshot and documentation, not the editable Python package or JSON registries mentioned above. The team would need an owner-reviewed source-code contribution before those instructions can be followed from a repository checkout. The package's refresh command recomputes eligible *feature-to-feature* correlations from an explicitly supplied CSV, then regenerates the HTML and notebook. It does not currently rebuild the *observed-variable* correlation matrix, calculate a new target / label, train or rescore models, or choose question recommendations. New source or derived observed variables need a separately validated matrix update. Nothing recalculates by clicking the HTML.

## Ask Claude to add something

Give Claude Code **both** your extracted `GridStress_Feature_Atlas` source bundle and a separate checkout of `grid-stress`. A checkout of this PR alone cannot create runnable features or regenerate correlations. In the HTML Team guide, expand **Ask Claude to help add or update an Atlas item** and use **Copy Claude request**. Fill in its brackets before pasting. The request asks for the exact new ID, source columns, formula and units, UTC timestamp and cadence, issue-time availability, missing-value rule, question, and the path to a qualified hourly MW CSV if one exists.

| Desired addition | Edit in the source bundle | Check before publishing |
|:--|:--|:--|
| Source or derived observed variable | `feature_supermarket/recipes.py`, `catalog/evidence.json` | Verify units, timestamps and missingness; recalculate `catalog/observed_variables_matrix.json` separately if measured rows support it. |
| Target / label | `catalog/evidence.json` → `target_only_outcomes`, plus reviewed outcome and evaluation code | Preserve existing label names; freeze fitted thresholds on earlier data; exclude realized labels from predictors. |
| Forecast feature using supported parents | `gridstress_features/team_recipes.py`, `catalog/team_features.json` | Test available-at timing and the new function; begin at `PROPOSED_UNSCORED`. New raw inputs also need ingestion work. |
| Model proposal | `catalog/team_models.json` | Record fixed versus proposed tuning settings; do not claim scores until a separately evaluated experiment exists. |
| HTML wording or navigation | `web/index.html` | Run `python scripts/build_site.py` and replace the generated HTML; verify anchors and controls. |

Claude should run `python -m unittest discover -s tests -v` in the source bundle. When you provide an hourly average-MW CSV with the required columns, every UTC hour of the requested year and adequate earlier history for lookbacks, Claude can run `python scripts/refresh_atlas.py --csv /absolute/path/to.csv --year YYYY`. That command rebuilds candidate-feature correlations and the HTML/notebook; it does not recalculate observed-variable correlations, generate new labels, choose question recommendations or train models. Review the source SHA, pair support, code and timing assumptions before copying `dist/GridStress_Feature_Atlas.html` into this branch's `docs/AtlasFeatureLibrary/index.html`. Keep the matching Python and catalog changes available for review so the page can be regenerated. The source CSV stays outside the published docs.

[Review the Atlas proposal and its updates in PR #29](https://github.com/fruchtgummikauer/grid-stress/pull/29). A copied `file://` link works only on the computer holding that HTML; a team-shareable Atlas page requires the repository owner's decision to host it.

## Sources and limits

- [Hari's feature-engineering notebook](https://github.com/fruchtgummikauer/grid-stress/blob/feature/pm-session2-hari/notebooks/04_feature_engineering/Hari_Gridstress_feature_engineering_baselines_metrics.ipynb) supplies the original calculation methods. The Atlas includes one additional feature marked as a proposal.
- [Hari's EDA notebook](https://github.com/fruchtgummikauer/grid-stress/blob/feature/pm-session2-hari/notebooks/01_eda/EDA-hari.ipynb) is related background.
- The 2024 labels on the Pearson and Spearman matrices identify **8,784 hourly target timestamps in the correlation sample**, not a version or expiry year for the feature recipes. They summarize pairwise association, not feature importance, model improvement, or causal effects.
- The embedded analysis used a separately supplied hourly MW extract. Its historical publication vintages are unverified; its saved correlations have **not** been recalculated from this repository's `data/smard.csv`. No raw CSV or credentials are included in this folder.
- Source-notebook model results are historical notebook outputs. Proposed Atlas models remain unscored and are not adopted team models.

This is a static, read-only snapshot. Updating a feature or correlation requires regenerating and reviewing the HTML; opening the page never writes back to GitHub or retrains a model.
