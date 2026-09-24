# AtlasFeature Library

An **optional, experimental** GridStress feature reference for the team to review. Open the [interactive HTML](index.html) to browse feature definitions, runnable recipe examples, proposed models, and descriptive correlation heatmaps.

## Open it in a browser

GitHub's file preview shows the HTML source. To use the interactive page **now**, download `index.html` from GitHub and open the downloaded file in a modern browser with JavaScript enabled. The interface works offline; the notebook links inside it need an internet connection.

If the repository owner chooses to publish this proposal, GitHub Pages can host it directly at `https://fruchtgummikauer.github.io/grid-stress/AtlasFeatureLibrary/` **after** this branch is merged and Pages is configured to serve `main` from `/docs`. That URL is a proposed future address, not a live site.

## Sources and limits

- [Hari's feature-engineering notebook](https://github.com/fruchtgummikauer/grid-stress/blob/feature/pm-session2-hari/notebooks/04_feature_engineering/Hari_Gridstress_feature_engineering_baselines_metrics.ipynb) supplies the original calculation methods. The Atlas includes one additional feature marked as a proposal.
- [Hari's EDA notebook](https://github.com/fruchtgummikauer/grid-stress/blob/feature/pm-session2-hari/notebooks/01_eda/EDA-hari.ipynb) is related background.
- The 2024 labels on the Pearson and Spearman matrices identify **8,784 hourly target timestamps in the correlation sample**, not a version or expiry year for the feature recipes. They summarize pairwise association, not feature importance, model improvement, or causal effects.
- The embedded analysis used a separately supplied hourly MW extract. Its historical publication vintages are unverified; its saved correlations have **not** been recalculated from this repository's `data/smard.csv`. No raw CSV or credentials are included in this folder.
- Source-notebook model results are historical notebook outputs. Proposed Atlas models remain unscored and are not adopted team models.

This is a static, read-only snapshot. Updating a feature or correlation requires regenerating and reviewing the HTML; opening the page never writes back to GitHub or retrains a model.
