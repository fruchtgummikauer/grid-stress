# ⚡ Grid Stress

Day-ahead forecasting of the **residual load** in the German power grid from public
[SMARD](https://www.smard.de) data, to flag the days when the grid is at risk of transmission
system operator (TSO) intervention.

Capstone project of the neuefische Data Science bootcamp.

## Problem

The **residual load** is the electricity demand left after wind and solar generation:
`grid load − (wind + solar)`. It must be covered by conventional plants, storage or imports.
Its extremes are where the grid gets stressed:

- **High residual load** — little wind and sun, high demand: imports and tight reserve margins.
- **Negative residual load** — renewables exceed demand: negative prices and downward redispatch.

A better day-ahead forecast of the residual load means earlier warning of these days.

## Approach

- **Benchmark:** SMARD's own `Forecast Residual Load`.
- **Model Output Statistics:** our models post-process SMARD's component forecasts (grid load,
  wind + solar) with actuals, calendar and capacity features. The claim is *"we reduce SMARD's
  error by X %"*, not *"we forecast better than the TSOs"*.
- **Forecast setting:** the forecast is issued at 18:00 on the day before (when SMARD's wind and
  solar forecast is published); actuals arrive with a 2 h lag, so the data cutoff is 16:00.
- **Scope:** generation features are wind and solar only, the only sources with published
  day-ahead forecasts and the largest variable ones.
- **Models:** seasonal naive baseline, Ridge, LightGBM, XGBoost and random forest (direct and
  hybrid variants), plus ensembles of these. Model decisions are made on a validation year; a
  separate test year only confirms them.
- **Risk labels:** high and low risk days are two independently thresholded directions, using a
  trailing 365-day quantile of the residual load.
- **Cost:** after modelling, the advantage over SMARD is priced at the imbalance price (reBAP).

## Data

| Source | Content | Access |
| --- | --- | --- |
| [SMARD](https://www.smard.de) (Bundesnetzagentur) | hourly grid load, generation, forecasts and installed capacity since 2019 | public API, no key |
| [netztransparenz.de](https://www.netztransparenz.de) | reBAP imbalance price, 15 min | free account, OAuth2 (see below) |

`data/` is gitignored. Only the trained model saves in `data/models/<model_key>/` are committed.
All other files are produced by the notebooks (see [Reproduce the pipeline](#reproduce-the-pipeline)).

SMARD data: [terms of use](https://www.smard.de/en/datennutzung), CC BY 4.0.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). Do not use `pip install`.
Details: [UV_SETUP.md](UV_SETUP.md).

```bash
make setup      # uv sync --all-groups
make lab        # Jupyter Lab
make format     # black
```

Add dependencies with `uv add <pkg>` (runtime) or `uv add --group dev <pkg>` (notebooks, tooling).

### reBAP credentials (netztransparenz.de)

The reBAP data needs a free account on the
[netztransparenz API portal](https://api-portal.netztransparenz.de).

1. [Register](https://api-portal.netztransparenz.de/registration) your own account.
2. Create a new client under [my clients](https://api-portal.netztransparenz.de/my-clients).

> [!Warning]
> Copy the `client secret` right away. You cannot see it again later.

3. Create your environment file and fill in the client id and secret:

```bash
cp .env.example .env
```

> [!CAUTION]
> `.env` holds credentials and must never be committed (it is in `.gitignore`).
> Never paste credentials into a notebook.

## Reproduce the pipeline

Run the notebooks in this order. Each one writes its exports to `data/` when `EXPORT_ENABLED` is
on.

| Step | Notebook | Writes |
| --- | --- | --- |
| 1. Download data | [API-connection.ipynb](notebooks/API-connection.ipynb) | `data/smard.csv`, `data/rebap.csv` |
| 2. Risk labels | [risk-definition.ipynb](notebooks/03_risk_classification/risk-definition.ipynb) | `data/risk_classification/risk_labels_*.csv` |
| 3. SMARD benchmark | [forecast-metrics-claude.ipynb](notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb) | `data/metrics/smard_*.csv` |
| 4. Models | [regression-models-claude.ipynb](notebooks/05_modeling/regression-models-claude.ipynb) | `data/models/model_*.csv`, model saves |
| 5. Ensemble | [ensemble-claude.ipynb](notebooks/05_modeling/ensemble-claude.ipynb) | `data/models/ensemble_*.csv` |
| 6. Best models vs SMARD | [visualization-01-regression-best-models.ipynb](notebooks/05_modeling/visualization-01-regression-best-models.ipynb) | – |
| 7. Risk labels on the models | [visualization-02-classification-risk-labels.ipynb](notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb) | `data/risk_classification/model_risk_labels_*.csv` |
| 8. reBAP cost | [visualization-03-rebap-cost.ipynb](notebooks/05_modeling/visualization-03-rebap-cost.ipynb) | `data/models/model_rebap_cost_hourly.csv` |

Notes:

- The download range is set by `START` / `END` in `API-connection.ipynb` (`END` is exclusive).
- After a new download, re-run **every** later step. The visualization notebooks stop when their
  inputs come from different runs.
- Models notebook: per model, `load_saved` fits from scratch (`False`), refits the saved tuning
  (`"config"`, works on any data) or loads the saved results (`"results"`, same data snapshot
  only). A full fit takes about 30 min on 16 cores. `SAVE_MODELS = True` writes new saves.
  `USE_GPU` runs XGBoost on a GPU if one is found (results then differ slightly from CPU).
- The ensemble reads the model saves (no refit) and takes about 4 min.

## Streamlit app

An interactive tour of the project: when the grid is under pressure, how we forecast, whether we
beat SMARD, a page to try it yourself, what it is worth, and who we are.

Run it **from the repo root**, so the theme in `.streamlit/config.toml` is found:

```bash
uv run streamlit run streamlit/streamlit_app.py
```

The app reads the exports in `data/`, so run the [pipeline](#reproduce-the-pipeline) first. The
*Who are we?* page has a data check that lists every file the app needs, whether it was found, and
which notebook produces it.

## Repository structure

```text
notebooks/
  API-connection.ipynb        data download (SMARD + reBAP)
  01_eda/                     exploratory analysis (team-EDA.ipynb is the shared one)
  02_forecast_metrics/        SMARD's forecast errors: the benchmark
  03_risk_classification/     risk day definition and labels
  04_feature_engineering/     feature exploration and documentation
  05_modeling/                regression models, ensemble, result visualizations
streamlit/                    Streamlit app (pages in app_pages/, shared parts in components/)
data/                         gitignored, except the model saves in data/models/<model_key>/
```

Notebook name suffixes: no suffix = adopted by the team; `-claude` = reference notebook generated
from a spec in `.claude/specs/`; `-<name>` = a team member's own exploration.

## Contributing

- Work on `feature/*` branches off `main` and open a pull request.
- Notebooks are JSON, so resolve merge conflicts with **nbdime** (in the `dev` group):

```bash
nbdime config-git --enable
```

Enable it once per clone. When a conflict occurs, open the merge tool and pick the cells side by
side:

```bash
nbdime mergetool
```

Then `git add` the notebook and commit.

## Team

Robert, Marco, Hari and Monica, at the [neuefische](https://www.neuefische.de) Data Science
bootcamp, with [Claude Code](https://claude.com/claude-code) as the intern.

## License

[MIT](LICENSE)
