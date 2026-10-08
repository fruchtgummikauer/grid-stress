"""Model results for the Streamlit model pages — our forecasts against SMARD's.

Ported from `notebooks/05_modeling/visualization-01-regression-best-models.ipynb` §1.2–§2 (spec 09:
common hours, bins, full months, self-check, pick rules) and
`notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb` §1.2 / §3 (spec 10:
loading the label export, scoring flags) and the cost export of
`notebooks/05_modeling/visualization-03-rebap-cost.ipynb` (spec 11). Those notebooks stay the source of truth; this module
only reads their input and output files and adds Streamlit caching and "file not yet generated"
handling on top. It computes no new results: risk flags and thresholds come from spec 10's export,
never recomputed (CLAUDE.md).
"""

import json
from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd
import streamlit as st

from components.layout import current_page, next_page
from data_loading import _find_data_dir

MODEL_SOURCE = (
    "notebooks/05_modeling/regression-models-claude.ipynb with EXPORT_ENABLED = True"
)
SMARD_SOURCE = "notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb"
RISK_SOURCE = (
    "notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb "
    "with EXPORT_ENABLED = True (after the model exports)"
)
ENSEMBLE_SOURCE = "notebooks/05_modeling/ensemble-claude.ipynb with EXPORT_ENABLED = True (after the model saves)"
REBAP_SOURCE = (
    "notebooks/05_modeling/visualization-03-rebap-cost.ipynb with EXPORT_ENABLED = True "
    "(after the model and ensemble exports; needs data/rebap.csv)"
)
SOURCES = {
    "scoreboard": ("models/model_scoreboard.csv", MODEL_SOURCE),
    "hourly": ("models/model_forecast_errors_hourly.csv", MODEL_SOURCE),
    "smard": ("metrics/smard_forecast_errors_hourly.csv", SMARD_SOURCE),
    "risk_daily": ("risk_classification/model_risk_labels_daily.csv", RISK_SOURCE),
    "risk_hourly": ("risk_classification/model_risk_labels_hourly.csv", RISK_SOURCE),
    "ensemble_hourly": ("models/ensemble_forecast_errors_hourly.csv", ENSEMBLE_SOURCE),
    "ensemble_scoreboard": ("models/ensemble_scoreboard.csv", ENSEMBLE_SOURCE),
    "ensemble_weights": ("models/ensemble_weights.csv", ENSEMBLE_SOURCE),
    "rebap_cost": ("models/model_rebap_cost_hourly.csv", REBAP_SOURCE),
}
ENSEMBLE_EXPORTS = ["ensemble_hourly", "ensemble_scoreboard", "ensemble_weights"]
ENSEMBLE_PREFIX = "ensemble_"  # spec 08's model keys: ensemble_<method>
STAMP = "%Y-%m-%d %H:%M:%S"

# Pick settings, as in spec 09's settings cell
N_PICKS = 3  # top picks per category
MIN_BIN_SHARE = 0.5  # by-forecast hours / by-actual hours, tail categories
CANDIDATE_SPLIT = "rolling"  # only these rows compete

SMARD_ROW = ("smard", "none")
CATEGORIES = ["overall", "ordinary", "below_zero", "low_extreme", "high_extreme"]
BIN_CATEGORIES = CATEGORIES[1:]
BINS = ["low_extreme", "ordinary", "high_extreme"]
TAIL_LEVELS = [0.01, 0.25, 0.75, 0.99]
ZERO = 0.0  # MWh: below_zero bin

# Risk labels, as in spec 10's settings cell
BASES = {"high": ["rolling"], "low": ["rolling", "zero"]}
RULES = ["any", "3h"]
SOURCES_SCORED = ["model", "smard"]
OUTCOMES = ["hit", "miss", "false alarm", "quiet"]
PERSISTENCE = pd.Timedelta("3h")  # the `3h` day rule, as a duration


def export_stamp(name):
    """Modification time of export `name`; raises `FileNotFoundError` naming its notebook."""
    relative, producer = SOURCES[name]
    path = _find_data_dir() / relative
    if not path.exists():
        raise FileNotFoundError(
            f"`data/{relative}` not found. data/ is gitignored — create it by running "
            f"{producer}."
        )
    return path.stat().st_mtime


def load_export(name):
    """One file as a DataFrame.

    Raises `FileNotFoundError` naming the producing notebook. Cached on the file's modification
    time, so a re-export shows without clearing the cache (Streamlit-v3.md §2.4).
    """
    stamp = export_stamp(name)
    return _read_export(str(_find_data_dir() / SOURCES[name][0]), stamp)


@st.cache_data
def _read_export(path, mtime):
    """The cached body of `load_export`; `mtime` is only the cache key."""
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], format=STAMP)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    return frame


def get_or_stop(builder):
    """Call a builder; on a missing file show why, link to the next page, and stop the page."""
    try:
        return builder()
    except FileNotFoundError as exc:
        st.info(f"Model results not available yet: {exc}")
        next_page(current_page())
        st.stop()


def get_or_info(builder, what):
    """Call a builder for one section; on a missing file show why and return None.

    The page goes on: use it for sections whose export is optional (Streamlit-v3.md §2.4).
    """
    try:
        return builder()
    except FileNotFoundError as exc:
        st.info(f"{what} not available yet: {exc}")
        return None


def is_set(flag):
    """True where a flag is set; False and empty both give False (spec 10 §2.2).

    A new mask — the label itself stays three-state, an empty flag means "not evaluable".
    """
    return pd.Series(flag.to_numpy(dtype=bool, na_value=False), index=flag.index)


# ============================================================================
# Accuracy (spec 09)
# ============================================================================
@dataclass
class Accuracy:
    """Everything the accuracy charts need, rebuilt as in spec 09 §1.3."""

    test_hours: pd.DatetimeIndex
    test_days: pd.DatetimeIndex
    actual: pd.Series  # on the test hours
    forecasts: dict  # row -> forecast on the test hours, SMARD first
    common: pd.DatetimeIndex  # hours with an actual and a forecast from every row
    errors: dict  # row -> forecast − actual on the common hours
    edges: dict  # "P1", "P25", "P75", "P99" in MWh
    full_months: pd.PeriodIndex
    monthly_mae: pd.DataFrame  # full months × rows
    value: pd.DataFrame  # scoreboard values, rows × metrics
    count: pd.DataFrame  # scoreboard counts, rows × metrics
    candidates: list
    picks: dict  # category -> picked rows, in rank order
    rejected: dict  # category -> {row: failed rules}
    bands: (
        dict  # row -> 95 % band (`lower`, `upper`) on the common hours, SMARD excluded
    )
    ensembles: list = field(
        default_factory=list
    )  # spec 08's rows, if their exports exist
    ensemble_pick: tuple | None = (
        None  # best CANDIDATE_SPLIT ensemble (spec 09's ENSEMBLE_PICK)
    )
    problems: list = field(default_factory=list)  # self-check failures

    def bin_masks(self, values):
        """One mask per bin of `values`; below_zero overlaps low_extreme (spec 09 §1.3)."""
        p1, p25, p75, p99 = (self.edges[k] for k in ("P1", "P25", "P75", "P99"))
        return {
            "low_extreme": values <= p1,
            "below_zero": values < ZERO,
            "ordinary": (values >= p25) & (values <= p75),
            "high_extreme": values > p99,
        }

    @property
    def bin_hours(self):
        """By-actual masks on the common hours."""
        return self.bin_masks(self.actual[self.common])

    def bin_rule(self, name):
        """The bin's edge in plain words, e.g. `at or below -6,512 MWh`."""
        e = self.edges
        return {
            "low_extreme": f"at or below {e['P1']:,.0f} MWh",
            "below_zero": "below 0 MWh",
            "ordinary": f"between {e['P25']:,.0f} and {e['P75']:,.0f} MWh",
            "high_extreme": f"above {e['P99']:,.0f} MWh",
        }[name]

    def forecast_share(self, row, name):
        """By-forecast hours / by-actual hours of a row in a bin."""
        by_actual = self.count.loc[row, f"MAE_{name}_by_actual"]
        return (
            self.count.loc[row, f"MAE_{name}_by_forecast"] / by_actual
            if by_actual
            else np.nan
        )

    def skill(self, row, by):
        """Skill vs SMARD (%) of a row, grouped by `by` (an index aligned to the common hours)."""
        smard = self.errors[SMARD_ROW].abs().groupby(by).mean()
        return 100 * (1 - self.errors[row].abs().groupby(by).mean() / smard)


def _failed_rules(value, count, row, category):
    """The rules a candidate fails in a category; empty if it qualifies (spec 09 §2)."""
    if category == "overall":
        return [] if value.loc[row, "skill_pct"] > 0 else ["skill ≤ 0"]
    failed = []
    for view in ["actual", "forecast"]:
        metric = f"MAE_{category}_by_{view}"
        if not value.loc[row, metric] < value.loc[SMARD_ROW, metric]:
            failed.append(f"by {view}")
    by_actual = count.loc[row, f"MAE_{category}_by_actual"]
    share = (
        count.loc[row, f"MAE_{category}_by_forecast"] / by_actual
        if by_actual
        else np.nan
    )
    if not share >= MIN_BIN_SHARE:
        failed.append(f"forecast share {share:.2f} < {MIN_BIN_SHARE}")
    return failed


def _rank_key(value, row, category):
    """Sort key: lowest average miss first, ties by model key.

    Overall by MAE on all hours (team decision 2026-10-07; spec 09 ranked months beating SMARD
    first), a tail by its MAE by actual (spec 09 §2).
    """
    if category == "overall":
        return (value.loc[row, "MAE"], row[0])
    return (value.loc[row, f"MAE_{category}_by_actual"], row[0])


def is_ensemble(row):
    """True for a spec 08 ensemble row."""
    return row[0].startswith(ENSEMBLE_PREFIX)


def load_accuracy() -> Accuracy:
    """`Accuracy` of the model exports (plus spec 08's ensembles, if exported), cached on its
    input files."""
    names = ["scoreboard", "hourly", "smard"]
    with_ensembles = ensemble_ready()
    if with_ensembles:
        names += ["ensemble_hourly", "ensemble_scoreboard"]
    return _load_accuracy(tuple(export_stamp(name) for name in names), with_ensembles)


@st.cache_data
def _load_accuracy(stamps, with_ensembles=False) -> Accuracy:
    """Build `Accuracy` from the model and SMARD exports, identically to spec 09 §1.3–§2.

    Spec 08's ensembles join as extra rows (as viz-01 adds them): never candidates, scored on the
    same common hours, and only if the ensemble scoreboard's member and SMARD rows equal
    `model_scoreboard.csv`.
    """
    scoreboard = load_export("scoreboard")
    hourly = load_export("hourly")
    smard = load_export("smard").set_index("timestamp")

    test_hours = pd.DatetimeIndex(hourly["timestamp"].unique()).sort_values()
    test_days = test_hours.normalize().unique()
    actual = hourly.groupby("timestamp")["residual_load"].first().reindex(test_hours)

    # One column per (model, split_method) row; SMARD first, then every row with a forecast
    forecast = hourly.pivot(
        index="timestamp", columns=["model", "split_method"], values="forecast"
    )
    error = hourly.pivot(
        index="timestamp", columns=["model", "split_method"], values="err_residual_load"
    )
    forecasts = {SMARD_ROW: smard["fc_residual_load"].reindex(test_hours)}
    forecasts |= {
        row: forecast[row].reindex(test_hours)
        for row in forecast
        if forecast[row].notna().any()
    }

    has_all = actual.notna() & pd.concat(forecasts, axis=1).notna().all(axis=1)
    common = test_hours[has_all.to_numpy()]
    errors = {SMARD_ROW: smard.loc[common, "err_residual_load"]}
    errors |= {row: error.loc[common, row] for row in forecasts if row != SMARD_ROW}
    bands = {
        row: hourly_bands(hourly, row, common) for row in forecasts if row != SMARD_ROW
    }

    ensembles, ensemble_problems = [], []
    if with_ensembles:
        ensemble_hourly = load_export("ensemble_hourly")
        ensemble_board = load_export("ensemble_scoreboard")
        ensemble_forecast = ensemble_hourly.pivot(
            index="timestamp", columns=["model", "split_method"], values="forecast"
        )
        ensemble_error = ensemble_hourly.pivot(
            index="timestamp",
            columns=["model", "split_method"],
            values="err_residual_load",
        )
        for row in ensemble_forecast:
            forecasts[row] = ensemble_forecast[row].reindex(test_hours)
            errors[row] = ensemble_error[row].reindex(common)
            bands[row] = hourly_bands(ensemble_hourly, row, common)
            ensembles.append(row)
        is_member = ~ensemble_board["model"].str.startswith(ENSEMBLE_PREFIX)
        shared = ensemble_board[is_member].merge(
            scoreboard,
            on=["model", "split_method", "metric"],
            suffixes=("_ensemble", ""),
        )
        differs = shared[
            (shared["value_ensemble"] - shared["value"]).abs().gt(1e-6)
            | (shared["value_ensemble"].isna() != shared["value"].isna())
        ]
        if len(differs):
            ensemble_problems.append(
                f"the ensemble scoreboard's member rows differ from model_scoreboard.csv "
                f"({len(differs)} values, e.g. {tuple(differs.iloc[0][['model', 'split_method', 'metric']])}) "
                f"— re-run {ENSEMBLE_SOURCE}"
            )
        scoreboard = pd.concat([scoreboard, ensemble_board[~is_member]])

    quantiles = actual.quantile(TAIL_LEVELS).to_numpy()
    edges = dict(zip(["P1", "P25", "P75", "P99"], quantiles))

    # Full months: first and last day inside the test window; monthly MAE on the common hours
    full_months = pd.PeriodIndex(
        [
            month
            for month in test_days.to_period("M").unique()
            if month.start_time in test_days and month.end_time.normalize() in test_days
        ]
    )
    month_of = common.to_period("M")
    monthly_mae = pd.concat(
        {row: err.abs().groupby(month_of).mean() for row, err in errors.items()}, axis=1
    ).loc[full_months]

    value = scoreboard.pivot(
        index=["model", "split_method"], columns="metric", values="value"
    )
    count = scoreboard.pivot(
        index=["model", "split_method"], columns="metric", values="count"
    )

    candidates = [
        row
        for row in value.index
        if row[1] == CANDIDATE_SPLIT
        and row[0] not in ("smard", "seasonal_naive")
        and not is_ensemble(row)
    ]
    picks, rejected = {}, {}
    for category in CATEGORIES:
        failed = {row: _failed_rules(value, count, row, category) for row in candidates}
        qualified = sorted(
            (row for row in candidates if not failed[row]),
            key=lambda row: _rank_key(value, row, category),
        )
        picks[category] = qualified[:N_PICKS]
        rejected[category] = {row: rules for row, rules in failed.items() if rules}

    accuracy = Accuracy(
        test_hours=test_hours,
        test_days=test_days,
        actual=actual,
        forecasts=forecasts,
        common=common,
        errors=errors,
        edges=edges,
        full_months=full_months,
        monthly_mae=monthly_mae,
        value=value,
        count=count,
        candidates=candidates,
        picks=picks,
        rejected=rejected,
        bands=bands,
        ensembles=ensembles,
    )
    # Spec 09's ensemble pick: the best CANDIDATE_SPLIT ensemble by the overall ranking; shown as
    # a headline only if it beats SMARD
    qualified = [
        row
        for row in ensembles
        if row[1] == CANDIDATE_SPLIT and not _failed_rules(value, count, row, "overall")
    ]
    if qualified:
        accuracy.ensemble_pick = min(
            qualified, key=lambda row: _rank_key(value, row, "overall")
        )
    accuracy.problems = ensemble_problems + _self_check(accuracy, smard)
    if not candidates:
        accuracy.problems.append(
            f"no {CANDIDATE_SPLIT!r} rows in the model exports — re-run {MODEL_SOURCE}"
        )
    return accuracy


def hourly_bands(hourly, row, hours):
    """A row's 95 % band (`lower`, `upper`) from an hourly export, on `hours`."""
    rows = hourly[(hourly["model"] == row[0]) & (hourly["split_method"] == row[1])]
    return rows.set_index("timestamp")[["lower", "upper"]].reindex(hours)


def coverage(acc, row, hours=None):
    """Share of `hours` (default: the common hours) whose actual lies inside the row's band (%)."""
    band = acc.bands[row] if hours is None else acc.bands[row].loc[hours]
    actual = acc.actual[band.index]
    return 100 * float(((actual >= band["lower"]) & (actual <= band["upper"])).mean())


def _self_check(acc, smard):
    """Spec 09 §1.4, collecting failures instead of stopping: one message per problem."""
    regen_models = f"data/models/ ({MODEL_SOURCE})"
    regen_all = f"data/metrics/ ({SMARD_SOURCE}), then {regen_models}"
    problems = []

    gap = float(
        (acc.actual[acc.common] - smard.loc[acc.common, "residual_load"]).abs().max()
    )
    if gap > 1e-6:
        problems.append(
            f"residual_load differs between the two hourly files by up to {gap:,.2f} MWh — regenerate {regen_all}"
        )

    row_gap = sorted(set(acc.value.index) ^ set(acc.errors))
    if row_gap:
        problems.append(
            f"rows {row_gap} are in only one of the two model exports — regenerate {regen_models}"
        )

    for row, err in acc.errors.items():
        if row not in acc.value.index:
            continue
        mae, hours = err.abs().mean(), len(err)
        if (
            abs(mae - acc.value.loc[row, "MAE"]) > 1e-6
            or hours != acc.count.loc[row, "MAE"]
        ):
            problems.append(
                f"{row}: MAE {mae:,.3f} on {hours:,} h, scoreboard {acc.value.loc[row, 'MAE']:,.3f} "
                f"on {acc.count.loc[row, 'MAE']:,.0f} h — regenerate "
                f"{regen_all if row == SMARD_ROW else regen_models}"
            )

    for name, in_bin in acc.bin_hours.items():
        exported = acc.count[f"MAE_{name}_by_actual"]
        if not (exported == in_bin.sum()).all():
            problems.append(
                f"{name}: {in_bin.sum():,} by-actual hours disagree with the scoreboard — regenerate {regen_models}"
            )

    for row in acc.errors:
        if row == SMARD_ROW or row not in acc.value.index:
            continue
        months = int((acc.monthly_mae[row] < acc.monthly_mae[SMARD_ROW]).sum())
        exported = (
            acc.value.loc[row, "months_beating_smard"],
            acc.count.loc[row, "months_beating_smard"],
        )
        if (months, len(acc.full_months)) != exported:
            problems.append(
                f"{row}: {months} / {len(acc.full_months)} months beating SMARD, scoreboard {exported[0]:.0f} / {exported[1]:.0f} — regenerate {regen_models}"
            )
    return problems


# ============================================================================
# Risk labels (spec 10)
# ============================================================================
@dataclass
class RiskLabels:
    """Spec 10's label export: one row per test day / per common test hour."""

    daily: (
        pd.DataFrame
    )  # indexed by date; flags three-state `boolean`, ranges as datetimes
    hourly: pd.DataFrame  # indexed by timestamp; flags three-state `boolean`
    picks: dict  # direction -> model key (`high_model` / `low_model`)
    problems: list = field(default_factory=list)

    def flag(self, source, direction, basis, rule):
        """A day flag column, e.g. `model_high_risk_rolling_any`."""
        return self.daily[f"{source}_{direction}_risk_{basis}_{rule}"]

    def hour_flag(self, source, direction, basis):
        """An hour flag column, e.g. `smard_low_risk_hour_zero`."""
        return self.hourly[f"{source}_{direction}_risk_hour_{basis}"]


def load_risk_labels() -> RiskLabels:
    """Spec 10's risk labels, cached on the export files and the model exports they are checked
    against."""
    stamps = tuple(
        export_stamp(name)
        for name in ("risk_daily", "risk_hourly", "scoreboard", "hourly", "smard")
    )
    return _load_risk_labels(stamps)


@st.cache_data
def _load_risk_labels(stamps) -> RiskLabels:
    """Read spec 10's export with three-state flags (an empty flag is "not evaluable")."""
    daily = load_export("risk_daily").set_index("date")
    hourly = load_export("risk_hourly").set_index("timestamp")

    for frame in (daily, hourly):
        flags = [c for c in frame.columns if "_risk_" in c]
        frame[flags] = frame[flags].astype("boolean")
    for column in [c for c in daily.columns if "_range_" in c]:
        daily[column] = pd.to_datetime(daily[column], format=STAMP)
    daily["evaluable"] = daily["evaluable"].astype(bool)

    picks = {d: daily[f"{d}_model"].iloc[0] for d in ("high", "low")}
    labels = RiskLabels(daily=daily, hourly=hourly, picks=picks)

    # Same snapshot as the model exports: identical actuals on the shared hours
    try:
        accuracy = load_accuracy()
        shared = hourly.index.intersection(accuracy.test_hours)
        gap = float(
            (hourly.loc[shared, "residual_load"] - accuracy.actual[shared]).abs().max()
        )
        if len(shared) == 0 or gap > 1e-6:
            labels.problems.append(
                f"the risk-label export and the model exports come from different runs — "
                f"re-run {RISK_SOURCE}"
            )
        # Same model run: the actuals survive a refit, so compare the picks' forecasts too
        for direction, model in picks.items():
            current = accuracy.forecasts.get((model, CANDIDATE_SPLIT))
            if current is None or len(shared) == 0:
                continue  # a pick without a current forecast (e.g. an unexported ensemble)
            stored = hourly.loc[shared, f"{direction}_model_forecast"]
            gap = float((stored - current.reindex(shared)).abs().max())
            if not gap <= 1e-3:
                labels.problems.append(
                    f"the risk labels' {direction} pick ({model}) has other forecasts than the "
                    f"model exports (up to {gap:,.0f} MWh apart) — re-run {RISK_SOURCE}"
                )
    except FileNotFoundError:
        pass  # the accuracy page reports its own missing files
    return labels


# ============================================================================
# reBAP cost (spec 11)
# ============================================================================
REBAP_CATEGORIES = [
    "overall",
    "below_zero",
    "low_extreme",
    "high_extreme",
]  # no `ordinary`


@dataclass
class RebapCost:
    """Spec 11's cost export: every miss priced at the hour's mean |reBAP| (a proxy, not a bill)."""

    price: pd.Series  # mean |reBAP| per hour (EUR/MWh), on the priced hours
    cost: pd.DataFrame  # hours × rows: |miss| × price (EUR), SMARD included
    saved: (
        pd.DataFrame
    )  # hours × rows: SMARD's cost − the row's cost (EUR), SMARD excluded
    problems: list = field(default_factory=list)


def load_rebap_cost() -> RebapCost:
    """Spec 11's cost export, cached on its file and the model exports it is checked against."""
    stamps = tuple(
        export_stamp(name) for name in ("rebap_cost", "scoreboard", "hourly", "smard")
    )
    return _load_rebap_cost(stamps)


@st.cache_data
def _load_rebap_cost(stamps) -> RebapCost:
    """Read spec 11's export and check it against `load_accuracy()`: same run, same hours.

    Reads only the export, never `rebap.csv` (reBAP stays out of shared work, CLAUDE.md).
    """
    frame = load_export("rebap_cost")
    rows = ["model", "split_method"]
    cost = frame.pivot(index="timestamp", columns=rows, values="cost_eur")
    saved = frame.pivot(index="timestamp", columns=rows, values="saved_eur").drop(
        columns=[SMARD_ROW]
    )
    error = frame.pivot(index="timestamp", columns=rows, values="err_residual_load")
    hourly = frame.groupby("timestamp").first()

    regen = f"re-run {REBAP_SOURCE}"
    accuracy = load_accuracy()
    # Rows `load_accuracy` doesn't have (spec 08's ensembles while their exports are missing)
    # can't be checked, so they are left out
    extra = [row for row in error if row not in accuracy.errors]
    cost, saved, error = (
        f.drop(columns=[r for r in extra if r in f]) for f in (cost, saved, error)
    )
    rebap = RebapCost(price=hourly["rebap_abs_mean"], cost=cost, saved=saved)
    rebap.problems += [
        f"{row} is not in the model exports — {regen}"
        for row in extra
        if not row[0].startswith("ensemble_")
    ]
    if not cost.index.equals(accuracy.common):
        rebap.problems.append(
            f"the cost export prices {len(cost):,} hours, the model exports have "
            f"{len(accuracy.common):,} common hours — {regen}"
        )
        return rebap
    gap = float(
        (hourly["residual_load"] - accuracy.actual[accuracy.common]).abs().max()
    )
    if gap > 1e-6:
        rebap.problems.append(
            f"residual_load differs from the model exports by up to {gap:,.2f} MWh — {regen}"
        )
    for row in error:
        if float((error[row] - accuracy.errors[row]).abs().max()) > 1e-6:
            rebap.problems.append(
                f"{row}: misses differ from the model exports — {regen}"
            )
    priced = (error.abs().mul(rebap.price, axis=0) - cost).abs().max().max()
    if priced > 1e-3:
        rebap.problems.append(
            f"cost ≠ |miss| × |reBAP| (up to {priced:,.3f} EUR) — {regen}"
        )
    for name in REBAP_CATEGORIES[1:]:
        if not (
            hourly[f"in_{name}"].to_numpy() == accuracy.bin_hours[name].to_numpy()
        ).all():
            rebap.problems.append(
                f"{name}: hours differ from the model exports — {regen}"
            )
    return rebap


def outcome(forecast, actual):
    """hit, miss, false alarm or quiet per row; empty where either flag is empty (spec 10 §3)."""
    f, a = is_set(forecast), is_set(actual)
    labels = np.select([f & a, a & ~f, f & ~a], OUTCOMES[:3], OUTCOMES[3])
    return pd.Series(labels, index=actual.index).where(
        forecast.notna() & actual.notna()
    )


def risk_scores(labels, direction, basis, rule):
    """Day and hour counts per scored source, as one row of spec 10 §3's table.

    Hours have no `3h` rule, so a `3h` score repeats the hour counts of `any`.
    """
    scores = {}
    actual_days = labels.flag("actual", direction, basis, rule)
    actual_hours = labels.hour_flag("actual", direction, basis)
    for source in SOURCES_SCORED:
        days = outcome(labels.flag(source, direction, basis, rule), actual_days)
        hours = outcome(labels.hour_flag(source, direction, basis), actual_hours)
        hit, miss, false_alarm = (int(days.eq(name).sum()) for name in OUTCOMES[:3])
        scores[source] = {
            "actual days": hit + miss,
            "hit days": hit,
            "missed days": miss,
            "false-alarm days": false_alarm,
            "recall %": 100 * hit / (hit + miss) if hit + miss else np.nan,
            "precision %": (
                100 * hit / (hit + false_alarm) if hit + false_alarm else np.nan
            ),
            "hit hours": int(hours.eq("hit").sum()),
            "missed hours": int(hours.eq("miss").sum()),
            "false-alarm hours": int(hours.eq("false alarm").sum()),
        }
    return scores


def german_holidays(days):
    """Federal German holidays over the years of `days` (`country_holidays("DE")`, no `subdiv`)."""
    return holidays.country_holidays(
        "DE", years=range(days[0].year, days[-1].year + 1), language="en_US"
    )


def persistent_hours(flag):
    """Flagged hours in a run of at least `PERSISTENCE` within their day (spec 10 §4.2).

    The run length is a duration, converted through the measured resolution of `flag`'s index.
    """
    resolution = flag.index.to_series().diff().mode().iloc[0]
    min_run = int(PERSISTENCE / resolution)
    filled = is_set(flag)
    run_id = (filled != filled.shift()).cumsum()
    run_len = filled.groupby([flag.index.normalize(), run_id]).transform("size")
    return filled & run_len.ge(min_run)


def flagged_span(labels, source, direction, basis, day, rule="any"):
    """A day's flagged range of a source as text, e.g. `16:00–20:00`, or `—` (spec 10 §4.2)."""
    start = labels.daily.loc[day, f"{source}_{direction}_range_start_{basis}_{rule}"]
    end = labels.daily.loc[day, f"{source}_{direction}_range_end_{basis}_{rule}"]
    if pd.isna(start):
        return "—"
    return f"{start:%H:%M}" if start == end else f"{start:%H:%M}–{end:%H:%M}"


def week_of(days):
    """The Monday of each day's ISO week (a day or an index)."""
    return days.to_period("W-SUN").start_time


def default_zoom_weeks(labels):
    """Spec 10's zoom rules: per direction, two (Monday or None, reason) pairs."""
    daily = labels.daily
    days_off = german_holidays(daily.index)

    def lowest_flagged_day(months, holidays_only):
        flagged = daily[
            is_set(daily["actual_low_risk_rolling_any"])
            & daily.index.month.isin(months)
        ]
        if holidays_only:
            flagged = flagged[[day in days_off for day in flagged.index]]
        return None if flagged.empty else week_of(flagged["actual_min"].idxmin())

    def most_actual_high_days(_):
        flagged = is_set(daily["actual_high_risk_rolling_any"])
        weeks = pd.DataFrame({"days": flagged, "peak": daily["actual_max"]}).groupby(
            week_of(daily.index)
        )
        table = weeks.agg(days=("days", "sum"), peak=("peak", "max")).sort_values(
            ["days", "peak"], ascending=False
        )
        return table.index[0] if table["days"].iloc[0] > 0 else None

    def most_high_errors(excluded):
        actual = daily["actual_high_risk_rolling_any"]
        errors = sum(
            outcome(daily[f"{source}_high_risk_rolling_any"], actual)
            .isin(["miss", "false alarm"])
            .astype(int)
            for source in SOURCES_SCORED
        )
        per_week = (
            errors.groupby(week_of(daily.index)).sum().drop(excluded, errors="ignore")
        )
        return (
            per_week.idxmax() if per_week.max() > 0 else None
        )  # idxmax: the earliest on ties

    rules = {
        "high": [
            ("the week with the most high-risk days", most_actual_high_days),
            ("the week with the most misses and false alarms", most_high_errors),
        ],
        "low": [
            (
                "the spring/summer holiday with the lowest residual load",
                lambda _: lowest_flagged_day([3, 4, 5, 6, 7, 8], True),
            ),
            (
                "the winter day with the lowest residual load",
                lambda _: lowest_flagged_day([12, 1, 2], False),
            ),
        ],
    }
    weeks = {}
    for direction, pairs in rules.items():
        weeks[direction] = []
        for reason, pick in pairs:
            first = weeks[direction][0][0] if weeks[direction] else None
            weeks[direction].append((pick(first), reason))
    return weeks


# ============================================================================
# Model windows (spec 06 saves)
# ============================================================================
def model_windows():
    """The validation and test windows of the spec 06 saves, as (start, end) Timestamps.

    Read from `data/models/<model_key>/config.json` (`tuned_on`), which a clone carries. Raises
    `FileNotFoundError` when no save is found.
    """
    configs = sorted((_find_data_dir() / "models").glob("*/config.json"))
    for path in configs:
        tuned_on = json.loads(path.read_text()).get("tuned_on", {})
        if "validation" in tuned_on and "test" in tuned_on:
            return {
                name: tuple(pd.Timestamp(part) for part in tuned_on[name].split(" .. "))
                for name in ("validation", "test")
            }
    raise FileNotFoundError(
        "no model save with `tuned_on` windows in data/models/ — run "
        "notebooks/05_modeling/regression-models-claude.ipynb with SAVE_MODELS = True."
    )


def load_ensemble_weights():
    """Spec 08's final weights: `method`, `split_method`, `regime`, `member`, `weight`
    (`adaptive`: one `regime` per test day)."""
    return load_export("ensemble_weights")


def ensemble_ready():
    """True when all of spec 08's ensemble exports exist (the app's ensemble blocks are still
    placeholders; this only changes their wording)."""
    return all(
        (_find_data_dir() / SOURCES[name][0]).is_file() for name in ENSEMBLE_EXPORTS
    )
