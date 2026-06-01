import pandas as pd
import numpy as np
from pathlib import Path
import json
import importlib.util
import os
import shutil
import lightgbm as lgb
import re
import torch

from tsfeatures import tsfeatures
from datasetsforecast.losses import (
    mae, mape, rmse, smape,
)
from typing import Sequence, TypeVar

from matplotlib.ticker import FuncFormatter

# Type definitions for array-like inputs
ArrayLike = TypeVar('ArrayLike', np.ndarray, Sequence[float], Sequence[int])

def cleanup_lightning_artifacts() -> None:
    """
    Remove the ``lightning_logs`` and ``checkpoints`` folders that
    PyTorch Lightning (used by GluonTS' DeepAR / TFT) writes to the working
    directory. Safe to call even when neither folder exists.
    """
    for folder in ("lightning_logs", "checkpoints"):
        path = Path.cwd() / folder
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


col_order = ["series_id", "date", "fcst", "runtime", "model"]


def _nullify_partial_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace every numeric cell of any row with at least one NaN with NaN,
    so partial-NaN rows read as fully unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose numeric columns may contain sporadic NaN values.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` where every numeric cell in a row that contained at
        least one NaN is set to NaN.
    """
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number])
    if numeric.empty:
        return out
    not_run = numeric.isna().any(axis=1)
    if not_run.any():
        out.loc[not_run, numeric.columns] = np.nan
    return out

_PERIOD_TO_OFFSET_FREQ = {"M": "MS", "Q": "QS", "Y": "AS-JAN", "A": "AS-JAN"}

def to_period_freq(freq: str) -> str:
    """
    Convert a pandas DateOffset-style frequency string to a Period-style one.

    Parameters
    ----------
    freq : str
        A pandas DateOffset frequency alias (e.g., ``"MS"``, ``"QS"``,
        ``"YS-JAN"``).

    Returns
    -------
    str
        The equivalent Period frequency alias (e.g., ``"M"``, ``"Q"``,
        ``"A"``). Returned unchanged when no mapping exists.
    """
    return {
        "MS": "M",
        "QS": "Q",
        "AS-JAN": "A",
        "AS": "A",
        "YS-JAN": "A",
        "YS": "A",
        "Y": "A",
    }.get(freq, freq)


def to_offset_freq(freq: str) -> str:
    """
    Convert a pandas Period-style frequency string to a DateOffset-style one.

    Inverse of ``to_period_freq``.

    Parameters
    ----------
    freq : str
        A pandas Period frequency alias (e.g., ``"M"``, ``"Q"``, ``"Y"``).

    Returns
    -------
    str
        The equivalent DateOffset frequency alias (e.g., ``"MS"``, ``"QS"``,
        ``"AS-JAN"``). Returned unchanged when no mapping exists.
    """
    return _PERIOD_TO_OFFSET_FREQ.get(freq, freq)


def to_statsforecast_frames(
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list = None,
) -> tuple:
    """
    Reshape HyperTrees DataFrames into the statsforecast schema.

    Converts ``(series_id, date, value [, features])`` columns into the
    ``(unique_id, ds, y)`` schema required by statsforecast.

    Parameters
    ----------
    train : pd.DataFrame
        Training DataFrame with at least ``series_id``, ``date``, ``value``
        columns.
    test : pd.DataFrame
        Test DataFrame with the same column structure as ``train``.
    features : list, optional
        Names of exogenous feature columns to carry through. When ``None``,
        no exogenous frame is built for the test set.

    Returns
    -------
    tuple
        ``(ts_train, ts_test_X)`` where ``ts_train`` has the statsforecast
        ``(unique_id, ds, y)`` schema and ``ts_test_X`` is a DataFrame of
        exogenous features (or ``None`` when ``features`` is not provided).
    """
    keep = ["series_id", "date", "value"] + (list(features) if features else [])
    ts_train = train[keep].rename(
        columns={"series_id": "unique_id", "date": "ds", "value": "y"}
    ).copy()
    ts_train["ds"] = pd.to_datetime(ts_train["ds"])
    if features:
        ts_test_X = test[["series_id", "date"] + list(features)].rename(
            columns={"series_id": "unique_id", "date": "ds"}
        ).copy()
        ts_test_X["ds"] = pd.to_datetime(ts_test_X["ds"])
    else:
        ts_test_X = None
    return ts_train, ts_test_X


def to_statsforecast_output(
        raw: pd.DataFrame,
        test: pd.DataFrame,
        model_label: str,
        runtime: float,
        col_order: list,
) -> pd.DataFrame:
    """
    Transform a statsforecast forecast DataFrame into the HyperTrees output schema.

    The forecast column is detected as the only non-metadata column (ignoring
    ``unique_id``, ``ds``, ``cutoff``, ``index``, ``level_0``).

    The ``date`` column is taken from ``test`` via a positional merge on
    ``series_id`` + within-series row index; both sides are assumed to be
    (series_id, date)-ascending within each series. ``series_id`` is cast to
    ``str`` on both sides so int-vs-str mismatches don't break the merge.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw output from ``statsforecast.forecast()``.
    test : pd.DataFrame
        Test DataFrame containing the authoritative ``date`` column used to
        align forecast rows to calendar dates.
    model_label : str
        Label written into the ``model`` column of the result.
    runtime : float
        Elapsed training time in seconds, written into the ``runtime`` column.
    col_order : list
        Column ordering for the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns matching ``col_order``.
    """
    if "unique_id" not in raw.columns:
        raw = raw.reset_index()
    meta = {"unique_id", "ds", "cutoff", "index", "level_0"}
    model_cols = [c for c in raw.columns if c not in meta]
    if not model_cols:
        raise ValueError(
            f"No forecast column in statsforecast output: {raw.columns.tolist()}"
        )
    fcst_col = model_cols[0]
    out = raw[["unique_id", "ds", fcst_col]].rename(
        columns={"unique_id": "series_id", "ds": "date", fcst_col: "fcst"}
    )
    out["series_id"] = out["series_id"].astype(str)

    # Positional merge on series_id + within-series row index. Relies on both
    # sides being (series_id, date)-ascending within each series.
    out["_t"] = out.groupby("series_id", sort=False).cumcount()

    test_dates = test[["series_id", "date"]].copy()
    test_dates["series_id"] = test_dates["series_id"].astype(str)
    test_dates["_t"] = test_dates.groupby("series_id", sort=False).cumcount()

    out = (
        out.drop(columns=["date"])
        .merge(test_dates, on=["series_id", "_t"], how="left")
        .drop(columns=["_t"])
    )
    if out["date"].isna().any():
        raise ValueError(
            "Positional merge with test grid produced missing dates; check "
            "that `test` has every (series_id, forecast-step) row expected."
        )

    out["model"] = model_label
    out["runtime"] = runtime
    return out[col_order]


def create_lag_features(
        df: pd.DataFrame,
        lags: list,
        features: list,
) -> pd.DataFrame:
    """
    Add lag columns per series via groupby + shift.

    Creates ``lag{N}`` columns (one per entry in ``lags``) sorted by
    ``series_id, date``. The first ``max(lags)`` rows per series are dropped
    because their lag values are undefined.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least ``series_id``, ``date``, and ``value``
        columns.
    lags : list
        Positive integers specifying which lags to create (e.g.,
        ``[1, 2, 3]``).
    features : list
        Names of additional columns (beyond ``series_id``, ``date``,
        ``value``) to carry through.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original columns plus ``lag{N}`` columns, rows
        with incomplete lags removed.
    """
    out = df[["series_id", "date", "value"] + features].sort_values(
        ["series_id", "date"]
    ).reset_index(drop=True).copy()

    grouped = out.groupby("series_id", sort=False)["value"]
    lag_cols = []
    for lag in lags:
        col = f"lag{lag}"
        out[col] = grouped.shift(lag)
        lag_cols.append(col)

    return out.dropna(subset=lag_cols).reset_index(drop=True)


def fill_missing_dates(
        df_missing: pd.DataFrame,
        date_column='date',
        series_id_column='series_id',
        freq="D"
) -> pd.DataFrame:
    """
    Reindex a DataFrame so every (series_id, date) pair in the full date range exists.

    Missing values are filled with 0.

    Parameters
    ----------
    df_missing : pd.DataFrame
        DataFrame that may have gaps in its date index.
    date_column : str, optional
        Name of the date column (default ``'date'``).
    series_id_column : str, optional
        Name of the series identifier column (default ``'series_id'``).
    freq : str, optional
        Pandas frequency alias used to build the complete date range
        (default ``"D"``).

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed to the full Cartesian product of series IDs and
        the complete date range, with missing numeric values filled with 0.
    """
    # Get the min and max dates across all series
    min_date = df_missing[date_column].min()
    max_date = df_missing[date_column].max()

    # Create a complete date range
    full_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    # Get unique series IDs
    series_ids = df_missing[series_id_column].unique()

    # Create a MultiIndex with all combinations of series_id and dates
    multi_idx = pd.MultiIndex.from_product(
        [series_ids, full_date_range],
        names=[series_id_column, date_column]
    )

    # Reindex the DataFrame using the MultiIndex
    df_filled = (df_missing
                    .set_index([series_id_column, date_column])
                    .reindex(multi_idx)
                    .fillna(0)
                    .reset_index()
                    )

    return df_filled

def load_experiments_specs(
        dataset: str,
        train_type: str,
) -> dict:
    """
    Load train, test, metadata, and per-model config for a dataset.

    The ``_ets`` entries in the returned dict are populated only when a
    padded ETS variant exists for the dataset.

    Parameters
    ----------
    dataset : str
        Directory name of the dataset under ``experiments/datasets/``
        (e.g., ``"airpassengers"``).
    train_type : str
        Training scope; one of ``"local"`` or ``"global"``.

    Returns
    -------
    dict
        Dictionary with keys ``train``, ``test``, ``train_ets``,
        ``test_ets``, ``meta``, ``meta_ets``, ``config``.
    """
    base_path = (Path(__file__).parent / "datasets").resolve()

    # Load configuration with Hyper-Parameters for each model
    if train_type == "local":
        config_path = Path(base_path)  / dataset / "config_local.py"
    elif train_type == "global":
        config_path = Path(base_path)  / dataset / "config_global.py"
        torch.set_float32_matmul_precision('medium')
    spec = importlib.util.spec_from_file_location("config_module", str(config_path))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    # Load datasets and metadata
    dataset_path = base_path / dataset

    # Non Padded datasets
    train: pd.DataFrame = pd.read_parquet(dataset_path / "train.parquet")
    test: pd.DataFrame = pd.read_parquet(dataset_path / "test.parquet")
    with open(dataset_path / "meta.json", "r") as f:
        meta: dict = json.load(f)

    # Padded datasets (only used for global ETS models for specific datasets)
    if train_type == "global" and dataset in ["auselectricity", "m3_yearly", "tourism_monthly"]:
        train_pad: pd.DataFrame = pd.read_parquet(dataset_path / "train_padded.parquet")
        test_pad: pd.DataFrame = pd.read_parquet(dataset_path / "test_padded.parquet")
        with open(dataset_path / "meta_ets.json", "r") as f:
            meta_pad: dict = json.load(f)
    else:
        train_pad = None
        test_pad = None
        meta_pad = None

    return {
        "train": train,
        "test": test,
        "train_ets": train_pad,
        "test_ets": test_pad,
        "meta": meta,
        "meta_ets": meta_pad,
        "config": config
    }

def wape(
        true: ArrayLike,
        fcst: ArrayLike
) -> float:
    """
    Weighted Absolute Percentage Error: ``(Σ|true - fcst| / Σ|true|) * 100``.

    Raises ``ValueError`` on shape mismatch or when ``Σ|true| == 0``.

    Parameters
    ----------
    true : ArrayLike
        Array of actual values.
    fcst : ArrayLike
        Array of forecast values; must have the same shape as ``true``.

    Returns
    -------
    float
        WAPE expressed as a percentage.
    """
    # Convert inputs to numpy arrays if they aren't already
    true_array = np.asarray(true)
    fcst_array = np.asarray(fcst)

    # Check for shape mismatch
    if true_array.shape != fcst_array.shape:
        raise ValueError(f"Input shapes must match. Got true: {true_array.shape}, fcst: {fcst_array.shape}")

    # Calculate sum of absolute true values
    sum_abs_true = np.sum(np.abs(true_array))

    # Check for division by zero
    if sum_abs_true == 0:
        raise ValueError(
            "Sum of absolute true values is zero, which would cause division by zero in WAPE calculation.")

    # Calculate WAPE
    return (np.sum(np.abs(true_array - fcst_array)) / sum_abs_true) * 100


def calculate_error_metrics(df: pd.DataFrame,
                            round_digit: int = 5,
                            ) -> pd.Series:
    """
    Compute MAPE, sMAPE, WAPE, RMSE, and MAE from actual and forecast columns.

    Formulas match the AWS Forecast definitions; sMAPE is implemented locally
    to avoid a ``datasetsforecast.losses.smape`` bug on newer numpy versions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``value`` (actual) and ``fcst`` (forecast)
        columns.
    round_digit : int, optional
        Number of decimal places to round each metric to (default ``5``).

    Returns
    -------
    pd.Series
        Series indexed by metric name (``MAPE``, ``sMAPE``, ``WAPE``,
        ``RMSE``, ``MAE``).
    """
    true = df["value"].to_numpy().reshape(-1, 1)
    fcst = df["fcst"].to_numpy().reshape(-1, 1)

    # Local sMAPE implementation
    denom = np.abs(true) + np.abs(fcst)
    smape_val = np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(true - fcst) / denom))

    metrics = {
        "MAPE": mape(true, fcst).round(round_digit),
        "sMAPE": round(float(smape_val), round_digit),
        "WAPE": wape(true, fcst).round(round_digit),
        "RMSE": rmse(true, fcst).round(round_digit),
        "MAE": mae(true, fcst).round(round_digit),
    }

    return pd.Series(metrics)

def create_tsfeatures(
        train: pd.DataFrame,
        freq: str = "D"
) -> tuple:
    """
    Extract Nixtla ``tsfeatures`` from a training DataFrame.

    Missing dates are filled for the Rossmann dataset before feature
    extraction.

    Parameters
    ----------
    train : pd.DataFrame
        Training DataFrame with at least ``date``, ``series_id``, ``value``,
        and ``dataset`` columns.
    freq : str, optional
        Pandas frequency alias used when filling missing dates
        (default ``"D"``).

    Returns
    -------
    tuple
        ``(ts_feat_df, ts_feats)`` where ``ts_feat_df`` is a DataFrame of
        extracted features indexed by ``series_id`` and ``ts_feats`` is a
        list of the feature column names.
    """
    # Define the frequency dictionary for tsfeatures
    dict_freqs = {
        "D": 7,
        "W": 52,
        "MS": 12,
        "QS": 4,
        "AS-JAN": 1,
        "YS-JAN": 1,
    }

    # Fill missing dates
    if train["dataset"].unique()[0] == "Rossmann Store Sales":
        ts_feat_df = fill_missing_dates(
            df_missing=train[["date", "series_id", "value"]],
            date_column="date",
            series_id_column="series_id",
            freq=freq
        ).rename(columns={"series_id": "unique_id", "date": "ds", "value": "y"}).copy()
    else:
        ts_feat_df = train[["date", "series_id", "value"]].rename(columns={"series_id": "unique_id", "date": "ds", "value": "y"}).copy()

    # Add time features
    ts_feat_df = tsfeatures(ts_feat_df, dict_freqs=dict_freqs).rename(columns={"unique_id": "series_id"}).fillna(0)
    ts_feats = ts_feat_df.columns.tolist()[1:]

    return ts_feat_df, ts_feats



def evaluate_forecasts(
        results_dir: str,
        train_type: str,
        round_decimals: int = 3,
) -> pd.DataFrame:
    """
    Load forecast CSVs and compute averaged error metrics per (dataset, model).

    Reads all ``*fcsts.csv`` files from ``{results_dir}/{train_type}/``,
    computes per-series error metrics, averages across series, and returns
    a summary DataFrame. MASE is computed relative to AutoETS (which is
    dropped from the output when ``train_type='global'``).

    Parameters
    ----------
    results_dir : str
        Root directory containing ``global/`` and ``local/`` subdirectories
        with forecast CSV files.
    train_type : str
        Training scope; one of ``"local"`` or ``"global"``.
    round_decimals : int, optional
        Number of decimal places to round metrics to (default ``3``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(dataset, model)`` with columns MAPE, sMAPE,
        WAPE, RMSE, MAE, MASE (and ``runtime`` for global).
    """

    # Load and concatenate all forecast files
    fcsts_dir = os.path.join(results_dir, train_type)
    fcst_files = [f for f in os.listdir(fcsts_dir) if f.endswith('fcsts.csv')]
    if not fcst_files:
        raise FileNotFoundError(f"No forecast CSV files found in {fcsts_dir}")

    results_df_list = pd.concat(
        [pd.read_csv(os.path.join(fcsts_dir, f)) for f in fcst_files],
        axis=0, ignore_index=True
    )

    # Remove model specification from model names (e.g., "Hyper-Tree-AR(12)" -> "Hyper-Tree-AR")
    results_df_list["model"] = results_df_list['model'].str.replace(r'\(.*?\)', '', regex=True)

    # Extract runtimes (one value per dataset/model)
    runtimes_df = (
        results_df_list.groupby(["dataset", "model"])["runtime"]
        .mean()
        .reset_index()
    )

    # Calculate error metrics per dataset, series, model
    fcsts_df = results_df_list.drop(columns=["runtime"])
    dta_sets = fcsts_df["dataset"].unique()
    err_df_list = []
    for dta_set in dta_sets:
        fcsts_df_sub = fcsts_df[fcsts_df["dataset"] == dta_set]
        err_df_series = fcsts_df_sub.groupby(["dataset", "series_id", "model"]).apply(
            calculate_error_metrics
        ).reset_index()

        err_df_series["series_id"] = err_df_series["series_id"].astype(str)

        # Compute MASE relative to AutoETS (global only)
        if train_type == "global":
            autoets_mae = err_df_series[err_df_series['model'] == 'AutoETS'].groupby(["series_id"])['MAE'].first()
            err_df_series['MASE'] = err_df_series['MAE'] / err_df_series['series_id'].map(autoets_mae)
            err_df_series = err_df_series[err_df_series['model'] != 'AutoETS'].reset_index(drop=True)

        # Average across series
        numeric_columns = err_df_series.select_dtypes(include=[np.number]).columns
        err_df_list.append(err_df_series.groupby(["dataset", "model"])[numeric_columns].mean())

    err_df = pd.concat(err_df_list, axis=0, ignore_index=False)

    # Merge error metrics with runtimes
    eval_df = (
        err_df.reset_index()
        .merge(runtimes_df, on=["dataset", "model"], how="left")
        .set_index(["dataset", "model"])
        .round(round_decimals)
    )

    if train_type == "local":
        eval_df.drop(columns="runtime", inplace=True)

    return _nullify_partial_rows(eval_df)


# Per-dataset AutoETS source rule used by the ablation evaluators.
# Datasets not listed here default to ``"global"``.
AUTOETS_SOURCE_BY_DATASET = {
    # Keys are the display names stored in the forecast CSVs' ``dataset`` column.
    "Air Passengers": "local",
}
AUTOETS_DEFAULT_SOURCE = "global"


def _autoets_source_for(dataset: str) -> str:
    """
    Return the required AutoETS source for a given dataset.

    Parameters
    ----------
    dataset : str
        Display name of the dataset (as stored in the forecast CSV's
        ``dataset`` column).

    Returns
    -------
    str
        ``"global"`` or ``"local"``, looked up from
        ``AUTOETS_SOURCE_BY_DATASET`` (default ``"global"``).
    """
    return AUTOETS_SOURCE_BY_DATASET.get(dataset, AUTOETS_DEFAULT_SOURCE)


def _load_autoets_mae(results_dir: str) -> pd.Series:
    """
    Load AutoETS forecasts and return per-series MAE.

    Reads AutoETS forecast CSVs from the global and local result directories,
    selects the appropriate source per dataset according to
    ``AUTOETS_SOURCE_BY_DATASET``, and computes per-series MAE.

    Parameters
    ----------
    results_dir : str
        Root directory containing ``global/`` and ``local/`` subdirectories
        with forecast CSV files.

    Returns
    -------
    pd.Series
        MAE values indexed by ``(dataset, series_id)``.
    """
    autoets_rows = []
    for source in ("global", "local"):
        source_dir = os.path.join(results_dir, source)
        if not os.path.isdir(source_dir):
            continue
        for f in os.listdir(source_dir):
            if not f.endswith("_fcsts.csv"):
                continue
            df = pd.read_csv(os.path.join(source_dir, f))
            ae_df = df[df["model"].str.startswith("AutoETS")]
            if not ae_df.empty:
                ae_df = ae_df.assign(source=source)
                autoets_rows.append(ae_df)

    if not autoets_rows:
        raise FileNotFoundError(
            f"AutoETS forecasts not found in {results_dir}/global or {results_dir}/local. "
            "Run the global and/or local stages first; the ablation evaluators "
            "rely on those AutoETS forecasts to compute MASE."
        )

    autoets_df = pd.concat(autoets_rows, ignore_index=True)
    autoets_df["model"] = "AutoETS"
    autoets_df["series_id"] = autoets_df["series_id"].astype(str)

    # Keep only the row whose `source` matches the required source for its
    # dataset; see AUTOETS_SOURCE_BY_DATASET above.
    required = autoets_df["dataset"].map(_autoets_source_for)
    autoets_df = (
        autoets_df[autoets_df["source"] == required]
        .drop(columns=["source"])
        .reset_index(drop=True)
    )

    err = (
        autoets_df.groupby(["dataset", "series_id", "model"])
        .apply(calculate_error_metrics)
        .reset_index()
    )
    return err.set_index(["dataset", "series_id"])["MAE"]


def evaluate_ablation_rossmann(
        results_dir: str = "results/",
        round_decimals: int = 3,
) -> pd.DataFrame:
    """
    Evaluate Rossmann ablation forecasts (Base, A1-A11).

    MASE is computed relative to the AutoETS forecasts in the global
    Rossmann run.

    Parameters
    ----------
    results_dir : str, optional
        Root results directory (default ``"results/"``).
    round_decimals : int, optional
        Number of decimal places to round metrics to (default ``3``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(dataset, model, ablation)`` with columns
        MAPE, sMAPE, WAPE, RMSE, MAE, MASE.
    """
    fcsts_dir = os.path.join(results_dir, "ablation", "rossmann")
    fcst_files = [f for f in os.listdir(fcsts_dir) if f.endswith("fcsts.csv")]
    if not fcst_files:
        raise FileNotFoundError(f"No forecast CSV files found in {fcsts_dir}")

    df = pd.concat(
        [pd.read_csv(os.path.join(fcsts_dir, f)) for f in fcst_files],
        axis=0, ignore_index=True,
    )

    # Strip model specification suffixes (e.g., "Hyper-Tree-AR(21)" -> "Hyper-Tree-AR")
    df["model"] = df["model"].str.replace(r"\(.*?\)", "", regex=True)

    # "Base" row = unablated metrics taken from the global Rossmann run.
    # Paper (Table 4) reports each A1-A11 variant alongside the Base model
    # whose metrics come from the global run (Table 3).
    base_path = os.path.join(results_dir, "global", "rossmann_hypertrees_fcsts.csv")
    base_df = pd.read_csv(base_path)
    base_df["model"] = base_df["model"].str.replace(r"\(.*?\)", "", regex=True)
    base_df = base_df[base_df["model"].isin(["Hyper-Tree-AR", "Hyper-TreeNet-AR"])].copy()
    base_df["ablation"] = "Base"
    df = pd.concat([df, base_df[df.columns]], axis=0, ignore_index=True)

    # Error metrics per (dataset, model, ablation, series)
    err_df = df.groupby(["dataset", "model", "ablation", "series_id"]).apply(
        calculate_error_metrics
    ).reset_index()
    err_df["series_id"] = err_df["series_id"].astype(str)

    # Order ablations: Base first, then A1, A2, ..., A11 (not lexicographic).
    ablation_order = ["Base"] + sorted(
        [a for a in err_df["ablation"].unique() if a != "Base"],
        key=lambda s: int(str(s).lstrip("A")),
    )
    err_df["ablation"] = pd.Categorical(
        err_df["ablation"], categories=ablation_order, ordered=True
    )

    # MASE = MAE / AutoETS-MAE per series, looked up from the global Rossmann run.
    # The `dataset` column in the CSVs uses the display name ("Rossmann Store
    # Sales") rather than the directory name, so read it from the ablation
    # CSVs themselves and slice on that.
    rossmann_dataset_name = df["dataset"].iloc[0]
    autoets_mae = _load_autoets_mae(results_dir).xs(rossmann_dataset_name, level="dataset")
    err_df["MASE"] = err_df.apply(
        lambda row: row["MAE"] / autoets_mae[row["series_id"]]
        if row["series_id"] in autoets_mae.index else np.nan,
        axis=1,
    )

    # Average across series per (dataset, model, ablation)
    numeric_columns = err_df.select_dtypes(include=[np.number]).columns
    eval_df = (
        err_df.groupby(["dataset", "model", "ablation"], observed=True)[numeric_columns]
        .mean()
        .round(round_decimals)
        .sort_index()
    )

    return _nullify_partial_rows(eval_df)


def evaluate_ablation_embeddings(
        results_dir: str = "results/",
        round_decimals: int = 3,
) -> pd.DataFrame:
    """
    Evaluate embedding-dimension ablation forecasts.

    MASE is computed relative to AutoETS per (dataset, series_id).

    Parameters
    ----------
    results_dir : str, optional
        Root results directory (default ``"results/"``).
    round_decimals : int, optional
        Number of decimal places to round metrics to (default ``3``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(dataset, model, embedding_dim)`` with
        columns MAPE, sMAPE, WAPE, RMSE, MAE, MASE, and ``runtime``.
    """
    fcsts_dir = os.path.join(results_dir, "ablation", "embedding_evaluation")
    fcst_files = [f for f in os.listdir(fcsts_dir) if f.endswith("fcsts.csv")]
    if not fcst_files:
        raise FileNotFoundError(f"No forecast CSV files found in {fcsts_dir}")

    df = pd.concat(
        [pd.read_csv(os.path.join(fcsts_dir, f)) for f in fcst_files],
        axis=0, ignore_index=True,
    )

    # Strip model specification suffixes
    df["model"] = df["model"].str.replace(r"\(.*?\)", "", regex=True)

    # Drop any AutoETS rows in the ablation CSVs; AutoETS is pulled from the
    # global/local stages for the MASE denominator.
    df = df[df["model"] != "AutoETS"].reset_index(drop=True)

    # embedding_dim=1 baseline: pull Hyper-TreeNet-AR results from the
    # global/local hypertrees runs (default config uses embedding_dimension=1).
    # airpassengers runs locally; all other datasets run globally.
    base_frames = []
    for subdir, pattern in [("global", "_hypertrees_fcsts.csv"), ("local", "airpassengers_hypertrees_fcsts.csv")]:
        ht_dir = os.path.join(results_dir, subdir)
        if not os.path.isdir(ht_dir):
            continue
        for f in os.listdir(ht_dir):
            if subdir == "local" and f != pattern:
                continue
            if not f.endswith("_hypertrees_fcsts.csv"):
                continue
            base_df = pd.read_csv(os.path.join(ht_dir, f))
            base_df["model"] = base_df["model"].str.replace(r"\(.*?\)", "", regex=True)
            base_df = base_df[base_df["model"] == "Hyper-TreeNet-AR"].copy()
            base_df["embedding_dim"] = 1
            base_frames.append(base_df)
    if base_frames:
        base_all = pd.concat(base_frames, axis=0, ignore_index=True)
        df = pd.concat([df, base_all[df.columns]], axis=0, ignore_index=True)

    # Extract runtimes (one value per dataset/model/embedding_dim)
    runtimes_df = (
        df.groupby(["dataset", "model", "embedding_dim"])["runtime"]
        .mean()
        .reset_index()
    )
    runtimes_df["embedding_dim"] = pd.to_numeric(runtimes_df["embedding_dim"])
    df = df.drop(columns=["runtime"])

    # Error metrics per (dataset, model, embedding_dim, series)
    err_df = df.groupby(["dataset", "model", "embedding_dim", "series_id"]).apply(
        calculate_error_metrics
    ).reset_index()
    err_df["series_id"] = err_df["series_id"].astype(str)
    # Ensure embedding_dim sorts numerically (1, 3, 5, 10) rather than
    err_df["embedding_dim"] = pd.to_numeric(err_df["embedding_dim"])

    # MASE = MAE / AutoETS-MAE per (dataset, series_id); AutoETS is loaded
    # from the global (and local, for AirPassengers) result directories.
    autoets_mae = _load_autoets_mae(results_dir)
    err_df["MASE"] = err_df.apply(
        lambda row: row["MAE"] / autoets_mae[(row["dataset"], row["series_id"])]
        if (row["dataset"], row["series_id"]) in autoets_mae.index else np.nan,
        axis=1,
    )

    # Average across series per (dataset, model, embedding_dim).
    groupby_keys = ["dataset", "model", "embedding_dim"]
    numeric_columns = [
        c for c in err_df.select_dtypes(include=[np.number]).columns
        if c not in groupby_keys
    ]
    eval_df = (
        err_df.groupby(groupby_keys)[numeric_columns]
        .mean()
        .sort_index()
    )

    eval_df = (
        eval_df.reset_index()
        .merge(runtimes_df, on=groupby_keys, how="left")
        .set_index(groupby_keys)
        .round(round_decimals)
    )

    return _nullify_partial_rows(eval_df)


def thousand_separator(x, pos):
    """
    Matplotlib tick formatter that adds thousand separators.

    Parameters
    ----------
    x : float
        Tick value.
    pos : int
        Tick position (required by the Matplotlib formatter interface,
        unused).

    Returns
    -------
    str
        ``x`` formatted with comma thousand separators and no decimal
        places.
    """
    return f'{x:,.0f}'

formatter = FuncFormatter(thousand_separator)

def custom_format(value, is_min=False):
    """
    Format a float for LaTeX tables, bolding the value if it is the row minimum.

    Parameters
    ----------
    value : float
        Numeric value to format.
    is_min : bool, optional
        When ``True``, wrap the formatted string in ``\\textbf{}``
        (default ``False``).

    Returns
    -------
    str
        LaTeX-ready string; empty string for NaN values.
    """
    if pd.isna(value):
        return ''
    if abs(value) >= 1000:
        formatted = f"{value:,.1f}"
    else:
        formatted = f"{value:.3f}"
    if is_min:
        return r'\textbf{' + formatted + '}'
    return formatted

def latex_table_to_dataframe(latex_content):
    """
    Parse a LaTeX tabular environment into a pandas DataFrame.

    Parameters
    ----------
    latex_content : str
        Raw LaTeX string containing a ``\\begin{tabular}...\\end{tabular}``
        block.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns are the table headers and whose rows are
        the data between ``\\midrule`` and ``\\bottomrule``. Numeric
        columns are converted automatically.
    """
    # Remove newlines and extra spaces
    latex_content = re.sub(r'\s+', ' ', latex_content.strip())

    # Extract the table content
    table_pattern = re.compile(r'\\begin{tabular}.*?\\end{tabular}')
    table_match = table_pattern.search(latex_content)
    if not table_match:
        raise ValueError("Could not find a tabular environment in the LaTeX content")

    latex_table = table_match.group(0)

    # Extract column names
    header_match = re.search(r'\\begin{tabular}{.*?}(.*?)\\\\', latex_table)
    if header_match:
        headers = [h.strip() for h in header_match.group(1).split('&')]
    else:
        raise ValueError("Could not find table headers")

    # Extract data rows
    data_pattern = re.compile(r'\\midrule(.*?)\\bottomrule', re.DOTALL)
    data_match = data_pattern.search(latex_table)
    if data_match:
        data_rows = data_match.group(1).strip().split('\\\\')
        data = [row.strip().split('&') for row in data_rows if row.strip()]
    else:
        raise ValueError("Could not find table data")

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Strip whitespace from all entries
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert numeric columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df

def format_runtime(value, is_min):
    """
    Format a runtime value for LaTeX tables, bolding the minimum.

    Parameters
    ----------
    value : float
        Runtime in seconds.
    is_min : bool
        When ``True``, wrap the formatted string in ``\\textbf{}``.

    Returns
    -------
    str
        LaTeX-ready string; ``'-'`` for NaN values.
    """
    if pd.isna(value):
        return '-'
    formatted = f"{value:.3f}"
    return r'\textbf{' + formatted + '}' if is_min else formatted


def load_params_long(
        dataset: str,
        train_type: str,
        results_dir,
        series_id=None,
):
    """
    Load AR parameters and reshape into plot-ready long format.

    Reads ``{results_dir}/{train_type}/{dataset}_ar_parameters.csv`` and
    derives ``train_end`` and ``fcst_h`` from the matching
    ``{dataset}_hypertrees_fcsts.csv``.

    Parameters
    ----------
    dataset : str
        Dataset directory name (e.g., ``"airpassengers"``).
    train_type : str
        Training scope; one of ``"local"`` or ``"global"``.
    results_dir : str or Path
        Root results directory.
    series_id : str, optional
        When provided, filter to a single series before reshaping.

    Returns
    -------
    tuple
        ``(long_df, train_end, max_lag, fcst_h)`` where ``long_df`` is the
        melted parameter DataFrame, ``train_end`` is a ``pd.Timestamp``,
        ``max_lag`` is the number of AR lags, and ``fcst_h`` is the forecast
        horizon length.
    """
    from pathlib import Path

    results_dir = Path(results_dir)
    params_csv = results_dir / train_type / f"{dataset}_ar_parameters.csv"
    df = pd.read_csv(params_csv, parse_dates=["date"])
    df["series_id"] = df["series_id"].astype(str)
    if series_id is not None:
        df = df[df["series_id"] == str(series_id)].reset_index(drop=True)

    fcst_csv = results_dir / train_type / f"{dataset}_hypertrees_fcsts.csv"
    fcst = pd.read_csv(fcst_csv, parse_dates=["date"])
    fcst["series_id"] = fcst["series_id"].astype(str)
    if series_id is not None:
        fcst = fcst[fcst["series_id"] == str(series_id)]
    fcst_start = fcst["date"].min()
    train_end = df.loc[df["date"] < fcst_start, "date"].max()
    fcst_h = (
        fcst.groupby(["series_id", "model"])["date"].nunique().max()
        if not fcst.empty else 1
    )

    ar_cols = sorted(
        [c for c in df.columns if c.startswith("AR(")],
        key=lambda c: int(c[3:-1]),
    )
    max_lag = len(ar_cols)
    long = df.melt(
        id_vars=["series_id", "date", "model"],
        value_vars=ar_cols,
        var_name="variable",
        value_name="value",
    )
    long["variable"] = long["variable"].str.replace(r"AR\((\d+)\)", r"Lag \1", regex=True)
    long["variable"] = pd.Categorical(
        long["variable"],
        categories=[f"Lag {i}" for i in range(1, max_lag + 1)],
        ordered=True,
    )
    long["type"] = np.where(long["date"] <= train_end, "train", "test")
    return long, train_end, max_lag, int(fcst_h)


def train_end_from_fcsts(dataset: str, results_dir) -> pd.Timestamp:
    """
    Return the last training date for a dataset.

    Computed as the first forecast date minus one day, read from the
    matching ``{dataset}_hypertrees_fcsts.csv`` in either
    ``{results_dir}/local/`` or ``{results_dir}/global/``, whichever
    exists first.

    Parameters
    ----------
    dataset : str
        Dataset directory name (e.g., ``"airpassengers"``).
    results_dir : str or Path
        Root results directory.

    Returns
    -------
    pd.Timestamp
        Last training date.
    """
    from pathlib import Path

    results_dir = Path(results_dir)
    for subdir in ("local", "global"):
        fcst_path = results_dir / subdir / f"{dataset}_hypertrees_fcsts.csv"
        if fcst_path.exists():
            fcst = pd.read_csv(fcst_path, parse_dates=["date"])
            return fcst["date"].min() - pd.Timedelta(days=1)
    raise FileNotFoundError(
        f"No forecast CSV found for '{dataset}' under "
        f"{results_dir}/local or {results_dir}/global."
    )


def load_embeddings_and_params(
        dataset: str,
        embedding_dim: int,
        ablation_dir,
        results_dir,
):
    """
    Load tree-embeddings and AR parameters in plot-ready long format.

    Reads the CSVs saved by ``embedding_ablation.ipynb`` for
    ``{dataset}_{embedding_dim}`` and annotates each row with a
    train/test label.

    Parameters
    ----------
    dataset : str
        Dataset directory name (e.g., ``"airpassengers"``).
    embedding_dim : int
        Embedding dimensionality. When ``1``, the baseline files from the
        main ``global/`` or ``local/`` run are used instead of the
        ablation directory.
    ablation_dir : str or Path
        Directory containing the ablation embedding/parameter CSVs.
    results_dir : str or Path
        Root results directory (used to locate baseline files and to
        derive ``train_end``).

    Returns
    -------
    tuple
        ``(embeds_long, params_long, train_end)`` where ``embeds_long`` and
        ``params_long`` are melted DataFrames and ``train_end`` is a
        ``pd.Timestamp``.
    """
    from pathlib import Path

    ablation_dir = Path(ablation_dir)
    results_dir = Path(results_dir)

    if embedding_dim == 1:
        train_type = "local" if dataset == "airpassengers" else "global"
        base_dir = results_dir / train_type
        embeds_csv = base_dir / f"{dataset}_tree_embeddings.csv"
        params_csv = base_dir / f"{dataset}_ar_parameters.csv"
    else:
        embeds_csv = ablation_dir / f"{dataset}_{embedding_dim}_tree_embeddings.csv"
        params_csv = ablation_dir / f"{dataset}_{embedding_dim}_ar_parameters.csv"

    embeds_df = pd.read_csv(embeds_csv, parse_dates=["date"])
    params_df = pd.read_csv(params_csv, parse_dates=["date"])
    train_end = train_end_from_fcsts(dataset, results_dir)

    def _annotate(df):
        df = df.copy()
        df["type"] = np.where(df["date"] <= train_end, "Train", "Test")
        df["type"] = pd.Categorical(df["type"], categories=["Train", "Test"], ordered=True)
        return df

    embeds_df = _annotate(embeds_df)
    params_df = _annotate(params_df)

    embed_cols = sorted(
        [c for c in embeds_df.columns if c.startswith("tree_embedding_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    embeds_long = embeds_df.melt(
        id_vars=["series_id", "date", "model", "type"],
        value_vars=embed_cols,
        var_name="embedding",
        value_name="value",
    )
    embeds_long["embedding"] = embeds_long["embedding"].str.replace(
        "tree_embedding_", "Tree-Embedding "
    )
    embeds_long["embedding"] = pd.Categorical(
        embeds_long["embedding"],
        categories=[f"Tree-Embedding {i + 1}" for i in range(embedding_dim)],
        ordered=True,
    )

    ar_cols = sorted(
        [c for c in params_df.columns if c.startswith("AR(")],
        key=lambda c: int(c[3:-1]),
    )
    max_lag = len(ar_cols)
    params_long = params_df.melt(
        id_vars=["series_id", "date", "model", "type"],
        value_vars=ar_cols,
        var_name="parameter",
        value_name="value",
    )
    params_long["parameter"] = params_long["parameter"].str.replace(
        r"AR\((\d+)\)", r"Lag \1", regex=True,
    )
    params_long["parameter"] = pd.Categorical(
        params_long["parameter"],
        categories=[f"Lag {i}" for i in range(1, max_lag + 1)],
        ordered=True,
    )

    return embeds_long, params_long, train_end


def auto_layout_plots(plots, plots_per_row: int = 3):
    """
    Arrange patchworklib plots in a grid layout.

    Plots are placed in rows of ``plots_per_row`` horizontally, with rows
    stacked vertically.

    Parameters
    ----------
    plots : list
        List of patchworklib ``Brick`` objects.
    plots_per_row : int, optional
        Maximum number of plots per row (default ``3``).

    Returns
    -------
    pwl.Brick
        Combined patchworklib ``Brick`` containing all plots arranged in a
        grid.
    """
    total_plots = len(plots)
    combined_plot = None

    for i in range(0, total_plots, plots_per_row):
        row_plots = plots[i:i + plots_per_row]
        row = row_plots[0]

        for plot in row_plots[1:]:
            row = row | plot

        if combined_plot is None:
            combined_plot = row
        else:
            combined_plot = combined_plot / row

    return combined_plot


def save_plot(plot, stem: str, plots_dir) -> None:
    """
    Save a plotnine plot as PDF and PNG.

    Files are written to ``{plots_dir}/{stem}.pdf`` and
    ``{plots_dir}/{stem}.png`` (no tight-crop; clipped to the
    ``theme(figure_size=(w, h))`` canvas). The directory is created if it
    does not exist.

    Parameters
    ----------
    plot : plotnine.ggplot
        The plotnine plot object to save.
    stem : str
        File name stem (without extension).
    plots_dir : str or Path
        Target directory for the saved files.
    """
    from pathlib import Path
    from plotnine import save_as_pdf_pages

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plots_dir / f"{stem}.pdf"
    png_path = plots_dir / f"{stem}.png"
    save_as_pdf_pages([plot], filename=str(pdf_path), verbose=False)
    plot.save(str(png_path), dpi=150, verbose=False,
              limitsize=False, bbox_inches=None, pad_inches=0)
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


_LEGEND_JUSTIFICATION_INSTALLED = False


def install_legend_justification_backport() -> None:
    """
    Enable ``theme(legend_justification=...)`` on plotnine 0.12.4
    """
    global _LEGEND_JUSTIFICATION_INSTALLED
    if _LEGEND_JUSTIFICATION_INSTALLED:
        return

    from matplotlib.offsetbox import AnchoredOffsetbox
    from plotnine._mpl import _plotnine_tight_layout as _pntl
    from plotnine.themes.themeable import themeable

    class legend_justification(themeable):
        """Accepts a tuple ``(x, y)`` in [0, 1]^2 or one of
        ``'left' | 'right' | 'top' | 'bottom' | 'center'``."""
        pass

    just_to_loc = {
        (0, 0):     "lower left",   (1, 0):   "lower right",
        (0, 1):     "upper left",   (1, 1):   "upper right",
        (0.5, 0.5): "center",
        (0, 0.5):   "center left",  (1, 0.5): "center right",
        (0.5, 0):   "lower center", (0.5, 1): "upper center",
        "left":     "center left",  "right":  "center right",
        "top":      "upper center", "bottom": "lower center",
        "center":   "center",
    }

    orig_set_legend_position = _pntl.set_legend_position

    def patched_set_legend_position(legend, position, tparams, fig):
        orig_set_legend_position(legend, position, tparams, fig)
        if not isinstance(position, str) and fig.axes:
            engine = fig.get_layout_engine()
            plot = getattr(engine, "plot", None)
            lj = None
            if plot is not None:
                lj = plot.theme.themeables.get("legend_justification")
            just_val = lj.theme_element if lj is not None else (0, 1)
            loc_name = just_to_loc.get(just_val, "upper left")
            legend.loc = AnchoredOffsetbox.codes[loc_name]
            ax = fig.axes[0]
            legend.set_bbox_to_anchor(position, ax.transAxes)

    _pntl.set_legend_position = patched_set_legend_position
    _LEGEND_JUSTIFICATION_INSTALLED = True
