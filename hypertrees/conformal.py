"""Conformal prediction intervals for Hyper-Tree models.

Acknowledgement
---------------
The conformal-interval approach in this module is adapted from Nixtla's
open-source forecasting libraries:

- statsforecast: https://github.com/Nixtla/statsforecast  (Apache-2.0)
- mlforecast:    https://github.com/Nixtla/mlforecast     (Apache-2.0)
- neuralforecast: https://github.com/Nixtla/neuralforecast (Apache-2.0)

The calibration procedure, the two interval construction methods
(``conformal_distribution`` and ``conformal_error``), the per-horizon-step
quantile logic, and the output column naming convention (``<model>-lo-<level>``
/ ``<model>-hi-<level>``) follow Nixtla's design. See the individual
repositories for the original implementations.

Description
-----------
1. **Calibration** runs a rolling-window cross-validation over the training data
   and collects the *absolute residuals* ``|y_hat - y|`` (the conformity score)
   for each window, series, and forecast-horizon step.
2. **At prediction time**, for each confidence ``level`` the intervals are built
   from per-horizon quantiles of the conformity scores, using one of two methods:

   - ``conformal_distribution`` (Nixtla's default): build synthetic forecast paths
     ``[y_hat - scores, y_hat + scores]`` and take the symmetric
     ``[alpha/200, 1 - alpha/200]`` quantiles, where ``alpha = 100 - level``.
   - ``conformal_error``: take the ``level/100`` quantile of the absolute
     residuals and form ``y_hat +/- q``.

Quantiles are computed independently per horizon step and per series, matching
Nixtla's implementation.

The module is intentionally model-agnostic: it only relies on a ``model_factory``
that returns a fresh model exposing the standard Hyper-Tree ``train`` / ``forecast``
interface, so it can be reused for the other Hyper-Tree models in the future.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

_VALID_METHODS = ("conformal_distribution", "conformal_error")


@dataclass
class ForecastIntervals:
    """Configuration for conformal prediction intervals.

    Parameters
    ----------
    n_windows : int
        Number of rolling cross-validation windows used to collect conformity
        scores. Must be at least 2. More windows give a more stable calibration
        at the cost of additional refits.
    method : str
        Interval construction method, either ``"conformal_distribution"`` (default)
        or ``"conformal_error"``.
    step_size : int
        Step (in time steps) between consecutive cross-validation windows. The
        default of 1 produces maximally overlapping windows and therefore the most
        conformity scores for a given series length.
    refit : bool
        If ``True`` (default), a fresh model is trained for every CV window
        (rolling-origin evaluation). If ``False``, a single model is trained on
        the oldest window's training split and reused to forecast all windows,
        matching Nixtla's ``mlforecast`` behaviour. ``refit=False`` is
        substantially faster but may under-estimate errors for later windows
        whose training data the model never saw.

    Notes
    -----
    Calibration is always performed at the model's own forecast horizon
    (``fcst_h``), yielding per-horizon-step intervals.
    """

    n_windows: int = 5
    method: str = "conformal_distribution"
    step_size: int = 1
    refit: bool = True

    def __post_init__(self):
        if not isinstance(self.n_windows, int) or self.n_windows < 2:
            raise ValueError("n_windows must be an integer >= 2.")
        if self.method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}.")
        if not isinstance(self.step_size, int) or self.step_size < 1:
            raise ValueError("step_size must be a positive integer.")
        if not isinstance(self.refit, bool):
            raise ValueError("refit must be a boolean.")


def validate_calibration_length(
    train_data: pd.DataFrame,
    fcst_h: int,
    forecast_intervals: ForecastIntervals,
    min_train: int,
) -> None:
    """Validate that every series is long enough for the rolling-window calibration.

    Each series needs enough observations so that, in the oldest window, the
    training portion still has at least ``min_train`` rows after carving out the
    cross-validation test blocks.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data with a ``series_id`` column.
    fcst_h : int
        Forecast horizon (length of each cross-validation test block).
    forecast_intervals : ForecastIntervals
        Calibration configuration.
    min_train : int
        Minimum number of training rows required by the model (e.g. ``p + 1`` for
        an AR(p) model, so at least one training sample remains after lagging).

    Raises
    ------
    ValueError
        If any series is too short.
    """
    pi = forecast_intervals
    needed = fcst_h + (pi.n_windows - 1) * pi.step_size + min_train
    lengths = train_data.groupby("series_id", sort=False).size()
    bad = lengths[lengths < needed]
    if len(bad) > 0:
        raise ValueError(
            f"Conformal calibration with n_windows={pi.n_windows}, "
            f"step_size={pi.step_size}, fcst_h={fcst_h} requires at least "
            f"{needed} observations per series, but these series are too short: "
            f"{bad.to_dict()}. Reduce n_windows/step_size or provide longer series."
        )


def rolling_origin_residuals(
    model_factory: Callable[[], object],
    train_data: pd.DataFrame,
    fcst_h: int,
    forecast_intervals: ForecastIntervals,
    train_kwargs: dict,
) -> Tuple[np.ndarray, List]:
    """Collect absolute-residual conformity scores via rolling-window CV.

    For each window ``w = 0, ..., n_windows - 1`` the test block is the ``fcst_h``
    observations ending at ``L - w * step_size`` (per series), and the model is
    trained on all earlier observations.

    When ``forecast_intervals.refit`` is ``True`` (default) a fresh model is
    trained for every window. When ``False``, a single model is trained on the
    oldest window's training split and reused to forecast all windows. Before
    each window's forecast the model's forecast seed (lags, states, etc.) is
    re-anchored to the window's own history via ``set_forecast_origin``, so
    that residuals reflect the correct origin — only the GBDT refit is skipped.

    Parameters
    ----------
    model_factory : Callable[[], object]
        Zero-argument callable returning a fresh, untrained model exposing
        ``train(train_data=..., **train_kwargs)``,
        ``forecast(test_data=..., type="forecast")``, and
        ``set_forecast_origin(history: pd.DataFrame)`` (required for
        ``refit=False``; re-anchors the forecast seed without retraining).
    train_data : pd.DataFrame
        Full training data (``series_id``, ``date``, ``value`` + features).
    fcst_h : int
        Forecast horizon / length of each CV test block.
    forecast_intervals : ForecastIntervals
        Calibration configuration.
    train_kwargs : dict
        Keyword arguments forwarded to each fresh model's ``train`` call (e.g.
        ``lgb_params``, ``num_iterations``, ``seed``). Must not contain
        ``train_data`` or ``forecast_intervals``.

    Returns
    -------
    scores : np.ndarray
        Absolute residuals with shape ``(n_windows, n_series, fcst_h)``. If the
        data carries a ``mask`` column, residuals at padded rows (``mask == 0``)
        are NaN and are excluded from the interval quantiles downstream.
    series_order : list
        Series ids in first-appearance order (axis 1 of ``scores``).
    """
    pi = forecast_intervals
    series_order = list(dict.fromkeys(train_data["series_id"].tolist()))
    grouped = {sid: g for sid, g in train_data.groupby("series_id", sort=False)}

    scores = np.empty((pi.n_windows, len(series_order), fcst_h), dtype=float)

    # When refit=False, train once on the oldest window (w = n_windows - 1,
    # which has the least training data) so the model never sees any of the
    # test observations across all windows.
    shared_model = None
    if not pi.refit:
        oldest_offset = (pi.n_windows - 1) * pi.step_size
        train_parts = []
        for sid in series_order:
            g = grouped[sid]
            start = len(g) - oldest_offset - fcst_h
            train_parts.append(g.iloc[:start])
        oldest_train_df = pd.concat(train_parts, ignore_index=True)
        shared_model = model_factory()
        shared_model.train(train_data=oldest_train_df, **train_kwargs)

    for w in range(pi.n_windows):
        offset = w * pi.step_size
        train_parts, test_parts = [], []
        for sid in series_order:
            g = grouped[sid]
            end = len(g) - offset
            start = end - fcst_h
            train_parts.append(g.iloc[:start])
            test_parts.append(g.iloc[start:end])

        test_df = pd.concat(test_parts, ignore_index=True)

        if pi.refit:
            train_df = pd.concat(train_parts, ignore_index=True)
            model = model_factory()
            model.train(train_data=train_df, **train_kwargs)
        else:
            model = shared_model
            window_train_df = pd.concat(train_parts, ignore_index=True)
            model.set_forecast_origin(window_train_df)

        fcst = model.forecast(test_data=test_df, type="forecast")

        # Residuals are computed positionally; enforce the row-order contract
        # (one forecast row per input row, in input order) so a model that
        # reorders or reshapes its output fails loudly instead of silently
        # mis-assigning residuals across series.
        if not (
            np.array_equal(
                fcst["series_id"].to_numpy(), test_df["series_id"].to_numpy()
            )
            and np.array_equal(
                pd.to_datetime(fcst["date"]).to_numpy(),
                pd.to_datetime(test_df["date"]).to_numpy(),
            )
        ):
            raise RuntimeError(
                "model.forecast() returned rows in a different order than "
                "test_data. rolling_origin_residuals computes residuals "
                "positionally and requires one forecast row per input row, "
                "in input order."
            )

        resid = np.abs(fcst["fcst"].to_numpy() - test_df["value"].to_numpy())
        if "mask" in test_df.columns:
            # Padded pseudo-observations (mask == 0, used by HyperTreeETS for
            # uniform series lengths) carry no information about real forecast
            # errors; mark them NaN so the NaN-aware interval quantiles ignore
            # them.
            resid = np.where(test_df["mask"].to_numpy().astype(bool), resid, np.nan)
        scores[w] = resid.reshape(len(series_order), fcst_h)

    return scores, series_order


def _align_scores(
    scores: np.ndarray, cal_order: Sequence, target_order: Sequence
) -> np.ndarray:
    """Reorder the series axis of ``scores`` to match ``target_order``."""
    cal_order = list(cal_order)
    target_order = list(target_order)
    if cal_order == target_order:
        return scores
    missing = set(target_order) - set(cal_order)
    if missing:
        raise ValueError(
            f"Series {missing} were not seen during conformal calibration."
        )
    idx = [cal_order.index(s) for s in target_order]
    return scores[:, idx, :]


def _distribution_bands(
    point: np.ndarray, scores: np.ndarray, levels: List[int]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """``conformal_distribution`` intervals (synthetic-path symmetric quantiles).

    NaN scores (residuals at padded pseudo-observations) are excluded via
    NaN-aware quantiles; a cell whose scores are all NaN yields NaN bounds.
    """
    # Synthetic forecast paths: (2 * n_windows, n_series, h)
    paths = np.concatenate([point[None] - scores, point[None] + scores], axis=0)
    bands = {}
    for lv in levels:
        alpha = 100 - lv
        lo = np.nanquantile(paths, (alpha / 2) / 100.0, axis=0)
        hi = np.nanquantile(paths, 1.0 - (alpha / 2) / 100.0, axis=0)
        bands[lv] = (lo, hi)
    return bands


def _error_bands(
    point: np.ndarray, scores: np.ndarray, levels: List[int]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """``conformal_error`` intervals (symmetric ``y_hat +/- quantile``).

    NaN scores (residuals at padded pseudo-observations) are excluded via
    NaN-aware quantiles; a cell whose scores are all NaN yields NaN bounds.
    """
    bands = {}
    for lv in levels:
        q = np.nanquantile(scores, lv / 100.0, axis=0)
        bands[lv] = (point - q, point + q)
    return bands


def interval_columns(
    point: np.ndarray,
    scores: np.ndarray,
    levels: List[int],
    method: str,
    model_name: str,
    cal_order: Sequence,
    target_order: Sequence,
) -> "Dict[str, np.ndarray]":
    """Build the ``<model>-lo-<level>`` / ``<model>-hi-<level>`` columns.

    Parameters
    ----------
    point : np.ndarray
        Point forecasts shaped ``(n_series, fcst_h)`` in ``target_order``.
    scores : np.ndarray
        Conformity scores shaped ``(n_windows, n_series, fcst_h)`` in ``cal_order``.
    levels : list of int
        Confidence levels in ``(0, 100)``.
    method : str
        ``"conformal_distribution"`` or ``"conformal_error"``.
    model_name : str
        Prefix for the interval columns (the model's ``model`` string).
    cal_order, target_order : sequence
        Series order of ``scores`` and of the desired output, respectively.

    Returns
    -------
    dict of str -> np.ndarray
        Ordered mapping of column name to a flat ``(n_series * fcst_h,)`` array,
        flattened series-major to match the forecast DataFrame's row order.
    """
    levels = sorted(int(lv) for lv in levels)
    for lv in levels:
        if not 0 < lv < 100:
            raise ValueError(f"level values must be in (0, 100); got {lv}.")

    # With few calibration windows, high-level tail quantiles sit at the
    # extremes of the available scores and the intervals will undercover.
    n_windows = scores.shape[0]
    for lv in levels:
        if n_windows * (100 - lv) < 100:
            warnings.warn(
                f"level={lv} requires tail quantiles beyond the resolution of "
                f"n_windows={n_windows} conformity scores per series and "
                f"horizon step; the bounds then sit at the extremes of the "
                f"scores and the interval will likely undercover. Increase "
                f"ForecastIntervals(n_windows=...) or request a lower level."
            )

    scores = _align_scores(scores, cal_order, target_order)

    if method == "conformal_distribution":
        bands = _distribution_bands(point, scores, levels)
    elif method == "conformal_error":
        bands = _error_bands(point, scores, levels)
    else:
        raise ValueError(f"method must be one of {_VALID_METHODS}.")

    columns: Dict[str, np.ndarray] = {}
    # Mirror Nixtla column ordering: lower bounds (widest first), then upper bounds.
    for lv in sorted(levels, reverse=True):
        columns[f"{model_name}-lo-{lv}"] = bands[lv][0].reshape(-1)
    for lv in sorted(levels):
        columns[f"{model_name}-hi-{lv}"] = bands[lv][1].reshape(-1)
    return columns
