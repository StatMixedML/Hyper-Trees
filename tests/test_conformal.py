"""Unit tests for the conformal prediction math (model-agnostic)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, call

from hypertrees import ForecastIntervals
from hypertrees.conformal import (
    interval_columns,
    rolling_origin_residuals,
    _distribution_bands,
    _error_bands,
)


def test_forecast_intervals_validation():
    with pytest.raises(ValueError):
        ForecastIntervals(n_windows=1)  # must be >= 2
    with pytest.raises(ValueError):
        ForecastIntervals(method="bogus")
    with pytest.raises(ValueError):
        ForecastIntervals(step_size=0)
    with pytest.raises(ValueError):
        ForecastIntervals(refit="yes")  # must be bool


def test_distribution_vs_error_bands_symmetry():
    rng = np.random.default_rng(0)
    point = rng.standard_normal((2, 3))
    scores = np.abs(rng.standard_normal((5, 2, 3)))
    # error bands are symmetric around the point forecast by construction
    bands = _error_bands(point, scores, [90])
    lo, hi = bands[90]
    np.testing.assert_allclose(hi - point, point - lo)
    # distribution bands: higher level => wider interval
    d80 = _distribution_bands(point, scores, [80])[80]
    d90 = _distribution_bands(point, scores, [90])[90]
    assert np.all(d90[0] <= d80[0] + 1e-9)  # lower bound wider
    assert np.all(d90[1] >= d80[1] - 1e-9)  # upper bound wider


def test_interval_columns_naming_and_ordering():
    point = np.zeros((2, 3))
    scores = np.ones((4, 2, 3))
    cols = interval_columns(
        point=point,
        scores=scores,
        levels=[80, 90],
        method="conformal_error",
        model_name="M",
        cal_order=[0, 1],
        target_order=[0, 1],
    )
    assert list(cols.keys()) == ["M-lo-90", "M-lo-80", "M-hi-80", "M-hi-90"]
    for v in cols.values():
        assert v.shape == (2 * 3,)


def test_refit_false_calls_set_forecast_origin():
    """rolling_origin_residuals with refit=False must re-anchor per window."""
    n_series, T, fcst_h = 2, 20, 3
    n_windows, step_size = 3, 1
    dates = pd.date_range("2020-01-01", periods=T, freq="MS")
    train_data = pd.concat([
        pd.DataFrame({
            "series_id": sid, "date": dates,
            "value": np.arange(T, dtype=float) + sid * 100,
        })
        for sid in range(n_series)
    ], ignore_index=True)

    # Build a mock model that returns deterministic forecasts
    mock_model = MagicMock()
    mock_model.set_forecast_origin = MagicMock()

    def fake_forecast(test_data, type="forecast"):
        return pd.DataFrame({
            "series_id": test_data["series_id"].values,
            "date": test_data["date"].values,
            "fcst": test_data["value"].values + 0.5,
            "model": "mock",
        })

    mock_model.forecast = MagicMock(side_effect=fake_forecast)
    mock_model.train = MagicMock()

    pi = ForecastIntervals(n_windows=n_windows, step_size=step_size, refit=False)
    scores, series_order = rolling_origin_residuals(
        model_factory=lambda: mock_model,
        train_data=train_data,
        fcst_h=fcst_h,
        forecast_intervals=pi,
        train_kwargs={},
    )

    # train() called once (oldest window)
    assert mock_model.train.call_count == 1
    # set_forecast_origin called once per window
    assert mock_model.set_forecast_origin.call_count == n_windows
    # scores shape is correct
    assert scores.shape == (n_windows, n_series, fcst_h)
    # residuals should all be 0.5 (our fake_forecast adds 0.5)
    np.testing.assert_allclose(scores, 0.5)
