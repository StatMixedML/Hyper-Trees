"""Unit tests for the conformal prediction math (model-agnostic)."""

import numpy as np
import pytest

from hypertrees import ForecastIntervals
from hypertrees.conformal import (
    interval_columns,
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
