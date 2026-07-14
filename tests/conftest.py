"""Shared fixtures for the test suite."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def assert_forecast_preserves_dataframe():
    """Regression check for GitHub issue #11: forecast() must hand LightGBM a
    pandas DataFrame (not ``.values``), so ``category``-dtype features keep
    their categorical encoding at prediction time.

    Spies on the trained Booster's ``predict`` and asserts that every call
    receives a DataFrame in which the test data's categorical columns retain
    their ``category`` dtype.
    """

    def _assert(model, test, **forecast_kwargs):
        cat_cols = [
            col for col in test.columns
            if isinstance(test[col].dtype, pd.CategoricalDtype)
        ]
        with patch.object(model.model, "predict", wraps=model.model.predict) as spy:
            fcst = model.forecast(test_data=test, **forecast_kwargs)
        assert spy.call_count >= 1, "Booster.predict was never called"
        for call in spy.call_args_list:
            frame = call.args[0]
            assert isinstance(frame, pd.DataFrame), (
                f"forecast passed {type(frame)} to Booster.predict, not a DataFrame"
            )
            for col in cat_cols:
                if col in frame.columns:
                    assert isinstance(frame[col].dtype, pd.CategoricalDtype), (
                        f"feature '{col}' lost its category dtype on the way "
                        f"to Booster.predict"
                    )
        if "fcst" in fcst.columns:
            assert np.isfinite(fcst["fcst"]).all()

        return fcst

    return _assert
