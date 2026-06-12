"""Tests for the Hyper-Tree-TSB intermittent demand model."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hypertrees.conformal import ForecastIntervals
from hypertrees.models.HyperTreeTSB import HyperTreeTSB

K, T, FCST_H = 3, 80, 6
LGB_PARAMS = {"learning_rate": 0.1, "min_data_in_leaf": 5}


def make_panel(k=K, n_train=T, fcst_h=FCST_H, seed=0, demand_prob=0.4):
    """Aligned intermittent-demand panel: Bernoulli occurrence x Poisson size."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-06", periods=n_train + fcst_h, freq="W-MON")

    frames = []
    for i in range(k):
        occurrence = rng.binomial(1, demand_prob, len(dates))
        sizes = rng.poisson(5 + 3 * i, len(dates)) + 1
        frames.append(pd.DataFrame({
            "series_id": f"sku_{i}",
            "date": dates,
            "value": (occurrence * sizes).astype(float),
            "month": dates.month,
            "series_num": i,
        }))
    df = pd.concat(frames, ignore_index=True)
    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return train, test


def manual_tsb(y, alpha_p, alpha_d):
    """Reference TSB recursion (statsforecast convention): SES on the
    occurrence indicator, SES on nonzero demands (held between demands),
    fitted_t = p_{t-1} * z_{t-1}."""
    d = (y != 0).astype(float)
    p = d[0]
    nonzero = y[y != 0]
    z = nonzero[0] if len(nonzero) else 0.0
    fits = [y[0]]
    for t in range(1, len(y)):
        fits.append(p * z)
        p = p + alpha_p * (d[t] - p)
        if d[t] == 1:
            z = z + alpha_d * (y[t] - z)

    return np.array(fits), p, z


class TestTSBRecursion:
    def test_forward_matches_reference(self):
        """The forward pass with constant parameters must reproduce the
        classical TSB recursion (statsforecast convention)."""
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.n_series = 1
        rng = np.random.RandomState(0)
        y = (rng.binomial(1, 0.3, 60) * (rng.poisson(4, 60) + 1)).astype(float)
        y[0] = 3.0  # start with a demand so the init is unambiguous

        alpha_p, alpha_d = 0.2, 0.3
        target = torch.tensor(y.reshape(1, -1), dtype=torch.float32)
        mask = torch.ones_like(target)
        params = torch.tensor([alpha_p, alpha_d]).repeat(1, len(y), 1)

        p_T, z_T, fits = model.forward(params, target, mask)
        fits = torch.stack(fits, dim=1).numpy().ravel()

        ref_fits, ref_p, ref_z = manual_tsb(y, alpha_p, alpha_d)
        np.testing.assert_allclose(fits, ref_fits, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(p_T.item(), ref_p, rtol=1e-5)
        np.testing.assert_allclose(z_T.item(), ref_z, rtol=1e-5)

    def test_init_states(self):
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        target = torch.tensor([
            [0.0, 0.0, 5.0, 0.0],   # starts with no demand; first demand 5
            [3.0, 0.0, 2.0, 1.0],   # starts with a demand of 3
            [0.0, 0.0, 0.0, 0.0],   # all-zero series
        ])
        mask = torch.ones_like(target)
        p0, z0 = model._init_states(target, mask)
        np.testing.assert_allclose(p0.numpy(), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(z0.numpy(), [5.0, 3.0, 0.0])

    def test_masked_padding_does_not_change_states(self):
        """Back-padded pseudo-observations (mask == 0) must leave the final
        states untouched."""
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.n_series = 1
        rng = np.random.RandomState(1)
        y = (rng.binomial(1, 0.4, 50) * (rng.poisson(4, 50) + 1)).astype(float)
        pad = np.full(5, 99.0)  # padding values must be ignored entirely
        y_full = np.concatenate([y, pad])
        params = torch.tensor([0.2, 0.3]).repeat(1, len(y_full), 1)

        target_full = torch.tensor(y_full.reshape(1, -1), dtype=torch.float32)
        mask_padded = torch.ones_like(target_full)
        mask_padded[0, len(y):] = 0.0
        p_pad, z_pad, _ = model.forward(params, target_full, mask_padded)

        target_short = target_full[:, :len(y)]
        p_short, z_short, _ = model.forward(
            params[:, :len(y)], target_short, torch.ones_like(target_short)
        )

        np.testing.assert_allclose(p_pad.item(), p_short.item(), rtol=1e-6)
        np.testing.assert_allclose(z_pad.item(), z_short.item(), rtol=1e-6)


class TestTSBTraining:
    def test_train_forecast_shapes(self):
        train, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        result = model.train(lgb_params=LGB_PARAMS, num_iterations=20, train_data=train)
        assert result.training_time is not None

        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["fcst"] >= 0).all()
        assert (fcst["model"] == "Hyper-Tree-TSB").all()

    def test_forecast_is_flat_over_horizon(self):
        """Classical TSB point forecasts are flat: p_T * z_T for every step."""
        train, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        fcst = model.forecast(test_data=test)
        per_series_unique = fcst.groupby("series_id")["fcst"].nunique()
        assert (per_series_unique == 1).all()

    def test_all_zero_series_forecasts_zero(self):
        train, test = make_panel()
        train.loc[train["series_id"] == "sku_2", "value"] = 0.0
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        fcst = model.forecast(test_data=test)
        assert (fcst.loc[fcst["series_id"] == "sku_2", "fcst"] == 0).all()

    def test_parameters_output(self):
        train, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        params = model.forecast(test_data=test, type="parameters")
        assert list(params.columns) == ["series_id", "date", "model", "alpha_p", "alpha_d"]
        for col in ["alpha_p", "alpha_d"]:
            assert params[col].between(0, 1).all()

    def test_conformal_intervals(self):
        train, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS, num_iterations=10, train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo, hi = "Hyper-Tree-TSB-lo-80", "Hyper-Tree-TSB-hi-80"
        assert lo in fcst.columns and hi in fcst.columns
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()

    def test_validation_and_early_stopping(self):
        train, _ = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        result = model.train(
            lgb_params=LGB_PARAMS, num_iterations=20, train_data=train,
            validation=True, early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_unequal_series_lengths_raise(self):
        train, _ = make_panel()
        train = train.iloc[:-1]
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        with pytest.raises(ValueError, match="same length"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)

    def test_l1_loss_rejected(self):
        with pytest.raises(ValueError, match="L1Loss"):
            HyperTreeTSB(freq="W", fcst_h=FCST_H, loss_fn=nn.L1Loss())


def _trained(seed=0, **train_kwargs):
    """Train a small TSB model and return ``(model, test)``."""
    train, test = make_panel(seed=seed)
    model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
    model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train, **train_kwargs)
    return model, test


class TestTSBConstructorValidation:
    def test_invalid_fcst_h(self):
        with pytest.raises(ValueError, match="fcst_h"):
            HyperTreeTSB(freq="W", fcst_h=0)

    def test_loss_fn_not_module(self):
        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeTSB(freq="W", fcst_h=FCST_H, loss_fn="not_a_loss")

    def test_non_mse_loss_warns(self):
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeTSB(freq="W", fcst_h=FCST_H, loss_fn=nn.HuberLoss())

    def test_freq_not_string(self):
        with pytest.raises(TypeError, match="freq must be a string"):
            HyperTreeTSB(freq=123, fcst_h=FCST_H)


class TestTSBTrainValidation:
    def setup_method(self):
        self.model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        self.train, _ = make_panel()

    def test_train_data_none(self):
        with pytest.raises(ValueError, match="train_data must be provided"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=None)

    def test_lgb_params_none(self):
        with pytest.raises(ValueError, match="lgb_params must be provided"):
            self.model.train(lgb_params=None, train_data=self.train)

    def test_train_data_not_dataframe(self):
        with pytest.raises(TypeError, match="train_data must be a pandas DataFrame"):
            self.model.train(lgb_params=LGB_PARAMS, train_data="not_a_df")

    def test_lgb_params_not_dict(self):
        with pytest.raises(TypeError, match="lgb_params must be a dictionary"):
            self.model.train(lgb_params="x", train_data=self.train)

    def test_num_iterations_invalid(self):
        with pytest.raises(ValueError, match="num_iterations must be a positive integer"):
            self.model.train(lgb_params=LGB_PARAMS, num_iterations=0, train_data=self.train)

    def test_seed_not_int(self):
        with pytest.raises(TypeError, match="seed must be an integer"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, seed="x")

    def test_verbose_not_int(self):
        with pytest.raises(TypeError, match="verbose must be an integer"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, verbose="x")

    def test_early_stopping_round_invalid(self):
        with pytest.raises(ValueError, match="early_stopping_round must be a positive integer"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, early_stopping_round=0)

    def test_validation_not_bool(self):
        with pytest.raises(TypeError, match="validation must be a boolean"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, validation="x")

    def test_deterministic_not_bool(self):
        with pytest.raises(TypeError, match="deterministic must be a boolean"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, deterministic="x")

    def test_forecast_intervals_wrong_type(self):
        with pytest.raises(TypeError, match="forecast_intervals must be a ForecastIntervals"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train, forecast_intervals="x")

    def test_early_stopping_without_validation(self):
        with pytest.raises(ValueError, match="can only be used when validation is True"):
            self.model.train(
                lgb_params=LGB_PARAMS, train_data=self.train,
                early_stopping_round=5, validation=False,
            )

    def test_validation_without_early_stopping(self):
        with pytest.raises(ValueError, match="must be provided when validation is True"):
            self.model.train(
                lgb_params=LGB_PARAMS, train_data=self.train,
                validation=True, early_stopping_round=None,
            )

    def test_missing_required_column(self):
        with pytest.raises(ValueError, match="Required column 'value' not found"):
            self.model.train(lgb_params=LGB_PARAMS, train_data=self.train.drop(columns=["value"]))

    def test_deterministic_false_path(self):
        train, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train, deterministic=False)
        assert len(model.forecast(test_data=test)) == K * FCST_H

    def test_training_failure_wrapped(self):
        with patch("hypertrees.models.HyperTreeTSB.lgb.train", side_effect=Exception("boom")):
            with pytest.raises(RuntimeError, match="Training failed"):
                self.model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=self.train)


class TestTSBMask:
    def test_mask_column_is_used_and_autofilled(self):
        """A 'mask' column is consumed as the validity mask during training and
        auto-filled with ones at forecast time when absent from test_data."""
        train, test = make_panel()
        train = train.copy()
        train["mask"] = 1
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        assert "mask" in model.features
        fcst = model.forecast(test_data=test)  # test_data has no 'mask' -> auto-filled
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()


class TestTSBForecastValidation:
    def test_forecast_before_train(self):
        _, test = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.forecast(test_data=test)

    def test_missing_states(self):
        model, test = _trained()
        model.fcst_states = None
        with pytest.raises(RuntimeError, match="Final states not found"):
            model.forecast(test_data=test)

    def test_missing_required_column(self):
        model, test = _trained()
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            model.forecast(test_data=test.drop(columns=["date"]))

    def test_series_mismatch(self):
        model, test = _trained()
        test = test.copy()
        test.loc[test["series_id"] == "sku_0", "series_id"] = "sku_X"
        with pytest.raises(ValueError, match="series in test_data"):
            model.forecast(test_data=test)

    def test_wrong_rows_per_series(self):
        model, test = _trained()
        with pytest.raises(ValueError, match="exactly fcst_h"):
            model.forecast(test_data=test.drop(test.index[0]))

    def test_invalid_type(self):
        model, test = _trained()
        with pytest.raises(ValueError, match="must be either 'forecast' or 'parameters'"):
            model.forecast(test_data=test, type="bogus")

    def test_level_with_parameters_type(self):
        model, test = _trained()
        with pytest.raises(ValueError, match="level is only supported"):
            model.forecast(test_data=test, type="parameters", level=[80])

    def test_level_without_calibration(self):
        model, test = _trained()
        with pytest.raises(RuntimeError, match="was not calibrated"):
            model.forecast(test_data=test, level=[80])

    def test_level_not_a_list(self):
        model, test = _trained(forecast_intervals=ForecastIntervals(n_windows=2, refit=False))
        with pytest.raises(ValueError, match="non-empty list"):
            model.forecast(test_data=test, level=80)

    def test_level_out_of_range(self):
        model, test = _trained(forecast_intervals=ForecastIntervals(n_windows=2, refit=False))
        with pytest.raises(ValueError, match="integers in"):
            model.forecast(test_data=test, level=[150])

    def test_missing_feature_wrapped(self):
        """A missing feature raises inside the forecast try-block, surfacing as
        the wrapped RuntimeError (also exercises the generic except path)."""
        model, test = _trained()
        with pytest.raises(RuntimeError, match="Missing features"):
            model.forecast(test_data=test.drop(columns=["month"]))


class TestTSBRecursiveValidationMetric:
    """The validation metric is the recursive h-step forecast (flat p_T * z_T
    scored against the holdout), not the degenerate in-sample one-step fit.

    The naive in-sample validation loss collapses to a parameter-independent
    fixed point (~0 for any parameters), so early stopping would select noise.
    These tests pin the fix: the validation curve is on a meaningful scale and
    varies with the boosted states, and the recursive branch reproduces the
    shared flat-forecast rollout used by ``forecast``.
    """

    def test_validation_metric_is_not_degenerate(self):
        train, _ = make_panel()
        model = HyperTreeTSB(freq="W", fcst_h=FCST_H)
        result = model.train(
            lgb_params=LGB_PARAMS, num_iterations=25, train_data=train,
            validation=True, early_stopping_round=999,
        )
        vals = np.asarray(next(iter(result.validation_metrics.values())), dtype=float)
        assert vals.size >= 5
        assert np.all(np.isfinite(vals))
        # Not the ~1e-12 degenerate in-sample metric: forecasting flat demand
        # against an intermittent holdout gives a clearly non-zero MSE.
        assert vals.min() > 1e-3
        # The metric depends on the boosted terminal states, so it moves.
        assert np.ptp(vals) > 1e-9

    def test_eval_fn_recursive_branch_matches_flat_forecast(self):
        model = HyperTreeTSB(freq="W", fcst_h=3)
        model.n_series = 2
        last_p = torch.tensor([0.3, 0.5])
        last_z = torch.tensor([6.0, 4.0])
        model._eval_boundary = (last_p, last_z)

        val_df = pd.DataFrame({"month": [1, 2, 3, 1, 2, 3],
                               "mask": [1, 1, 1, 1, 1, 1]})
        target = np.array([[0.0, 7.0, 0.0], [5.0, 0.0, 8.0]])
        mock_eval = Mock()
        mock_eval.data = val_df
        mock_eval.get_label.return_value = target.reshape(-1)
        model.dataset_references = {id(mock_eval): "validation"}

        # predt is ignored by the flat TSB forecast (independent of horizon params).
        predt = np.zeros(model.n_series * model.fcst_h * model.n_params)
        name, loss_val, is_higher_better = model.eval_fn(predt, mock_eval)

        point = model._roll_forecast(last_p, last_z, 3)
        exp_loss = nn.MSELoss()(point, torch.tensor(target, dtype=torch.float32)).item()

        assert name == "MSELoss"
        assert is_higher_better is False
        assert abs(loss_val - exp_loss) < 1e-5

    def test_validation_mask_excludes_padding(self):
        """Padded holdout rows (mask == 0) must not contribute to the metric."""
        model = HyperTreeTSB(freq="W", fcst_h=3)
        model.n_series = 1
        model._eval_boundary = (torch.tensor([0.4]), torch.tensor([5.0]))

        val_df = pd.DataFrame({"month": [1, 2, 3], "mask": [1, 1, 0]})
        target = np.array([[3.0, 0.0, 999.0]])  # last row padded; must be ignored
        mock_eval = Mock()
        mock_eval.data = val_df
        mock_eval.get_label.return_value = target.reshape(-1)
        model.dataset_references = {id(mock_eval): "validation"}

        _, loss_val, _ = model.eval_fn(np.zeros(model.n_params * 3), mock_eval)

        point = float((0.4 * 5.0))
        # Only the two valid rows count (3 and 0); the padded 999 is masked out.
        expected = ((point - 3.0) ** 2 + (point - 0.0) ** 2 + 0.0) / 3.0
        assert abs(loss_val - expected) < 1e-5
