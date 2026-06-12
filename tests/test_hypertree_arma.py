"""Tests for the Hyper-Tree-ARMA model (Hannan-Rissanen two-stage)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hypertrees.conformal import ForecastIntervals
from hypertrees.models import HyperTreeARMA

K, T, FCST_H = 2, 80, 6
P, Q, S1P = 2, 1, 4
LGB_PARAMS = {"learning_rate": 0.1, "min_data_in_leaf": 5}


def make_panel(k=K, n_train=T, fcst_h=FCST_H, seed=0):
    """Aligned panel of ARMA(1,1)-style series with a positive level."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2015-01-01", periods=n_train + fcst_h, freq="MS")

    frames = []
    for i in range(k):
        eps = rng.randn(len(dates))
        y = np.zeros(len(dates))
        for t in range(1, len(dates)):
            y[t] = 0.6 * y[t - 1] + eps[t] + 0.4 * eps[t - 1]
        frames.append(pd.DataFrame({
            "series_id": f"s{i}",
            "date": dates,
            "value": 50.0 + 5.0 * i + y,
            "month": dates.month,
            "series_num": i,
        }))
    df = pd.concat(frames, ignore_index=True)
    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return train, test


def make_model(**kwargs):
    defaults = dict(p=P, q=Q, freq="MS", fcst_h=FCST_H, stage1_p=S1P)
    defaults.update(kwargs)
    return HyperTreeARMA(**defaults)


def _trained(num_iterations=10, **train_kwargs):
    train, test = make_panel()
    model = make_model()
    model.train(
        lgb_params=LGB_PARAMS, num_iterations=num_iterations,
        train_data=train, **train_kwargs,
    )
    return model, train, test


class TestARMAInitialization:
    def test_default_initialization(self):
        model = HyperTreeARMA(p=3, q=2, freq="MS", fcst_h=FCST_H)
        assert model.n_params == 5
        assert model.stage1_p is None  # resolved at training time
        assert model.hessian_method == "analytic"
        assert model.is_trained is False

    def test_explicit_stage1_p_stored(self):
        model = make_model(stage1_p=7)
        assert model.stage1_p == 7

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="'p' must be a positive integer"):
            HyperTreeARMA(p=0, q=1)

    def test_invalid_q_points_to_ar(self):
        with pytest.raises(ValueError, match="HyperTreeAR"):
            HyperTreeARMA(p=2, q=0)

    def test_invalid_stage1_p(self):
        with pytest.raises(ValueError, match="stage1_p must be a positive integer"):
            HyperTreeARMA(p=2, q=1, stage1_p=0)

    def test_invalid_hessian_method(self):
        with pytest.raises(ValueError, match="hessian_method"):
            HyperTreeARMA(p=2, q=1, hessian_method="bogus")

    def test_l1_loss_rejected(self):
        with pytest.raises(ValueError, match="L1Loss"):
            HyperTreeARMA(p=2, q=1, loss_fn=nn.L1Loss())

    def test_gn_non_mse_warns(self):
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeARMA(p=2, q=1, hessian_method="gn", loss_fn=nn.HuberLoss())

    def test_invalid_fcst_h(self):
        with pytest.raises(ValueError, match="fcst_h"):
            HyperTreeARMA(p=2, q=1, fcst_h=0)


class TestStage1Resolution:
    def test_default_resolves_to_gomez_maravall(self):
        train, _ = make_panel()
        model = HyperTreeARMA(p=P, q=Q, freq="MS", fcst_h=FCST_H)  # stage1_p=None
        model.train(lgb_params=LGB_PARAMS, num_iterations=3, train_data=train)
        expected = max(int(np.floor(np.log(T) ** 2)), 2 * max(P, Q))
        assert model.stage1_p == expected
        assert model._stage1.p == expected

    def test_explicit_stage1_p_honored(self):
        model, _, _ = _trained(num_iterations=3)
        assert model.stage1_p == S1P
        assert model._stage1.p == S1P

    def test_too_short_series_raise(self):
        train, _ = make_panel()
        short = train.groupby("series_id", sort=False).head(8).reset_index(drop=True)
        model = make_model(stage1_p=10)
        with pytest.raises(ValueError, match="smaller stage1_p"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=3, train_data=short)


class TestARMARecursion:
    def test_residual_frame_nan_head(self):
        model, train, _ = _trained()
        work = model._stage1_residual_frame(train)
        for _, grp in work.groupby("series_id", sort=False):
            resid = grp["resid"].to_numpy()
            assert np.isnan(resid[:S1P]).all()
            assert np.isfinite(resid[S1P:]).all()

    def test_forecast_matches_manual_recursion(self):
        """The forecast must equal the manual ARMA recursion driven by the
        coefficients from type="parameters", the value seeds, and the
        residual seeds (with future innovations set to zero)."""
        model, _, test = _trained()
        fcst = model.forecast(test_data=test)
        params = model.forecast(test_data=test, type="parameters")

        for sid in test["series_id"].unique():
            pr = params[params["series_id"] == sid]
            phi = pr[[f"AR({j})" for j in range(1, P + 1)]].to_numpy()
            theta = pr[[f"MA({i})" for i in range(1, Q + 1)]].to_numpy()
            y_state = np.asarray(model.fcst_lags[sid], dtype=float).copy()
            e_state = np.asarray(model.fcst_eps[sid], dtype=float).copy()

            manual = []
            for h in range(FCST_H):
                val = float((phi[h] * y_state).sum() + (theta[h] * e_state).sum())
                manual.append(val)
                y_state = np.concatenate([[val], y_state[:-1]])
                e_state = np.concatenate([[0.0], e_state[:-1]])

            got = fcst.loc[fcst["series_id"] == sid, "fcst"].to_numpy()
            np.testing.assert_allclose(got, manual, rtol=1e-4, atol=1e-6)

    def test_forecast_seeds_shapes(self):
        model, train, _ = _trained()
        for sid in train["series_id"].unique():
            assert len(model.fcst_lags[sid]) == P
            assert len(model.fcst_eps[sid]) == Q
            assert np.isfinite(model.fcst_eps[sid]).all()


class TestARMAGradients:
    def _inputs(self, n=40, seed=0):
        rng = np.random.RandomState(seed)
        predt = rng.randn(n * (P + Q))
        target = torch.tensor(rng.randn(n, 1), dtype=torch.float32)
        design = torch.tensor(rng.randn(n, P + Q), dtype=torch.float32)
        return predt, target, design

    def test_analytic_matches_exact(self):
        predt, target, design = self._inputs()

        m_exact = make_model(hessian_method="exact")
        p_e, l_e = m_exact.get_params_loss(predt.copy(), target.clone(), design.clone(), requires_grad=True)
        g_e, h_e = m_exact.calculate_gradients_and_hessians(l_e, p_e)

        m_analytic = make_model(hessian_method="analytic")
        p_a, l_a = m_analytic.get_params_loss(predt.copy(), target.clone(), design.clone(), requires_grad=True)
        g_a, h_a = m_analytic.calculate_gradients_and_hessians(l_a, p_a)

        np.testing.assert_allclose(g_a, g_e, rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(h_a, h_e, rtol=1e-4, atol=1e-7)

    def test_gn_gradients_match_and_hessians_nonnegative(self):
        predt, target, design = self._inputs()

        m_analytic = make_model(hessian_method="analytic")
        p_a, l_a = m_analytic.get_params_loss(predt.copy(), target.clone(), design.clone(), requires_grad=True)
        g_a, _ = m_analytic.calculate_gradients_and_hessians(l_a, p_a)

        m_gn = make_model(hessian_method="gn")
        m_gn._iter_count = 1
        p_g, l_g = m_gn.get_params_loss(predt.copy(), target.clone(), design.clone(), requires_grad=True)
        g_g, h_g = m_gn.calculate_gradients_and_hessians(l_g, p_g)

        np.testing.assert_allclose(g_g, g_a, rtol=1e-4, atol=1e-7)
        assert (h_g >= 0).all()


class TestARMATraining:
    def test_train_forecast_shapes(self):
        model, _, test = _trained()
        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-Tree-ARMA({P},{Q})").all()

    def test_parameters_output(self):
        model, train, test = _trained()
        params = model.forecast(test_data=test, type="parameters")
        expected = (
            ["series_id", "date", "model"]
            + [f"AR({j})" for j in range(1, P + 1)]
            + [f"MA({i})" for i in range(1, Q + 1)]
        )
        assert list(params.columns) == expected
        # parameters can be requested for arbitrary-length input
        longer = train.groupby("series_id", sort=False).tail(10).reset_index(drop=True)
        params_long = model.forecast(test_data=longer, type="parameters")
        assert len(params_long) == K * 10

    def test_validation_and_early_stopping(self):
        train, _ = make_panel()
        model = make_model()
        result = model.train(
            lgb_params=LGB_PARAMS, num_iterations=20, train_data=train,
            validation=True, early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_conformal_intervals(self):
        model, _, test = _trained(
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo = f"Hyper-Tree-ARMA({P},{Q})-lo-80"
        hi = f"Hyper-Tree-ARMA({P},{Q})-hi-80"
        assert lo in fcst.columns and hi in fcst.columns
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()

    def test_set_forecast_origin_reanchors(self):
        model, train, test = _trained()
        # Concatenating the per-series train and test blocks interleaves the
        # series; re-sort so each series occupies one contiguous block.
        extended = (
            pd.concat([train, test], ignore_index=True)
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )
        model.set_forecast_origin(extended)
        for sid in test["series_id"].unique():
            tail = extended.loc[extended["series_id"] == sid, "value"].to_numpy()
            np.testing.assert_allclose(
                np.asarray(model.fcst_lags[sid], dtype=float), tail[-P:][::-1]
            )
            assert len(model.fcst_eps[sid]) == Q

    def test_set_forecast_origin_too_short_raises(self):
        model, train, _ = _trained()
        short = train.groupby("series_id", sort=False).head(3).reset_index(drop=True)
        with pytest.raises(ValueError, match="at least"):
            model.set_forecast_origin(short)

    def test_training_failure_wrapped(self):
        train, _ = make_panel()
        model = make_model()
        # The stage-1 HyperTreeAR trains first; failing its lgb.train must
        # surface as the ARMA's wrapped RuntimeError.
        with patch("hypertrees.models.HyperTreeAR.lgb.train", side_effect=Exception("boom")):
            with pytest.raises(RuntimeError, match="Training failed"):
                model.train(lgb_params=LGB_PARAMS, num_iterations=3, train_data=train)


class TestARMAForecastValidation:
    def test_forecast_before_train(self):
        _, test = make_panel()
        model = make_model()
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.forecast(test_data=test)

    def test_missing_required_column(self):
        model, _, test = _trained()
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            model.forecast(test_data=test.drop(columns=["date"]))

    def test_series_mismatch(self):
        model, _, test = _trained()
        test = test.copy()
        test.loc[test["series_id"] == "s0", "series_id"] = "sX"
        with pytest.raises(ValueError, match="series"):
            model.forecast(test_data=test)

    def test_wrong_rows_per_series(self):
        model, _, test = _trained()
        with pytest.raises(ValueError, match="exactly fcst_h"):
            model.forecast(test_data=test.drop(test.index[0]))

    def test_invalid_type(self):
        model, _, test = _trained()
        with pytest.raises(ValueError, match="'forecast' or 'parameters'"):
            model.forecast(test_data=test, type="bogus")

    def test_level_without_calibration(self):
        model, _, test = _trained()
        with pytest.raises(RuntimeError, match="was not calibrated"):
            model.forecast(test_data=test, level=[80])

    def test_level_not_a_list(self):
        model, _, test = _trained(
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        with pytest.raises(ValueError, match="non-empty list"):
            model.forecast(test_data=test, level=80)
