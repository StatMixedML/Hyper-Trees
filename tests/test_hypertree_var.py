"""Tests for the Hyper-Tree VAR models (HyperTreeVAR / HyperTreeNetVAR)."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hypertrees.conformal import ForecastIntervals
from hypertrees.models._var_base import (
    _HyperTreeVARBase,
    _build_var_lags,
    _validate_aligned_dates,
)
from hypertrees.models.HyperTreeVAR import HyperTreeVAR
from hypertrees.models.HyperTreeNetVAR import HyperTreeNetVAR

K, T, P, FCST_H = 3, 60, 2, 6
LGB_PARAMS = {"learning_rate": 0.1, "min_data_in_leaf": 10}
NET_PARAMS = {
    "learning_rate": 1e-3,
    "embedding_dimension": 1,
    "hidden_dim": 16,
    "dropout": 0.0,
    "use_random_projection": True,
    "rp_embed_dim": 8,
}


def make_panel(k=K, n_train=T, fcst_h=FCST_H, seed=0):
    """Stable VAR(1)-generated aligned panel with month/quarter features."""
    rng = np.random.RandomState(seed)
    A = 0.5 * np.eye(k) + 0.1 / k
    dates = pd.date_range("2015-01-01", periods=n_train + fcst_h, freq="MS")
    Y = np.zeros((len(dates), k))
    Y[0] = 100 + 10 * rng.rand(k)
    const = 50.0 * (np.eye(k) - A).sum(axis=1)
    for t in range(1, len(dates)):
        Y[t] = const + A @ Y[t - 1] + rng.randn(k)

    frames = []
    for i in range(k):
        frames.append(pd.DataFrame({
            "series_id": f"s{i}",
            "date": dates,
            "value": Y[:, i],
            "month": dates.month,
            "quarter": dates.quarter,
            "series_num": i,  # identifies the series, so equations can differ
        }))
    df = pd.concat(frames, ignore_index=True)
    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)
    return train, test


class FakeDataset:
    def __init__(self, label):
        self._label = label

    def get_label(self):
        return self._label


class TestBuildVarLags:
    def test_design_matrix_layout(self):
        Y = np.arange(12, dtype=float).reshape(4, 3)  # T=4, k=3
        Z = _build_var_lags(Y, p=2)
        assert Z.shape == (2, 6)
        # Row r corresponds to t = r + p: [y'_{t-1}, y'_{t-2}]
        np.testing.assert_array_equal(Z[0], np.concatenate([Y[1], Y[0]]))
        np.testing.assert_array_equal(Z[1], np.concatenate([Y[2], Y[1]]))


class TestAnalyticDerivatives:
    @pytest.mark.parametrize("hessian_method", ["analytic", "exact"])
    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.HuberLoss()])
    def test_gradients_and_hessians_match_autograd(self, loss_fn, hessian_method):
        torch.manual_seed(0)
        k, p, t_r = 3, 2, 11
        kp = k * p
        model = HyperTreeVAR(p=p, fcst_h=1, loss_fn=loss_fn, hessian_method=hessian_method)
        model.k = k
        model.n_params = kp
        Z = torch.randn(t_r, kp)
        model._Z_train = Z

        rng = np.random.RandomState(0)
        label = rng.randn(k * t_r)
        predt = rng.randn(k * t_r * kp)

        grad, hess = model.objective_fn(predt, FakeDataset(label))

        # Reference: per-parameter second-order autograd through the VAR fit.
        params = torch.tensor(
            predt.reshape(-1, kp, order="F"), dtype=torch.float32, requires_grad=True
        )
        fit = (params.reshape(k, t_r, kp) * Z.unsqueeze(0)).sum(dim=2)
        loss = loss_fn(fit, torch.tensor(label.reshape(k, t_r), dtype=torch.float32))
        g_ref = torch.autograd.grad(loss, params, create_graph=True)[0]
        h_ref = torch.stack(
            [
                torch.autograd.grad(g_ref[:, i].sum(), params, retain_graph=True)[0][:, i]
                for i in range(kp)
            ],
            dim=1,
        )

        np.testing.assert_allclose(
            grad, g_ref.detach().numpy().ravel(order="F"), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            hess, h_ref.detach().numpy().ravel(order="F"), rtol=1e-4, atol=1e-5
        )

    def test_gn_gradients_exact_and_hessians_unbiased(self):
        torch.manual_seed(0)
        k, p, t_r = 3, 2, 11
        kp = k * p
        model = HyperTreeVAR(p=p, fcst_h=1, hessian_method="gn", n_hessian_probes=200)
        model.k = k
        model.n_params = kp
        Z = torch.randn(t_r, kp)
        model._Z_train = Z

        rng = np.random.RandomState(0)
        label = rng.randn(k * t_r)
        predt = rng.randn(k * t_r * kp)
        grad, hess = model.objective_fn(predt, FakeDataset(label))

        # Gradients are exact regardless of the Hessian approximation.
        params = torch.tensor(
            predt.reshape(-1, kp, order="F"), dtype=torch.float32, requires_grad=True
        )
        fit = (params.reshape(k, t_r, kp) * Z.unsqueeze(0)).sum(dim=2)
        loss = nn.MSELoss()(fit, torch.tensor(label.reshape(k, t_r), dtype=torch.float32))
        g_ref = torch.autograd.grad(loss, params)[0].numpy().ravel(order="F")
        np.testing.assert_allclose(grad, g_ref, rtol=1e-4, atol=1e-6)

        # The GN Hessian is PSD and (for the linear VAR fit with MSE) an
        # unbiased Hutchinson estimate of the exact diagonal.
        assert (hess >= 0).all()
        h_true = (2.0 / (k * t_r)) * np.tile(Z.numpy() ** 2, (k, 1)).ravel(order="F")
        assert abs(hess.mean() / h_true.mean() - 1) < 0.15


class TestHyperTreeVAR:
    def test_train_forecast_shapes(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, freq="M", fcst_h=FCST_H)
        result = model.train(lgb_params=LGB_PARAMS, num_iterations=15, train_data=train)
        assert result.training_time is not None

        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-Tree-VAR({P})").all()
        # Rough sanity: forecasts should be in the ballpark of the series level.
        assert fcst["fcst"].between(0, 1000).all()

    def test_parameters_output(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        params = model.forecast(test_data=test, type="parameters")
        coef_cols = [
            f"A{j}(s{m})" for j in range(1, P + 1) for m in range(K)
        ]
        assert list(params.columns) == ["series_id", "date", "model"] + coef_cols
        assert len(params) == K * FCST_H

    def test_forecast_recursion_matches_manual(self):
        # scaling=None so the manual reconstruction can work in raw space.
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling=None)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)

        fcst = model.forecast(test_data=test)
        params = model.forecast(test_data=test, type="parameters")
        coef_cols = [f"A{j}(s{m})" for j in range(1, P + 1) for m in range(K)]
        Pmat = params[coef_cols].to_numpy().reshape(K, FCST_H, K * P)

        Y_train = train["value"].to_numpy().reshape(K, T).T  # (T, k)
        z = np.concatenate([Y_train[-j] for j in range(1, P + 1)])
        manual = np.empty((K, FCST_H))
        for h in range(FCST_H):
            y_next = Pmat[:, h, :] @ z
            manual[:, h] = y_next
            z = np.concatenate([y_next, z[:-K]])

        np.testing.assert_allclose(fcst["fcst"].to_numpy(), manual.reshape(-1), rtol=1e-10)

    def test_recovers_classical_var_ols(self):
        """With only the series-identity feature the parameters cannot vary
        over time, so the model degenerates to a constant-coefficient VAR and
        boosting must converge to the per-equation OLS solution (which equals
        the joint GLS/SUR estimate since all equations share the regressors).
        Verified against statsmodels VAR(trend="n") with matching results;
        the reference here is the equivalent lstsq solve to avoid the
        dependency."""
        rng = np.random.RandomState(7)
        k, p, n_train, h = 3, 2, 300, 6
        A1 = np.array([[0.5, 0.2, 0.0], [0.1, 0.4, 0.15], [0.0, 0.25, 0.45]])
        A2 = np.array([[-0.2, 0.0, 0.1], [0.05, -0.15, 0.0], [0.1, 0.0, -0.1]])
        Y = np.zeros((n_train + h, k))
        for t in range(2, n_train + h):
            Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + 0.5 * rng.randn(k)
        Y_train = Y[:n_train]
        dates = pd.date_range("1990-01-01", periods=n_train + h, freq="MS")

        df = pd.concat(
            [pd.DataFrame({"series_id": f"s{i}", "date": dates[:n_train],
                           "value": Y_train[:, i], "series_num": i})
             for i in range(k)],
            ignore_index=True,
        )
        test = pd.concat(
            [pd.DataFrame({"series_id": f"s{i}", "date": dates[n_train:], "series_num": i})
             for i in range(k)],
            ignore_index=True,
        )

        # scaling=None: the reference OLS coefficients live in raw space.
        model = HyperTreeVAR(p=p, fcst_h=h, scaling=None)
        model.train(
            lgb_params={"learning_rate": 0.05, "min_data_in_leaf": 5},
            num_iterations=2000,
            train_data=df,
        )

        # Per-equation OLS on the same design matrix (statsmodels trend="n").
        Z = _build_var_lags(Y_train, p)
        ols = np.stack(
            [np.linalg.lstsq(Z, Y_train[p:, i], rcond=None)[0].reshape(p, k) for i in range(k)]
        )  # (equation i, lag j, source series m)

        params = model.forecast(test_data=test, type="parameters")
        ours = np.stack(
            [params[params.series_id == f"s{i}"].iloc[0, 3:].to_numpy(dtype=float).reshape(p, k)
             for i in range(k)]
        )

        np.testing.assert_allclose(ours, ols, atol=5e-3)

    def test_validation_and_early_stopping(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        result = model.train(
            lgb_params=LGB_PARAMS,
            num_iterations=20,
            train_data=train,
            validation=True,
            early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_conformal_intervals(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS,
            num_iterations=10,
            train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo, hi = f"Hyper-Tree-VAR({P})-lo-80", f"Hyper-Tree-VAR({P})-hi-80"
        assert lo in fcst.columns and hi in fcst.columns
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()

    def test_level_without_calibration_raises(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        with pytest.raises(RuntimeError, match="not calibrated"):
            model.forecast(test_data=test, level=[80])

    def test_unequal_series_lengths_raise(self):
        train, _ = make_panel()
        train = train.iloc[:-1]  # truncate the last series by one row
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="same length"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)

    def test_misaligned_dates_raise(self):
        train, _ = make_panel()
        shifted = train["series_id"] == "s2"
        train.loc[shifted, "date"] = train.loc[shifted, "date"] + pd.DateOffset(days=1)
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="identical dates"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)

    def test_forecast_series_mismatch_raises(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        with pytest.raises(ValueError, match="Extra series not in test_data"):
            model.forecast(test_data=test[test["series_id"] != "s2"])

    def test_hessian_method_validation(self):
        with pytest.raises(ValueError, match="must be one of"):
            HyperTreeVAR(p=P, hessian_method="newton")
        with pytest.raises(ValueError, match="n_hessian_probes"):
            HyperTreeVAR(p=P, hessian_method="gn", n_hessian_probes=0)
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeVAR(p=P, hessian_method="gn", loss_fn=nn.HuberLoss())

    @pytest.mark.parametrize("method", ["exact", "gn"])
    def test_alternative_hessian_methods_train(self, method):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, hessian_method=method)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()

    def test_training_divergence_detection(self):
        model = HyperTreeVAR(p=2, fcst_h=1)
        model.k = 3
        model.n_params = 6
        model._Z_train = torch.full((11, 6), 1e30)
        predt = np.full(33 * 6, 1e10)  # fit overflows float32 -> loss inf
        with pytest.raises(RuntimeError, match="Training diverged"):
            model.objective_fn(predt, FakeDataset(np.zeros(33)))

    def test_forecast_divergence_detection(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        model._fcst_state = np.full(K * P, np.inf)
        with pytest.raises(RuntimeError, match="non-finite"):
            model.forecast(test_data=test)

    def test_l1_loss_rejected(self):
        with pytest.raises(ValueError, match="L1Loss"):
            HyperTreeVAR(p=P, loss_fn=nn.L1Loss())


class TestHyperTreeNetVAR:
    @pytest.mark.parametrize("hessian_method", ["exact", "gn"])
    def test_train_forecast_shapes(self, hessian_method):
        train, test = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H, hessian_method=hessian_method)
        model.train(
            lgb_params=LGB_PARAMS,
            network_params=NET_PARAMS,
            num_iterations=15,
            train_data=train,
        )
        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-TreeNet-VAR({P})").all()

    def test_parameters_and_embeddings_output(self):
        train, test = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS,
            network_params=NET_PARAMS,
            num_iterations=10,
            train_data=train,
        )
        params = model.forecast(test_data=test, type="parameters")
        assert f"A{P}(s{K - 1})" in params.columns
        embeds = model.forecast(test_data=test, type="tree_embeddings")
        assert "tree_embedding_1" in embeds.columns
        assert len(embeds) == K * FCST_H

    def test_conformal_intervals(self):
        train, test = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS,
            network_params=NET_PARAMS,
            num_iterations=10,
            train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80, 95])
        for col_suffix in ["lo-95", "lo-80", "hi-80", "hi-95"]:
            assert f"Hyper-TreeNet-VAR({P})-{col_suffix}" in fcst.columns

    def test_validation_and_early_stopping(self):
        train, _ = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        result = model.train(
            lgb_params=LGB_PARAMS,
            network_params=NET_PARAMS,
            num_iterations=15,
            train_data=train,
            validation=True,
            early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="hessian_method"):
            HyperTreeNetVAR(p=P, hessian_method="analytic")
        with pytest.raises(ValueError, match="n_hessian_probes"):
            HyperTreeNetVAR(p=P, hessian_method="gn", n_hessian_probes=0)
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeNetVAR(p=P, hessian_method="gn", loss_fn=nn.HuberLoss())

    def test_network_params_type_validation(self):
        train, _ = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(ValueError, match="network_params must be provided"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        with pytest.raises(TypeError, match="network_params must be a dictionary"):
            model.train(
                lgb_params=LGB_PARAMS, network_params="net",
                num_iterations=5, train_data=train,
            )

    def test_missing_network_params_raise(self):
        train, _ = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(ValueError, match="missing required keys"):
            model.train(
                lgb_params=LGB_PARAMS,
                network_params={"learning_rate": 1e-3},
                num_iterations=5,
                train_data=train,
            )


class TestBaseValidation:
    """Validation branches, hooks, and edge cases shared by both VAR models."""

    @pytest.mark.parametrize("kwargs,match", [
        ({"p": 0}, "'p' must be a positive integer"),
        ({"fcst_h": 0}, "'fcst_h' must be a positive integer"),
        ({"freq": 123}, "freq must be a string"),
        ({"loss_fn": "mse"}, "must be a PyTorch loss function"),
    ])
    def test_constructor_validation(self, kwargs, match):
        with pytest.raises((ValueError, TypeError), match=match):
            HyperTreeVAR(**kwargs)

    @pytest.mark.parametrize("overrides,match", [
        ({"train_data": None}, "train_data must be provided"),
        ({"lgb_params": None}, "lgb_params must be provided"),
        ({"train_data": "df"}, "train_data must be a pandas DataFrame"),
        ({"lgb_params": "params"}, "lgb_params must be a dictionary"),
        ({"num_iterations": 0}, "num_iterations must be a positive integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"verbose": "loud"}, "verbose must be an integer"),
        ({"validation": True, "early_stopping_round": 0},
         "early_stopping_round must be a positive integer"),
        ({"validation": "yes"}, "validation must be a boolean"),
        ({"deterministic": "yes"}, "deterministic must be a boolean"),
        ({"forecast_intervals": "conformal"}, "must be a ForecastIntervals instance"),
        ({"early_stopping_round": 5}, "can only be used when validation is True"),
        ({"validation": True}, "early_stopping_round must be provided"),
    ])
    def test_train_argument_validation(self, overrides, match):
        train, _ = make_panel()
        kwargs = dict(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        kwargs.update(overrides)
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises((ValueError, TypeError), match=match):
            model.train(**kwargs)

    def test_missing_required_column_raises(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(ValueError, match="Required column 'value'"):
            model.train(
                lgb_params=LGB_PARAMS, num_iterations=5,
                train_data=train.drop(columns="value"),
            )

    def test_base_hooks_are_abstract(self):
        base = _HyperTreeVARBase()
        with pytest.raises(NotImplementedError):
            base.objective_fn(None, None)
        with pytest.raises(NotImplementedError):
            base._fit_from_predt(None, None)
        with pytest.raises(NotImplementedError):
            base._forecast_params(None)
        with pytest.raises(NotImplementedError):
            base._num_class()
        with pytest.raises(NotImplementedError):
            base._forecast_tree_embeddings(None, None)
        assert base._post_datasets_setup(seed=0) is None  # default no-op

    def test_validate_aligned_dates_unequal_rows(self):
        df = pd.DataFrame({
            "series_id": ["a", "a", "b"],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
        })
        with pytest.raises(ValueError, match="same number of rows"):
            _validate_aligned_dates(df, name="test_data")

    def test_single_series_warns(self):
        train, _ = make_panel(k=1)
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.warns(UserWarning, match="consider HyperTreeAR instead"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)

    def test_series_shorter_than_lag_order_raises(self):
        train, _ = make_panel(n_train=P)
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="must exceed the lag order"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)

    def test_validation_split_too_short_raises(self):
        train, _ = make_panel(n_train=P + FCST_H)
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="Validation requires more than"):
            model.train(
                lgb_params=LGB_PARAMS, num_iterations=5, train_data=train,
                validation=True, early_stopping_round=2,
            )

    def test_build_panel_datasets_validation_without_early_stopping(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        valid_sets, valid_names, callbacks, evals_result = model._build_panel_datasets(
            train, validation=True, early_stopping_round=None
        )
        assert valid_names == ["train", "validation"]
        assert len(callbacks) == 1  # record_evaluation only, no early stopping

    def test_deterministic_false(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS, num_iterations=5, train_data=train,
            deterministic=False,
        )
        assert "deterministic" not in model.lgb_params
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()

    def test_category_dtype_features_roundtrip(self):
        """Identity features declared as pandas ``category`` dtype must survive
        training and forecasting for both models (LightGBM picks them up as
        true categoricals via categorical_feature='auto')."""
        train, test = make_panel()
        for df in (train, test):
            df["series_num"] = df["series_num"].astype("category")

        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()

        net = HyperTreeNetVAR(p=P, fcst_h=FCST_H)
        net.train(
            lgb_params=LGB_PARAMS, network_params=NET_PARAMS,
            num_iterations=10, train_data=train,
        )
        assert np.isfinite(net.forecast(test_data=test)["fcst"]).all()

    def test_eval_fn_unknown_dataset_warns(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(
            lgb_params=LGB_PARAMS, num_iterations=5, train_data=train,
            validation=True, early_stopping_round=2,
        )
        t_train = model._Z_train.shape[0]
        predt = np.zeros(K * t_train * K * P)
        with pytest.warns(UserWarning, match="Unknown dataset in metric_fn"):
            name, value, higher_better = model.eval_fn(
                predt, FakeDataset(np.zeros(K * t_train))
            )
        assert name == "MSELoss"
        assert higher_better is False

    def test_set_forecast_origin_validation(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="requires a trained model"):
            model.set_forecast_origin(train)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        with pytest.raises(ValueError, match="exactly the training series"):
            model.set_forecast_origin(train[train["series_id"] != "s2"])
        with pytest.raises(ValueError, match="at least p="):
            model.set_forecast_origin(
                train.groupby("series_id", sort=False).head(1).reset_index(drop=True)
            )

    def test_large_panel_warning(self):
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.n_params = 51
        with pytest.warns(UserWarning, match="consider HyperTreeNetVAR"):
            model._post_datasets_setup(seed=0)

    def test_forecast_params_handles_flat_output(self):
        # Defensive guard mirroring HyperTreeAR: Booster.predict normally
        # returns (n_rows, num_class) for multi-class output.
        model = HyperTreeVAR(p=1, fcst_h=1)
        model.n_params = 2

        class FlatBooster:
            def predict(self, X):
                return np.zeros(len(X) * 2)

        model.model = FlatBooster()
        out = model._forecast_params(pd.DataFrame({"a": [1.0, 2.0]}))
        assert out.shape == (2, 2)

    def test_forecast_validation_errors(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.forecast(test_data=test)

        model.train(
            lgb_params=LGB_PARAMS, num_iterations=5, train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2),
        )
        with pytest.raises(ValueError, match="Required column 'date'"):
            model.forecast(test_data=test.drop(columns="date"))
        with pytest.raises(ValueError, match="'type' must be one of"):
            model.forecast(test_data=test, type="components")
        with pytest.raises(ValueError, match="exactly fcst_h"):
            model.forecast(test_data=test.iloc[:-1])
        with pytest.raises(ValueError, match="Missing series in training"):
            renamed = test.copy()
            renamed.loc[renamed["series_id"] == "s2", "series_id"] = "s9"
            model.forecast(test_data=renamed)
        with pytest.raises(ValueError, match="Missing series in training"):
            extra_series = test[test["series_id"] == "s0"].assign(series_id="s9")
            model.forecast(test_data=pd.concat([test, extra_series], ignore_index=True))
        with pytest.raises(ValueError, match="identical dates"):
            shifted = test.copy()
            shifted.loc[shifted["series_id"] == "s2", "date"] = (
                shifted.loc[shifted["series_id"] == "s2", "date"] + pd.DateOffset(days=1)
            )
            model.forecast(test_data=shifted)
        with pytest.raises(ValueError, match="only supported with type='forecast'"):
            model.forecast(test_data=test, type="parameters", level=[80])
        with pytest.raises(ValueError, match="non-empty list"):
            model.forecast(test_data=test, level=[])
        with pytest.raises(ValueError, match=r"integers in \(0, 100\)"):
            model.forecast(test_data=test, level=[150])
        with pytest.raises(ValueError, match="Missing features"):
            model.forecast(test_data=test.drop(columns="month"))

        # Internal errors are wrapped, mirroring the univariate models.
        model._fcst_state = np.zeros(3)
        with pytest.raises(RuntimeError, match="Forecasting not successful"):
            model.forecast(test_data=test)


class TestScaling:
    """Per-series scaling: statistics, equivalence, de-normalization, guards."""

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError, match="scaling must be one of"):
            HyperTreeVAR(p=P, scaling="minmax")
        with pytest.raises(ValueError, match="scaling must be one of"):
            HyperTreeNetVAR(p=P, scaling="minmax")

    def test_scaling_none_identity_stats(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling=None)
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        np.testing.assert_array_equal(model._scale_loc, np.zeros(K))
        np.testing.assert_array_equal(model._scale_scale, np.ones(K))

    def test_mean_scaling_stats(self):
        train, _ = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)  # default scaling="mean"
        model.train(lgb_params=LGB_PARAMS, num_iterations=5, train_data=train)
        Y = train["value"].to_numpy().reshape(K, T).T
        np.testing.assert_array_equal(model._scale_loc, np.zeros(K))
        np.testing.assert_allclose(model._scale_scale, np.abs(Y).mean(axis=0))

    def test_standard_scaling_stats_and_forecast(self):
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling="standard")
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        Y = train["value"].to_numpy().reshape(K, T).T
        np.testing.assert_allclose(model._scale_loc, Y.mean(axis=0))
        np.testing.assert_allclose(model._scale_scale, Y.std(axis=0))
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()

    def test_mean_scaling_matches_manual_prescaling(self):
        """Internal scaling="mean" must equal manual pre-scaling + manual
        re-normalization of the forecasts (mean-abs scaling is location-free,
        so training on the scaled panel is an exact reparametrization)."""
        train, test = make_panel()

        model_internal = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling="mean")
        model_internal.train(lgb_params=LGB_PARAMS, num_iterations=15, train_data=train)
        fcst_internal = model_internal.forecast(test_data=test)["fcst"].to_numpy()

        # Manual pre-scaling with identically computed statistics
        Y = train["value"].to_numpy().reshape(K, T).T
        scale = np.abs(Y).mean(axis=0)
        scaled_train = train.copy()
        scaled_train["value"] = (Y / scale).T.ravel()
        model_manual = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling=None)
        model_manual.train(lgb_params=LGB_PARAMS, num_iterations=15, train_data=scaled_train)
        fcst_manual = model_manual.forecast(test_data=test)["fcst"].to_numpy()
        fcst_manual = fcst_manual * np.repeat(scale, FCST_H)

        np.testing.assert_allclose(fcst_internal, fcst_manual, rtol=1e-6)

    def test_forecast_invariant_to_test_row_order(self):
        """The lag state and scaling stats are positional in training series
        order; a permuted test panel must yield identical per-series results."""
        train, test = make_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)
        model.train(lgb_params=LGB_PARAMS, num_iterations=10, train_data=train)
        fcst = model.forecast(test_data=test)
        reordered = pd.concat(
            [test[test["series_id"] == sid] for sid in ["s2", "s0", "s1"]],
            ignore_index=True,
        )
        fcst_reordered = model.forecast(test_data=reordered)
        merged = fcst.merge(
            fcst_reordered, on=["series_id", "date"], suffixes=("", "_r")
        )
        np.testing.assert_allclose(merged["fcst"], merged["fcst_r"], rtol=1e-12)

    def test_constant_series_scale_guard(self):
        model = HyperTreeVAR(p=P, fcst_h=FCST_H)  # scaling="mean"
        scaled = model._fit_scaling(np.zeros((10, 2)))
        np.testing.assert_array_equal(model._scale_scale, np.ones(2))
        np.testing.assert_array_equal(scaled, np.zeros((10, 2)))

    def test_netvar_scaling_passthrough(self):
        train, test = make_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H, scaling="standard")
        model.train(
            lgb_params=LGB_PARAMS, network_params=NET_PARAMS,
            num_iterations=10, train_data=train,
        )
        assert model.scaling == "standard"
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()


# ----------------------------------------------------------------------
# Restricted (GVAR-style) factor design: HyperTreeVAR(type="factor") /
# HyperTreeNetVAR(type="factor")
# ----------------------------------------------------------------------
K_F, T_F = 6, 80
FACTOR_LGB_PARAMS = {"learning_rate": 0.05, "min_data_in_leaf": 10}


def make_factor_panel(k=K_F, n_train=T_F, fcst_h=FCST_H, seed=0):
    """Aligned panel where every series follows its own past and the common
    factor (cross-sectional mean), i.e. exactly the factor-design structure."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2015-01-01", periods=n_train + fcst_h, freq="MS")
    Y = np.full((len(dates), k), 10.0)
    for t in range(1, len(dates)):
        factor = Y[t - 1].mean()
        Y[t] = 4.0 + 0.4 * Y[t - 1] + 0.2 * factor + 0.5 * rng.randn(k)

    frames = []
    for i in range(k):
        frames.append(pd.DataFrame({
            "series_id": f"s{i}",
            "date": dates,
            "value": Y[:, i],
            "month": dates.month,
            "series_num": i,
        }))
    df = pd.concat(frames, ignore_index=True)
    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return train, test


class TestFactorAnalyticDerivatives:
    @pytest.mark.parametrize("hessian_method", ["analytic", "exact"])
    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.HuberLoss()])
    def test_gradients_and_hessians_match_autograd(self, loss_fn, hessian_method):
        torch.manual_seed(0)
        k, p, t_r = 3, 2, 11
        n_params = 2 * p
        model = HyperTreeVAR(
            p=p, fcst_h=1, loss_fn=loss_fn, type="factor", hessian_method=hessian_method
        )
        model.k = k
        model.n_params = n_params
        Z = torch.randn(k * t_r, n_params)
        model._Z_train = Z

        rng = np.random.RandomState(0)
        label = rng.randn(k * t_r)
        predt = rng.randn(k * t_r * n_params)

        grad, hess = model.objective_fn(predt, FakeDataset(label))

        # Reference: per-parameter second-order autograd through the fit.
        params = torch.tensor(
            predt.reshape(-1, n_params, order="F"), dtype=torch.float32, requires_grad=True
        )
        fit = (params * Z).sum(dim=1).reshape(k, t_r)
        loss = loss_fn(fit, torch.tensor(label.reshape(k, t_r), dtype=torch.float32))
        g_ref = torch.autograd.grad(loss, params, create_graph=True)[0]
        h_ref = torch.stack(
            [
                torch.autograd.grad(g_ref[:, i].sum(), params, retain_graph=True)[0][:, i]
                for i in range(n_params)
            ],
            dim=1,
        )

        np.testing.assert_allclose(
            grad, g_ref.detach().numpy().ravel(order="F"), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            hess, h_ref.detach().numpy().ravel(order="F"), rtol=1e-4, atol=1e-5
        )


class TestFactorDesign:
    def test_train_forecast_shapes(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, freq="M", fcst_h=FCST_H, type="factor")
        result = model.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=15, train_data=train)
        assert result.training_time is not None
        assert model.n_params == 2 * P  # independent of k

        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K_F * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-Tree-FactorVAR({P})").all()

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="type must be either 'full' or 'factor'"):
            HyperTreeVAR(p=P, type="sparse")
        with pytest.raises(ValueError, match="type must be either 'full' or 'factor'"):
            HyperTreeNetVAR(p=P, type="sparse")

    def test_parameters_output(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor")
        model.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=10, train_data=train)
        params = model.forecast(test_data=test, type="parameters")
        coef_cols = [f"A{j}(own)" for j in range(1, P + 1)] + [f"A{j}(factor)" for j in range(1, P + 1)]
        assert list(params.columns) == ["series_id", "date", "model"] + coef_cols
        assert len(params) == K_F * FCST_H

    def test_recovers_restricted_ols(self):
        """With only a series-identity feature the parameters cannot vary over
        time, so boosting must converge to the per-equation OLS solution on
        the restricted design [own lags, factor lags]. A white-noise panel
        keeps the regressors near-orthogonal, so the diagonal-Newton boosting
        reaches the OLS optimum exactly (estimator equivalence is a property
        of the algorithm, not of the data's dynamics)."""
        rng = np.random.RandomState(7)
        k, p, n_train, h = 6, 2, 400, 6
        Y = rng.randn(n_train + h, k)
        Y_train = Y[:n_train]
        dates = pd.date_range("1990-01-01", periods=n_train + h, freq="MS")

        df = pd.concat(
            [pd.DataFrame({"series_id": f"s{i}", "date": dates[:n_train],
                           "value": Y_train[:, i], "series_num": i})
             for i in range(k)],
            ignore_index=True,
        )
        test = pd.concat(
            [pd.DataFrame({"series_id": f"s{i}", "date": dates[n_train:], "series_num": i})
             for i in range(k)],
            ignore_index=True,
        )

        # scaling=None: the reference OLS coefficients live in raw space.
        model = HyperTreeVAR(p=p, fcst_h=h, scaling=None, type="factor")
        model.train(
            lgb_params={"learning_rate": 0.1, "min_data_in_leaf": 5},
            num_iterations=3000,
            train_data=df,
        )

        # Per-equation OLS on the same restricted design.
        factor = Y_train.mean(axis=1)
        factor_lags = np.column_stack([factor[p - j: n_train - j] for j in range(1, p + 1)])
        params = model.forecast(test_data=test, type="parameters")
        coef_cols = [f"A{j}(own)" for j in range(1, p + 1)] + [f"A{j}(factor)" for j in range(1, p + 1)]
        for i in range(k):
            own_lags = np.column_stack([Y_train[p - j: n_train - j, i] for j in range(1, p + 1)])
            design = np.concatenate([own_lags, factor_lags], axis=1)
            ols = np.linalg.lstsq(design, Y_train[p:, i], rcond=None)[0]
            ours = params[params["series_id"] == f"s{i}"][coef_cols].iloc[0].to_numpy(dtype=float)
            np.testing.assert_allclose(ours, ols, atol=1e-3)

    def test_forecast_recursion_matches_manual(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling=None, type="factor")
        model.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=10, train_data=train)

        fcst = model.forecast(test_data=test)
        params = model.forecast(test_data=test, type="parameters")
        coef_cols = [f"A{j}(own)" for j in range(1, P + 1)] + [f"A{j}(factor)" for j in range(1, P + 1)]
        Pmat = params[coef_cols].to_numpy().reshape(K_F, FCST_H, 2 * P)

        Y_train = train["value"].to_numpy().reshape(K_F, T_F).T  # (T, k)
        factor = Y_train.mean(axis=1)
        own = np.stack([Y_train[-j] for j in range(1, P + 1)], axis=1)  # (k, p)
        f_state = np.array([factor[-j] for j in range(1, P + 1)])
        manual = np.empty((K_F, FCST_H))
        for h in range(FCST_H):
            y_next = (Pmat[:, h, :P] * own).sum(axis=1) + Pmat[:, h, P:] @ f_state
            manual[:, h] = y_next
            own = np.concatenate([y_next[:, None], own[:, :-1]], axis=1)
            f_state = np.concatenate([[y_next.mean()], f_state[:-1]])

        np.testing.assert_allclose(fcst["fcst"].to_numpy(), manual.reshape(-1), rtol=1e-6)

    def test_forecast_invariant_to_test_row_order(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor")
        model.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=10, train_data=train)
        fcst = model.forecast(test_data=test)
        order = [f"s{i}" for i in (3, 0, 5, 1, 4, 2)]
        reordered = pd.concat(
            [test[test["series_id"] == sid] for sid in order], ignore_index=True
        )
        fcst_reordered = model.forecast(test_data=reordered)
        merged = fcst.merge(fcst_reordered, on=["series_id", "date"], suffixes=("", "_r"))
        np.testing.assert_allclose(merged["fcst"], merged["fcst_r"], rtol=1e-12)

    def test_scaling_matches_manual_prescaling(self):
        """Internal mean scaling must equal manual pre-scaling plus manual
        re-normalization (the factor is computed on the scaled panel in both
        cases)."""
        train, test = make_factor_panel()
        model_internal = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling="mean", type="factor")
        model_internal.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=15, train_data=train)
        fcst_internal = model_internal.forecast(test_data=test)["fcst"].to_numpy()

        Y = train["value"].to_numpy().reshape(K_F, T_F).T
        scale = np.abs(Y).mean(axis=0)
        scaled_train = train.copy()
        scaled_train["value"] = (Y / scale).T.ravel()
        model_manual = HyperTreeVAR(p=P, fcst_h=FCST_H, scaling=None, type="factor")
        model_manual.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=15, train_data=scaled_train)
        fcst_manual = model_manual.forecast(test_data=test)["fcst"].to_numpy()
        fcst_manual = fcst_manual * np.repeat(scale, FCST_H)

        np.testing.assert_allclose(fcst_internal, fcst_manual, rtol=1e-5)

    def test_conformal_intervals(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor")
        model.train(
            lgb_params=FACTOR_LGB_PARAMS, num_iterations=10, train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo, hi = f"Hyper-Tree-FactorVAR({P})-lo-80", f"Hyper-Tree-FactorVAR({P})-hi-80"
        assert lo in fcst.columns and hi in fcst.columns
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()

    def test_validation_and_early_stopping(self):
        train, _ = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor")
        result = model.train(
            lgb_params=FACTOR_LGB_PARAMS, num_iterations=20, train_data=train,
            validation=True, early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_gn_hessian_trains(self):
        train, test = make_factor_panel()
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor", hessian_method="gn")
        model.train(lgb_params=FACTOR_LGB_PARAMS, num_iterations=5, train_data=train)
        assert np.isfinite(model.forecast(test_data=test)["fcst"]).all()

    def test_no_large_panel_warning_for_factor_design(self):
        """The k * p runtime warning applies to the full design only; the
        factor design's coefficient count is independent of k."""
        import warnings as _warnings
        model = HyperTreeVAR(p=P, fcst_h=FCST_H, type="factor")
        model.n_params = 51
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            model._post_datasets_setup(seed=0)


class TestNetVARFactorDesign:
    def test_train_forecast_shapes(self):
        train, test = make_factor_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H, type="factor")
        model.train(
            lgb_params=FACTOR_LGB_PARAMS, network_params=NET_PARAMS,
            num_iterations=15, train_data=train,
        )
        assert model.n_params == 2 * P  # MLP output dimension, independent of k

        fcst = model.forecast(test_data=test)
        assert len(fcst) == K_F * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-TreeNet-FactorVAR({P})").all()

    def test_parameters_and_conformal(self):
        train, test = make_factor_panel()
        model = HyperTreeNetVAR(p=P, fcst_h=FCST_H, type="factor")
        model.train(
            lgb_params=FACTOR_LGB_PARAMS, network_params=NET_PARAMS,
            num_iterations=10, train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        params = model.forecast(test_data=test, type="parameters")
        assert f"A{P}(factor)" in params.columns

        fcst = model.forecast(test_data=test, level=[80])
        lo = f"Hyper-TreeNet-FactorVAR({P})-lo-80"
        hi = f"Hyper-TreeNet-FactorVAR({P})-hi-80"
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()
