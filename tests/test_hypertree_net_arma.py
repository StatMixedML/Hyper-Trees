"""Tests for the Hyper-TreeNet-ARMA model (GBDT encoder + MLP decoder)."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from hypertrees.conformal import ForecastIntervals
from hypertrees.models import HyperTreeNetARMA

K, T, FCST_H = 2, 80, 6
P, Q, S1P = 2, 1, 4
LGB_PARAMS = {"learning_rate": 0.1, "min_data_in_leaf": 5}
NETWORK_PARAMS = {
    "learning_rate": 1e-3,
    "embedding_dimension": 1,
    "hidden_dim": 16,
    "dropout": 0.1,
    "use_random_projection": True,
    "rp_embed_dim": P + Q,
}


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
    return HyperTreeNetARMA(**defaults)


def _trained(num_iterations=10, **train_kwargs):
    train, test = make_panel()
    model = make_model()
    model.train(
        lgb_params=LGB_PARAMS, network_params=NETWORK_PARAMS,
        num_iterations=num_iterations, train_data=train, **train_kwargs,
    )
    return model, train, test


class TestNetARMAInitialization:
    def test_default_initialization(self):
        model = HyperTreeNetARMA(p=3, q=2, freq="MS", fcst_h=FCST_H)
        assert model.n_params == 5
        assert model.stage1_p is None  # resolved at training time
        assert model.hessian_method == "exact"
        assert model.network is None
        assert model.is_trained is False

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="'p' must be a positive integer"):
            HyperTreeNetARMA(p=0, q=1)

    def test_invalid_q_points_to_net_ar(self):
        with pytest.raises(ValueError, match="HyperTreeNetAR"):
            HyperTreeNetARMA(p=2, q=0)

    def test_analytic_hessian_rejected(self):
        """The fit goes through the MLP, so it is not linear in the
        embeddings; only 'exact' and 'gn' are valid."""
        with pytest.raises(ValueError, match="either 'exact' or 'gn'"):
            HyperTreeNetARMA(p=2, q=1, hessian_method="analytic")

    def test_invalid_stage1_p(self):
        with pytest.raises(ValueError, match="stage1_p must be a positive integer"):
            HyperTreeNetARMA(p=2, q=1, stage1_p=0)

    def test_l1_loss_rejected(self):
        with pytest.raises(ValueError, match="L1Loss"):
            HyperTreeNetARMA(p=2, q=1, loss_fn=nn.L1Loss())

    def test_gn_non_mse_warns(self):
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeNetARMA(p=2, q=1, hessian_method="gn", loss_fn=nn.HuberLoss())


class TestNetARMATrainValidation:
    def test_network_params_required(self):
        train, _ = make_panel()
        model = make_model()
        with pytest.raises(ValueError, match="network_params must be provided"):
            model.train(lgb_params=LGB_PARAMS, num_iterations=3, train_data=train)

    def test_network_params_missing_keys(self):
        train, _ = make_panel()
        model = make_model()
        with pytest.raises(ValueError, match="missing required keys"):
            model.train(
                lgb_params=LGB_PARAMS, network_params={"learning_rate": 1e-3},
                num_iterations=3, train_data=train,
            )

    def test_too_short_series_raise(self):
        train, _ = make_panel()
        short = train.groupby("series_id", sort=False).head(8).reset_index(drop=True)
        model = make_model(stage1_p=10)
        with pytest.raises(ValueError, match="smaller stage1_p"):
            model.train(
                lgb_params=LGB_PARAMS, network_params=NETWORK_PARAMS,
                num_iterations=3, train_data=short,
            )

    def test_default_stage1_p_resolves_to_gomez_maravall(self):
        train, _ = make_panel()
        model = HyperTreeNetARMA(p=P, q=Q, freq="MS", fcst_h=FCST_H)  # stage1_p=None
        model.train(
            lgb_params=LGB_PARAMS, network_params=NETWORK_PARAMS,
            num_iterations=3, train_data=train,
        )
        expected = max(int(np.floor(np.log(T) ** 2)), 2 * max(P, Q))
        assert model.stage1_p == expected
        assert model._stage1.p == expected


class TestNetARMATraining:
    def test_train_forecast_shapes(self):
        model, _, test = _trained()
        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K * FCST_H
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == f"Hyper-TreeNet-ARMA({P},{Q})").all()

    def test_network_output_dim(self):
        model, _, _ = _trained(num_iterations=3)
        assert model.embedding_dim == NETWORK_PARAMS["embedding_dimension"]
        out_layer = [m for m in model.network.layers if isinstance(m, nn.Linear)][-1]
        assert out_layer.out_features == P + Q

    def test_forecast_matches_manual_recursion(self):
        """The forecast must equal the manual ARMA recursion driven by the
        MLP-decoded coefficients from type="parameters", the value seeds, and
        the residual seeds (with future innovations set to zero)."""
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
            np.testing.assert_allclose(got, manual, rtol=1e-3, atol=1e-4)

    def test_parameters_output(self):
        model, _, test = _trained()
        params = model.forecast(test_data=test, type="parameters")
        expected = (
            ["series_id", "date", "model"]
            + [f"AR({j})" for j in range(1, P + 1)]
            + [f"MA({i})" for i in range(1, Q + 1)]
        )
        assert list(params.columns) == expected
        for col in expected[3:]:
            assert np.isfinite(params[col]).all()

    def test_tree_embeddings_output(self):
        model, _, test = _trained()
        embeds = model.forecast(test_data=test, type="tree_embeddings")
        expected = ["series_id", "date", "model"] + [
            f"tree_embedding_{i + 1}" for i in range(NETWORK_PARAMS["embedding_dimension"])
        ]
        assert list(embeds.columns) == expected
        assert np.isfinite(embeds["tree_embedding_1"]).all()

    def test_gn_hessian_smoke(self):
        train, test = make_panel()
        model = make_model(hessian_method="gn")
        model.train(
            lgb_params=LGB_PARAMS, network_params=NETWORK_PARAMS,
            num_iterations=5, train_data=train,
        )
        fcst = model.forecast(test_data=test)
        assert np.isfinite(fcst["fcst"]).all()

    def test_validation_and_early_stopping(self):
        train, _ = make_panel()
        model = make_model()
        result = model.train(
            lgb_params=LGB_PARAMS, network_params=NETWORK_PARAMS,
            num_iterations=20, train_data=train,
            validation=True, early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_conformal_intervals(self):
        model, _, test = _trained(
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo = f"Hyper-TreeNet-ARMA({P},{Q})-lo-80"
        hi = f"Hyper-TreeNet-ARMA({P},{Q})-hi-80"
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


class TestNetARMAForecastValidation:
    def test_forecast_before_train(self):
        _, test = make_panel()
        model = make_model()
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.forecast(test_data=test)

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
        with pytest.raises(ValueError, match="'tree_embeddings'"):
            model.forecast(test_data=test, type="bogus")

    def test_level_without_calibration(self):
        model, _, test = _trained()
        with pytest.raises(RuntimeError, match="was not calibrated"):
            model.forecast(test_data=test, level=[80])
