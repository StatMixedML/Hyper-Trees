import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from unittest.mock import Mock, patch

from hypertrees.models.HyperTreeETS import HyperTreeETS
from hypertrees.utils import TrainingResult
from hypertrees import ForecastIntervals


class TestHyperTreeETSInitialization:
    """Test HyperTreeETS initialization and parameter validation."""

    def test_default_initialization_requires_seasonality_feature(self):
        """Default triple ETS requires a seasonality_feature."""
        with pytest.raises(ValueError, match="seasonality_feature must be provided"):
            HyperTreeETS()

    def test_triple_custom_initialization(self):
        """Test triple ETS initialization with explicit seasonality settings."""
        loss_fn = nn.MSELoss()
        model = HyperTreeETS(
            ets_type="triple",
            season_length=12,
            seasonality_feature="month",
            freq="M",
            fcst_h=6,
            loss_fn=loss_fn,
        )

        assert model.ets_type == "triple"
        assert model.season_length == 12
        assert model.seasonality_feature == "month"
        assert model.freq == "M"
        assert model.fcst_h == 6
        assert model.n_params == 4
        assert model.loss_fn is loss_fn
        assert model.loss_name == "MSELoss"

    def test_trend_initialization(self):
        """Trend ETS does not require seasonality_feature."""
        model = HyperTreeETS(ets_type="trend", season_length=4, fcst_h=3)
        assert model.ets_type == "trend"
        assert model.n_params == 2

    def test_invalid_ets_type(self):
        with pytest.raises(ValueError, match="ets_type must be one of 'triple', 'additive', or 'trend'"):
            HyperTreeETS(ets_type="invalid", season_length=12, seasonality_feature="month")

    def test_invalid_season_length(self):
        with pytest.raises(ValueError, match="season_length must be a positive integer"):
            HyperTreeETS(ets_type="triple", season_length=0, seasonality_feature="month")
        with pytest.raises(TypeError, match="season_length must be an integer"):
            HyperTreeETS(ets_type="triple", season_length=12.5, seasonality_feature="month")

    def test_invalid_fcst_h(self):
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeETS(ets_type="trend", season_length=4, fcst_h=0)

    def test_invalid_loss_function(self):
        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeETS(ets_type="trend", season_length=4, loss_fn="invalid")

    def test_invalid_freq_type(self):
        with pytest.raises(TypeError, match="freq must be a string"):
            HyperTreeETS(ets_type="trend", season_length=4, freq=123)


class TestHyperTreeETSSetForecastOrigin:
    """Test HyperTreeETS.set_forecast_origin delegates to _store_final_states."""

    def test_delegates_to_store_final_states(self):
        """set_forecast_origin should call _store_final_states."""
        model = HyperTreeETS(ets_type="trend", season_length=4, freq="Q")
        called_with = []
        model._store_final_states = lambda df: called_with.append(df)
        history = pd.DataFrame({
            "series_id": [0]*8,
            "date": pd.date_range("2020-01-01", periods=8, freq="QS"),
            "value": list(range(8)),
        })
        model.set_forecast_origin(history)
        assert len(called_with) == 1
        assert called_with[0] is history


class TestHyperTreeETSTraining:
    """Test HyperTreeETS training functionality."""

    @pytest.fixture
    def sample_train_data_triple(self):
        """Create sample training data for triple ETS (with month seasonality)."""
        np.random.seed(0)
        n_series = 3
        n_periods = 24
        dates = pd.date_range("2020-01-01", periods=n_periods, freq="ME")

        data = []
        for sid in range(n_series):
            df = pd.DataFrame(
                {
                    "series_id": sid,
                    "date": dates,
                    "value": (np.sin(np.arange(n_periods) / 6.0) * 10 + 100)
                    + np.random.randn(n_periods),
                    "feature1": np.random.randn(n_periods),
                }
            )
            df["month"] = df["date"].dt.month
            data.append(df)

        return pd.concat(data, ignore_index=True)

    def test_train_parameter_validation(self, sample_train_data_triple):
        model = HyperTreeETS(ets_type="triple", season_length=12, seasonality_feature="month")

        with pytest.raises(ValueError, match="train_data must be provided"):
            model.train(lgb_params={}, num_iterations=10)

        with pytest.raises(ValueError, match="lgb_params must be provided"):
            model.train(train_data=sample_train_data_triple, num_iterations=10)

        with pytest.raises(TypeError, match="train_data must be a pandas DataFrame"):
            model.train(lgb_params={}, train_data="invalid", num_iterations=10)

        with pytest.raises(TypeError, match="lgb_params must be a dictionary"):
            model.train(lgb_params="invalid", train_data=sample_train_data_triple, num_iterations=10)

        with pytest.raises(ValueError, match="num_iterations must be a positive integer"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=0)

        with pytest.raises(ValueError, match="early_stopping_round must be provided when validation is True"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, validation=True)

        with pytest.raises(ValueError, match="early_stopping_round can only be used when validation is True"):
            model.train(
                lgb_params={},
                train_data=sample_train_data_triple,
                num_iterations=10,
                validation=False,
                early_stopping_round=5,
            )

    def test_train_type_validation(self, sample_train_data_triple):
        """Test training parameter type validation for seed, verbose, validation, deterministic."""
        model = HyperTreeETS(ets_type="triple", season_length=12, seasonality_feature="month")

        with pytest.raises(TypeError, match="seed must be an integer"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, seed="bad")

        with pytest.raises(TypeError, match="verbose must be an integer"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, verbose="bad")

        with pytest.raises(TypeError, match="validation must be a boolean"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, validation="yes")

        with pytest.raises(TypeError, match="deterministic must be a boolean"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, deterministic="yes")

        with pytest.raises(ValueError, match="early_stopping_round must be a positive integer"):
            model.train(lgb_params={}, train_data=sample_train_data_triple, num_iterations=10, validation=True, early_stopping_round=-1)

    def test_train_requires_equal_series_length(self):
        """All series must have same length according to train() validation."""
        model = HyperTreeETS(ets_type="triple", season_length=12, seasonality_feature="month")
        # Build two series with different lengths
        s0 = pd.DataFrame(
            {
                "series_id": 0,
                "date": pd.date_range("2020-01-01", periods=10, freq="ME"),
                "value": np.random.randn(10),
                "feature1": np.random.randn(10),
                "month": pd.date_range("2020-01-01", periods=10, freq="ME").month,
            }
        )
        s1 = pd.DataFrame(
            {
                "series_id": 1,
                "date": pd.date_range("2020-01-01", periods=12, freq="ME"),
                "value": np.random.randn(12),
                "feature1": np.random.randn(12),
                "month": pd.date_range("2020-01-01", periods=12, freq="ME").month,
            }
        )
        df = pd.concat([s0, s1], ignore_index=True)

        with pytest.raises(ValueError, match="All series in train_data must have the same length"):
            model.train(lgb_params={}, train_data=df, num_iterations=10)

    @patch("hypertrees.models.HyperTreeETS.lgb.train")
    @patch("hypertrees.models.HyperTreeETS.prepare_datasets")
    @patch("hypertrees.models.HyperTreeETS.TimeSeriesPreprocessor")
    @patch.object(HyperTreeETS, "_store_final_states")
    def test_successful_training(
        self,
        mock_store_states,
        mock_preprocessor,
        mock_prepare_datasets,
        mock_lgb_train,
        sample_train_data_triple,
    ):
        model = HyperTreeETS(ets_type="triple", season_length=12, seasonality_feature="month")

        # Mock preprocessor to hand back features
        mock_pre = Mock()
        mock_preprocessor.return_value = mock_pre
        mock_pre.create_lags.return_value = sample_train_data_triple
        mock_pre.extract.return_value = {"features": pd.DataFrame({"feature1": [1, 2]})}

        # Mock dataset preparation
        mock_valid_sets = [Mock()]
        mock_prepare_datasets.return_value = (
            mock_valid_sets,
            ["train"],
            [],
            {"train": {"loss": [0.2]}},
            None,
            None,
            {},
        )

        # Mock LightGBM training
        mock_model = Mock()
        mock_model.best_iteration = 7
        mock_lgb_train.return_value = mock_model

        # Execute training
        result = model.train(
            lgb_params={"learning_rate": 0.1}, num_iterations=10, train_data=sample_train_data_triple
        )

        assert model.is_trained is True
        assert model.model is mock_model
        assert model.features == ["feature1"]
        assert isinstance(result, TrainingResult)
        assert result.train_metrics == {"loss": []}
        assert result.validation_metrics is None
        assert result.best_iteration == 7  # model.best_iteration when > 0
        assert result.training_time is not None
        mock_store_states.assert_called_once()


class TestHyperTreeETSForecasting:
    """Test HyperTreeETS forecasting functionality."""

    @pytest.fixture
    def trained_triple_model(self):
        model = HyperTreeETS(ets_type="triple", season_length=4, seasonality_feature="month", fcst_h=3)
        model.is_trained = True
        model.features = ["feature1"]
        model.n_series = 2
        model.series_order = [0, 1]

        # Mock LightGBM predict output: n_series * fcst_h * n_params
        n_params = 4
        mock_pred = np.random.randn(model.n_series * model.fcst_h * n_params)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = mock_pred
        model.model = mock_lgb_model

        # Provide stored final states for both series
        model.fcst_states = {
            0: {
                "last_level": torch.tensor(100.0),
                "last_trend": torch.tensor(1.5),
                "seasonality": torch.tensor([0.9, 1.0, 1.1, 1.0], dtype=torch.float32),
            },
            1: {
                "last_level": torch.tensor(200.0),
                "last_trend": torch.tensor(-0.5),
                "seasonality": torch.tensor([1.0, 0.95, 1.05, 1.0], dtype=torch.float32),
            },
        }
        return model

    @pytest.fixture
    def sample_test_data_triple(self):
        # Two series, 3 horizons each
        series_ids = [0, 0, 0, 1, 1, 1]
        dates = pd.date_range("2022-01-01", periods=6, freq="ME")
        df = pd.DataFrame(
            {
                "series_id": series_ids,
                "date": dates,
                "feature1": np.random.randn(6),
                # Seasons for each forecast step per series
                "month": [1, 2, 3, 1, 2, 3],
            }
        )
        return df

    def test_forecast_untrained_model(self):
        model = HyperTreeETS(ets_type="trend", season_length=4, fcst_h=2)
        with pytest.raises(RuntimeError, match="Model has not been trained"):
            model.forecast(test_data=pd.DataFrame())

    def test_forecast_parameter_validation(self, trained_triple_model):
        # Missing required columns: include both series IDs to bypass series-id check
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            trained_triple_model.forecast(
                test_data=pd.DataFrame({"series_id": [0, 1]})
            )

        # Missing features used during training (must have fcst_h rows per series)
        test_df = pd.DataFrame(
            {
                "series_id": [0]*3 + [1]*3,
                "date": pd.date_range("2022-01-01", periods=3, freq="MS").tolist() * 2,
                "month": [1, 2, 3, 1, 2, 3],
                # Missing required training feature 'feature1'
            }
        )
        with pytest.raises(RuntimeError, match="Missing features in test_data"):
            trained_triple_model.forecast(test_data=test_df)

        # Invalid type
        test_df = pd.DataFrame(
            {
                "series_id": [0]*3 + [1]*3,
                "date": pd.date_range("2022-01-01", periods=3, freq="MS").tolist() * 2,
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "month": [1, 2, 3, 1, 2, 3],
            }
        )
        with pytest.raises(ValueError, match="Parameter 'type' must be either 'forecast' or 'parameters'"):
            trained_triple_model.forecast(test_data=test_df, type="invalid")

    def test_forecast_series_id_mismatch(self, trained_triple_model, sample_test_data_triple):
        # Change a series id to create mismatch
        bad = sample_test_data_triple.copy()
        bad.loc[0, "series_id"] = 999
        with pytest.raises(ValueError, match="Missing series|Extra series"):
            trained_triple_model.forecast(test_data=bad)

    def test_forecast_success(self, trained_triple_model, sample_test_data_triple):
        result = trained_triple_model.forecast(test_data=sample_test_data_triple, type="forecast")
        assert isinstance(result, pd.DataFrame)
        assert set(["series_id", "date", "fcst", "model"]).issubset(result.columns)
        assert len(result) == len(sample_test_data_triple)
        assert result["model"].iloc[0] == "Hyper-Tree-ETS(triple)"

    def test_forecast_parameters_output(self, trained_triple_model, sample_test_data_triple):
        result = trained_triple_model.forecast(test_data=sample_test_data_triple, type="parameters")
        assert isinstance(result, pd.DataFrame)
        assert set(["series_id", "date", "model", "alpha", "beta", "gamma", "phi"]).issubset(
            result.columns
        )
        assert len(result) == len(sample_test_data_triple)

    @pytest.fixture
    def trained_trend_model(self):
        model = HyperTreeETS(ets_type="trend", season_length=4, fcst_h=3)
        model.is_trained = True
        model.features = ["feature1"]
        model.n_series = 2
        model.series_order = [0, 1]

        # Predict zeros so sigmoid -> 0.5 for both alpha and beta
        n_params = 2
        mock_pred = np.zeros(model.n_series * model.fcst_h * n_params, dtype=float)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = mock_pred
        model.model = mock_lgb_model

        # Provide stored final states required by forecast (no seasonality needed for trend)
        model.fcst_states = {
            0: {"last_level": torch.tensor(100.0), "last_trend": torch.tensor(2.0)},
            1: {"last_level": torch.tensor(200.0), "last_trend": torch.tensor(-1.0)},
        }
        return model

    @pytest.fixture
    def sample_test_data_trend(self):
        series_ids = [0, 0, 0, 1, 1, 1]
        dates = pd.date_range("2022-01-01", periods=6, freq="ME")
        return pd.DataFrame({
            "series_id": series_ids,
            "date": dates,
            "feature1": np.random.randn(6),
        })

    def test_forecast_parameters_trend(self, trained_trend_model, sample_test_data_trend):
        result = trained_trend_model.forecast(test_data=sample_test_data_trend, type="parameters")
        # Column presence for trend: only alpha and beta
        assert set(["series_id", "date", "model", "alpha", "beta"]).issubset(result.columns)
        assert "gamma" not in result.columns and "phi" not in result.columns
        assert len(result) == len(sample_test_data_trend)
        assert result["model"].iloc[0] == "Hyper-Tree-ETS(trend)"
        # With mocked zeros, sigmoid(0) -> 0.5
        assert np.allclose(result["alpha"].values, 0.5)
        assert np.allclose(result["beta"].values, 0.5)

    def test_forecast_trend_values(self, trained_trend_model, sample_test_data_trend):
        """Verify trend ETS forecast recursion produces correct numerical values."""
        result = trained_trend_model.forecast(test_data=sample_test_data_trend, type="forecast")
        fcsts = result["fcst"].values.reshape(2, 3)

        # With sigmoid(0)=0.5 for alpha and beta, the trend recursion simplifies:
        #   level_new = level + trend, trend_new = trend (constant)
        # Series 0: level=100, trend=2 -> [102, 104, 106]
        # Series 1: level=200, trend=-1 -> [199, 198, 197]
        np.testing.assert_allclose(fcsts[0], [102.0, 104.0, 106.0])
        np.testing.assert_allclose(fcsts[1], [199.0, 198.0, 197.0])

    def test_forecast_triple_values(self):
        """Verify triple ETS forecast recursion produces correct numerical values."""
        model = HyperTreeETS(ets_type="triple", season_length=4, seasonality_feature="month", fcst_h=3)
        model.is_trained = True
        model.features = ["feature1"]
        model.n_series = 2
        model.series_order = [0, 1]

        mock_pred = np.zeros(model.n_series * model.fcst_h * 4, dtype=float)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = mock_pred
        model.model = mock_lgb_model

        level_0, trend_0 = 100.0, 1.5
        level_1, trend_1 = 200.0, -0.5
        season_0 = [0.9, 1.0, 1.1, 1.0]
        season_1 = [1.0, 0.95, 1.05, 1.0]

        model.fcst_states = {
            0: {
                "last_level": torch.tensor(level_0),
                "last_trend": torch.tensor(trend_0),
                "seasonality": torch.tensor(season_0, dtype=torch.float32),
            },
            1: {
                "last_level": torch.tensor(level_1),
                "last_trend": torch.tensor(trend_1),
                "seasonality": torch.tensor(season_1, dtype=torch.float32),
            },
        }

        test_data = pd.DataFrame({
            "series_id": [0, 0, 0, 1, 1, 1],
            "date": pd.date_range("2022-01-01", periods=6, freq="ME"),
            "feature1": np.random.randn(6),
            "month": [1, 2, 3, 1, 2, 3],
        })

        result = model.forecast(test_data=test_data, type="forecast")
        fcsts = result["fcst"].values.reshape(2, 3)

        # Reference implementation: with all params=0.5 (sigmoid(0)),
        # compute the triple ETS recursion manually for each series.
        for s_idx, (level, trend, seasons) in enumerate([
            (level_0, trend_0, list(season_0)),
            (level_1, trend_1, list(season_1)),
        ]):
            alpha = beta = gamma = phi = 0.5
            s_idxs = [0, 1, 2]
            expected = []
            for h in range(3):
                s_h = seasons[s_idxs[h]]
                pseudo_y = (level + phi * trend) * s_h
                expected.append(pseudo_y)
                level_new = alpha * (pseudo_y / s_h) + (1 - alpha) * (level + phi * trend)
                trend_new = beta * (level_new - level) + (1 - beta) * phi * trend
                seasons[s_idxs[h]] = gamma * (pseudo_y / (level + phi * trend)) + (1 - gamma) * s_h
                level = level_new
                trend = trend_new
            np.testing.assert_allclose(fcsts[s_idx], expected, rtol=1e-5)


class TestHyperTreeETSCoreMethods:
    """Unit tests for internal objective and gradient utilities."""

    def test_get_params_loss_trend(self):
        """Smoke test get_params_loss for trend ETS with simple inputs."""
        model = HyperTreeETS(ets_type="trend", season_length=2, fcst_h=1)
        # Two series, 5 timesteps each (>= 2*season_length), 2 params
        model.n_series = 2
        T = 5
        n_params = model.n_params

        # predt for LightGBM: N*T*n_params values (Fortran order inside get_params_loss)
        predt = np.random.randn(model.n_series * T * n_params)
        target = torch.randn(model.n_series, T)

        # Mock lgb.Dataset with only a mask column needed by get_params_loss
        features_df = pd.DataFrame({"mask": np.ones(model.n_series * T, dtype=np.int32)})
        dset = Mock(spec=lgb.Dataset)
        dset.data = features_df

        params, loss = model.get_params_loss(predt, target, dset, requires_grad=False)
        assert isinstance(params, torch.nn.Parameter)
        assert params.shape == (model.n_series * T, n_params)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_get_params_loss_requires_grad(self):
        model = HyperTreeETS(ets_type="trend", season_length=2, fcst_h=1)
        model.n_series = 3
        T = 5
        n_params = model.n_params
        predt = np.random.randn(model.n_series * T * n_params)
        target = torch.randn(model.n_series, T)
        dset = Mock(spec=lgb.Dataset)
        dset.data = pd.DataFrame({"mask": np.ones(model.n_series * T, dtype=np.int32)})

        params, loss = model.get_params_loss(predt, target, dset, requires_grad=True)
        assert params.requires_grad is True
        assert loss.requires_grad is True

    def test_calculate_gradients_and_hessians(self):
        model = HyperTreeETS(ets_type="trend", season_length=4)
        model.n_params = 2  # Ensure alignment
        # Create differentiable params and a simple loss
        params = torch.randn(5, 2, requires_grad=True)
        # _fit must depend on params for Jacobian computation in GGN Hessian
        model._fit = params.sum(dim=1, keepdim=True)
        model._mask = torch.ones(5, 1)
        model._target = torch.randn(5, 1)
        loss = torch.sum(params ** 2)
        grad, hess = model.calculate_gradients_and_hessians(loss, params)
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (10,)
        assert hess.shape == (10,)

    def test_calculate_gradients_and_hessians_non_mse_loss(self):
        """Test the generic loss branch (non-MSELoss) that computes per-observation loss curvature."""
        model = HyperTreeETS(ets_type="trend", season_length=4, loss_fn=nn.SmoothL1Loss())
        model.n_params = 2
        params = torch.randn(5, 2, requires_grad=True)
        model._fit = params.sum(dim=1, keepdim=True)
        model._mask = torch.ones(5, 1)
        model._target = torch.randn(5, 1)
        loss = torch.sum(params ** 2)
        grad, hess = model.calculate_gradients_and_hessians(loss, params)
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (10,)
        assert hess.shape == (10,)

    def test_calculate_gradients_and_hessians_fortran_ordering(self):
        """Verify outputs use Fortran (column-major) ordering as required by LightGBM."""
        model = HyperTreeETS(ets_type="trend", season_length=4)
        model.n_params = 2
        n_obs = 4
        params = torch.randn(n_obs, 2, requires_grad=True)
        model._fit = params.sum(dim=1, keepdim=True)
        model._mask = torch.ones(n_obs, 1)
        model._target = torch.randn(n_obs, 1)
        loss = torch.sum(params ** 2)
        grad, hess = model.calculate_gradients_and_hessians(loss, params)

        # With loss = sum(params^2), gradient w.r.t. params[i,j] = 2*params[i,j]
        expected_grad = (2 * params).detach().numpy().ravel(order="F")
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_calculate_gradients_and_hessians_hessian_nonnegative(self):
        """GGN approximation guarantees positive semi-definite Hessians."""
        model = HyperTreeETS(ets_type="trend", season_length=4, n_hessian_probes=50)
        model.n_params = 2
        params = torch.randn(8, 2, requires_grad=True)
        model._fit = params.sum(dim=1, keepdim=True)
        model._mask = torch.ones(8, 1)
        model._target = torch.randn(8, 1)
        loss = model.loss_fn(model._fit, model._target)
        _, hess = model.calculate_gradients_and_hessians(loss, params)
        assert np.all(hess >= 0), f"GGN Hessian has negative entries: {hess[hess < 0]}"

    def test_calculate_gradients_and_hessians_cleanup(self):
        """Verify _fit, _mask, _target are released after computation."""
        model = HyperTreeETS(ets_type="trend", season_length=4)
        model.n_params = 2
        params = torch.randn(5, 2, requires_grad=True)
        model._fit = params.sum(dim=1, keepdim=True)
        model._mask = torch.ones(5, 1)
        model._target = torch.randn(5, 1)
        loss = torch.sum(params ** 2)
        model.calculate_gradients_and_hessians(loss, params)
        assert model._fit is None
        assert model._mask is None
        assert model._target is None

    def test_hutchinson_hessian_mse_shape_and_nonneg(self):
        """Test GaussNewtonHessian MSE path returns correct shape and non-negative values."""
        model = HyperTreeETS(ets_type="trend", season_length=4, n_hessian_probes=20)
        params = torch.randn(6, 2, requires_grad=True)
        fit_masked = params.sum(dim=1, keepdim=True)
        target = torch.randn(6, 1)
        rng = torch.Generator().manual_seed(42)
        hess = model._gn_hessian.estimate(fit_masked, target, params, rng)
        assert hess.shape == params.shape
        assert torch.all(hess >= 0)

    def test_hutchinson_hessian_mse_scaling(self):
        """Verify MSE Hessian scales with 2/N as expected."""
        model = HyperTreeETS(ets_type="trend", season_length=4, n_hessian_probes=200)
        n_obs = 8
        params = torch.randn(n_obs, 2, requires_grad=True)
        # Identity Jacobian: fit = params[:, 0:1]
        fit_masked = params[:, 0:1].clone()
        fit_masked.retain_grad()
        target = torch.randn(n_obs, 1)
        rng = torch.Generator().manual_seed(0)
        hess = model._gn_hessian.estimate(fit_masked, target, params, rng)
        # For J = [I, 0], H_GN = (2/N) * diag(J^T J) -> col 0 ~ 2/N, col 1 ~ 0
        expected_col0 = 2.0 / n_obs
        assert torch.allclose(hess[:, 0].mean(), torch.tensor(expected_col0), atol=0.05)

    def test_hutchinson_hessian_general_shape_and_nonneg(self):
        """Test GaussNewtonHessian general path returns correct shape and non-negative values."""
        model = HyperTreeETS(ets_type="trend", season_length=4,
                             loss_fn=nn.SmoothL1Loss(), n_hessian_probes=20)
        params = torch.randn(6, 2, requires_grad=True)
        fit_masked = params.sum(dim=1, keepdim=True)
        target = torch.randn(6, 1)
        rng = torch.Generator().manual_seed(42)
        hess = model._gn_hessian.estimate(fit_masked, target, params, rng)
        assert hess.shape == params.shape
        assert torch.all(hess >= 0)

    def test_hutchinson_hessian_dispatch(self):
        """Verify __init__ binds the correct Hessian estimate method based on loss type."""
        from hypertrees.utils import GaussNewtonHessian
        model_mse = HyperTreeETS(ets_type="trend", season_length=4)
        assert model_mse._gn_hessian.estimate == model_mse._gn_hessian._mse

        model_generic = HyperTreeETS(ets_type="trend", season_length=4, loss_fn=nn.SmoothL1Loss())
        assert model_generic._gn_hessian.estimate == model_generic._gn_hessian._general

    @patch.object(HyperTreeETS, "get_params_loss")
    def test_eval_fn_returns_metric(self, mock_get_params_loss):
        model = HyperTreeETS(ets_type="trend", season_length=4)
        model.n_series = 2
        # Mock loss returned by get_params_loss
        mock_get_params_loss.return_value = (torch.randn(6, 2), torch.tensor(1.23))

        # Mock lgb.Dataset with label and minimal data
        mock_eval = Mock(spec=lgb.Dataset)
        mock_eval.get_label.return_value = np.random.randn(model.n_series * 3)  # 3 steps
        mock_eval.data = pd.DataFrame({"mask": np.ones(model.n_series * 3, dtype=np.int32)})

        predt = np.random.randn(model.n_series * 3 * model.n_params)
        name, value, is_higher_better = model.eval_fn(predt, mock_eval)
        assert isinstance(name, str)
        assert isinstance(value, float)
        assert is_higher_better is False


class TestHyperTreeETSObjectiveFunction:
    """Tests for objective_fn behavior and data handling."""

    def test_objective_fn_data_preparation_and_tensor_conversion(self):
        model = HyperTreeETS(ets_type="trend", season_length=2, fcst_h=1)
        model.n_series = 3
        T = 4
        n_params = model.n_params

        # Prepare predt for samples * n_params
        n_samples = model.n_series * T
        predt = np.random.randn(n_samples * n_params)

        # Mock LightGBM dataset
        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(n_samples)

        with patch.object(model, "get_params_loss") as mock_get_params_loss, \
             patch.object(model, "calculate_gradients_and_hessians") as mock_calc_grad_hess:
            # Mock returns
            mock_params = torch.randn(n_samples, n_params, requires_grad=True)
            mock_loss = torch.tensor(0.5)
            mock_get_params_loss.return_value = (mock_params, mock_loss)
            exp_grad = np.random.randn(n_samples * n_params)
            exp_hess = np.random.randn(n_samples * n_params)
            mock_calc_grad_hess.return_value = (exp_grad, exp_hess)

            grad, hess = model.objective_fn(predt, mock_dataset)

            # Dataset label used
            mock_dataset.get_label.assert_called_once()

            # get_params_loss called with requires_grad=True
            mock_get_params_loss.assert_called_once()
            args, kwargs = mock_get_params_loss.call_args
            # predt passed through
            np.testing.assert_array_equal(args[0], predt)
            # target tensor shaped (n_series, T)
            target_tensor = args[1]
            assert isinstance(target_tensor, torch.Tensor)
            assert target_tensor.shape == (model.n_series, T)
            # dataset object passed
            assert args[2] is mock_dataset
            # requires_grad flag
            assert kwargs.get("requires_grad") is True

            # calculate_gradients_and_hessians called with outputs of get_params_loss
            mock_calc_grad_hess.assert_called_once_with(mock_loss, mock_params)

            # Return values propagate
            np.testing.assert_array_equal(grad, exp_grad)
            np.testing.assert_array_equal(hess, exp_hess)


class TestHyperTreeETSTripleForward:
    """Tests for the internal _forward_triple method."""

    def test_forward_triple_shapes_and_values(self):
        # Setup model
        model = HyperTreeETS(ets_type="triple", season_length=4, seasonality_feature="month")
        model.n_series = 2
        T = 8  # multiple of season_length

        # Params: alpha, beta, gamma in (0,1), phi in (0,1)
        alpha = 0.5
        beta = 0.3
        gamma = 0.4
        phi = 0.8
        params = torch.tensor(
            np.tile([[alpha, beta, gamma, phi]], (model.n_series * T, 1)), dtype=model.dtype
        ).reshape(model.n_series, T, 4)

        # Build target with multiplicative seasonality and simple trend
        base = np.arange(T, dtype=float)
        seasonal = np.array([1.0, 0.9, 1.1, 1.0])
        y0 = (100 + 2 * base) * np.tile(seasonal, T // 4)
        y1 = (120 + 1.0 * base) * np.tile(seasonal, T // 4)
        target = torch.tensor(np.vstack([y0, y1]), dtype=model.dtype)

        # Mask (all valid)
        mask = torch.ones((model.n_series, T), dtype=model.dtype)

        # Build dataset with seasonality_feature column values 1..season_length per timestep per series
        months = np.tile(np.arange(1, model.season_length + 1), T // model.season_length)
        feat_df = pd.DataFrame({
            "month": np.concatenate([months, months])  # series 0 timesteps then series 1
        })
        dset = Mock(spec=lgb.Dataset)
        dset.data = feat_df

        level, trend, seasonality, fit = model._forward_triple(params, dset, target, mask)

        # Validate shapes: level/trend are final-step scalars, fit is a list of T tensors
        assert isinstance(level, torch.Tensor)
        assert level.shape == (model.n_series,)
        assert isinstance(trend, torch.Tensor)
        assert trend.shape == (model.n_series,)
        assert isinstance(fit, list) and len(fit) == T
        for t in range(T):
            assert fit[t].shape == (model.n_series,)

        assert isinstance(seasonality, torch.Tensor)
        assert seasonality.shape == (model.n_series, model.season_length)

        # Basic value checks
        # First fitted equals first target by initialization
        assert torch.allclose(fit[0], target[:, 0])
        # No NaNs in outputs
        assert torch.isfinite(level).all()
        assert torch.isfinite(trend).all()
        assert torch.isfinite(torch.stack(fit, dim=1)).all()
        assert torch.isfinite(seasonality).all()
        # Seasonality positive for this synthetic data
        assert (seasonality > 0).all()


class TestHyperTreeETSStoreFinalStates:
    """Tests for _store_final_states method."""

    def test_store_final_states_triple(self):
        # Create training data: 2 series x 8 periods, with month and a simple feature
        n_series = 2
        T = 8
        dates = pd.date_range("2021-01-01", periods=T, freq="ME")
        dfs = []
        for sid in range(n_series):
            df = pd.DataFrame(
                {
                    "series_id": sid,
                    "date": dates,
                    "value": 100 + sid * 10 + np.arange(T) * 0.5,
                    "feature1": np.random.randn(T),
                }
            )
            # Cycle 1..season_length to align with model.season_length indexing
            season_length = 4
            df["month"] = np.tile(np.arange(1, season_length + 1), T // season_length)
            dfs.append(df)
        train_data = pd.concat(dfs, ignore_index=True)

        # Initialize model
        model = HyperTreeETS(ets_type="triple", season_length=4, seasonality_feature="month")
        model.n_series = n_series
        model.features = ["feature1", "month"]  # ensure month is in features for forward seasonal indices

        # Mock LightGBM model predict to return zeros (sigmoid -> 0.5 params)
        mock_model = Mock()
        n_params = model.n_params  # 4
        N = len(train_data)  # n_series * T
        mock_model.predict.return_value = np.zeros(N * n_params, dtype=float)
        model.model = mock_model

        # Call the private method
        model._store_final_states(train_data)

        # Assert series_order is stored correctly
        assert model.series_order == train_data["series_id"].unique().tolist()

        # Assert fcst_states structure
        assert isinstance(model.fcst_states, dict)
        assert set(model.fcst_states.keys()) == set(model.series_order)

        for sid in model.series_order:
            st = model.fcst_states[sid]
            assert "last_level" in st and "last_trend" in st
            assert torch.is_tensor(st["last_level"]) and st["last_level"].ndim == 0
            assert torch.is_tensor(st["last_trend"]) and st["last_trend"].ndim == 0
            # Triple ETS should also store seasonality
            assert "seasonality" in st
            assert torch.is_tensor(st["seasonality"]) and st["seasonality"].shape == (
                model.season_length,
            )
            # No NaNs
            assert torch.isfinite(st["last_level"]).item()
            assert torch.isfinite(st["last_trend"]).item()
            assert torch.isfinite(st["seasonality"]).all()


class TestHyperTreeETSConformal:
    """Tests for conformal prediction intervals on HyperTreeETS."""

    FCST_H = 4
    N_SERIES = 2
    N_OBS = 60
    LGB_PARAMS = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 1, "min_data_in_bin": 1}

    def _make_data(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2015-01-01", periods=self.N_OBS, freq="MS")
        frames = []
        for sid in range(self.N_SERIES):
            frames.append(pd.DataFrame({
                "series_id": sid, "date": dates,
                "value": rng.standard_normal(self.N_OBS).cumsum() + 100,
                "month": dates.month, "quarter": dates.quarter,
            }))
        return pd.concat(frames, ignore_index=True)

    def _split(self, df):
        test = df.groupby("series_id", sort=False).tail(self.FCST_H).reset_index(drop=True)
        train = df.drop(df.groupby("series_id", sort=False).tail(self.FCST_H).index).reset_index(drop=True)
        return train, test

    @pytest.fixture
    def split(self):
        return self._split(self._make_data())

    @pytest.fixture
    def calibrated(self, split):
        train, test = split
        model = HyperTreeETS(ets_type="trend", season_length=12, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
                     forecast_intervals=ForecastIntervals(n_windows=3, refit=False))
        return model, train, test

    def test_calibration_sets_state(self, calibrated):
        model, _, _ = calibrated
        assert model._is_calibrated is True
        assert model._cs_scores.shape == (3, self.N_SERIES, self.FCST_H)
        assert np.all(model._cs_scores >= 0)

    def test_no_calibration_by_default(self, split):
        train, _ = split
        model = HyperTreeETS(ets_type="trend", season_length=12, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train)
        assert model._is_calibrated is False

    def test_forecast_adds_interval_columns(self, calibrated):
        model, _, test = calibrated
        mn = "Hyper-Tree-ETS(trend)"
        out = model.forecast(test_data=test, level=[80, 90])
        for lv in [80, 90]:
            assert f"{mn}-lo-{lv}" in out.columns
            assert f"{mn}-hi-{lv}" in out.columns

    def test_interval_nesting(self, calibrated):
        model, _, test = calibrated
        mn = "Hyper-Tree-ETS(trend)"
        out = model.forecast(test_data=test, level=[80, 90])
        assert np.all(out[f"{mn}-lo-90"].to_numpy() <= out[f"{mn}-lo-80"].to_numpy() + 1e-9)
        assert np.all(out[f"{mn}-hi-80"].to_numpy() <= out[f"{mn}-hi-90"].to_numpy() + 1e-9)

    def test_level_without_calibration_raises(self, split):
        train, test = split
        model = HyperTreeETS(ets_type="trend", season_length=12, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train)
        with pytest.raises(RuntimeError, match="not calibrated"):
            model.forecast(test_data=test, level=[90])


class TestInitTripleStates:
    """Tests for the decomposition-based state initialization (statsforecast-style)."""

    def test_l1_loss_rejected(self):
        """nn.L1Loss is rejected: zero curvature yields all-zero Hessians."""
        with pytest.raises(ValueError, match="L1Loss is not supported"):
            HyperTreeETS(
                ets_type="triple", season_length=4, seasonality_feature="month",
                loss_fn=nn.L1Loss(),
            )

    def test_seasonal_init_default_and_validation(self):
        """Default is 'classical'; invalid values raise."""
        model = HyperTreeETS(
            ets_type="triple", season_length=4, seasonality_feature="month"
        )
        assert model.seasonal_init == "classical"

        legacy = HyperTreeETS(
            ets_type="triple", season_length=4, seasonality_feature="month",
            seasonal_init="legacy",
        )
        assert legacy.seasonal_init == "legacy"

        with pytest.raises(ValueError, match="seasonal_init must be either"):
            HyperTreeETS(
                ets_type="triple", season_length=4, seasonality_feature="month",
                seasonal_init="bogus",
            )

    def test_legacy_init_reproduces_pre_020_formula(self):
        """seasonal_init='legacy' must match the pre-0.2.0 code verbatim,
        so earlier results (incl. the paper benchmarks) stay reproducible."""
        m, cycles = 4, 5
        T = m * cycles
        torch.manual_seed(3)
        y = 100.0 + torch.rand(1, T) * 50.0
        mask = torch.ones_like(y)
        slots = (torch.arange(T) % m).view(1, -1)

        model = HyperTreeETS(
            ets_type="triple", season_length=m, seasonality_feature="month",
            seasonal_init="legacy",
        )
        model.n_series = 1
        seasonality, level0, trend0 = model._init_triple_states(y, mask, slots)

        # Reference: the original (pre-0.2.0) implementation, replicated verbatim
        init_len = min(m, T)
        seasonal_avg = torch.zeros((1, init_len))
        for i in range(init_len):
            valid_obs = mask[:, i::init_len]
            seasonal_avg[:, i] = (y[:, i::init_len] * valid_obs.float()).sum(
                1) / valid_obs.float().sum(1).clamp(min=1)
        ref_seasonality = y[:, :init_len] / (seasonal_avg + model.eps)
        ref_season_adj = y[:, :init_len] / ref_seasonality[:, :init_len]

        torch.testing.assert_close(seasonality, ref_seasonality)
        torch.testing.assert_close(level0, ref_season_adj[:, 0])
        torch.testing.assert_close(trend0, ref_season_adj[:, 1] - ref_season_adj[:, 0])

    def test_recovers_seasonality_and_trend_with_offset_start(self):
        """The init must recover the seasonal shape under a linear trend,
        with correct slot alignment for a series starting mid-season.

        The old positional init both cancelled the seasonal shape and, even
        when shape-preserving, rotated the profile for series not starting
        at season position 1.
        """
        m, cycles = 4, 6
        T = m * cycles
        S = torch.tensor([0.8, 1.2, 1.1, 0.9])  # mean exactly 1.0
        start_offset = 1  # series starts at season position 2 (slot index 1)
        slots = (start_offset + torch.arange(T)) % m
        trend = 10.0 + 0.5 * torch.arange(T, dtype=torch.float32)
        y = (trend * S[slots]).view(1, -1)
        mask = torch.ones_like(y)

        model = HyperTreeETS(
            ets_type="triple", season_length=m, seasonality_feature="month"
        )
        model.n_series = 1

        seasonality, level0, trend0 = model._init_triple_states(
            y, mask, slots.view(1, -1)
        )

        # Seasonal shape recovered in SLOT order despite trend and offset start
        assert seasonality.shape == (1, m)
        torch.testing.assert_close(seasonality[0], S, atol=0.05, rtol=0.0)
        # OLS seeds: slope ~ 0.5, level ~ value of the trend line at t = 1
        assert abs(trend0.item() - 0.5) < 0.1
        assert abs(level0.item() - 10.5) < 0.5

    def test_short_series_falls_back_without_crash(self):
        """Series shorter than the MA kernel use the simple per-slot fallback."""
        m = 12
        model = HyperTreeETS(
            ets_type="triple", season_length=m, seasonality_feature="month"
        )
        model.n_series = 1
        for T in [1, 3, 6]:
            y = 100.0 + torch.arange(T, dtype=torch.float32).view(1, -1)
            mask = torch.ones_like(y)
            slots = (torch.arange(T) % m).view(1, -1)
            seasonality, level0, trend0 = model._init_triple_states(y, mask, slots)
            assert seasonality.shape == (1, m)
            assert torch.isfinite(seasonality).all()
            assert (seasonality > 0).all()
            assert torch.isfinite(level0).all() and torch.isfinite(trend0).all()

    def test_init_cached_per_dataset(self):
        """The init must compute once per dataset, not once per boosting iteration.

        Cache hits must return the pristine initialization even though the
        recursion mutates the returned seasonality tensor in place.
        """
        m, cycles = 4, 6
        T = m * cycles
        y = (100.0 + torch.arange(T, dtype=torch.float32)).view(1, -1)
        mask = torch.ones_like(y)
        slots = (torch.arange(T) % m).view(1, -1)

        model = HyperTreeETS(
            ets_type="triple", season_length=m, seasonality_feature="month"
        )
        model.n_series = 1

        calls = {"n": 0}
        orig = model._init_triple_states

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        model._init_triple_states = counting

        data_a, data_b = object(), object()  # only identity is used as cache key

        s1, l1, t1 = model._cached_init_triple_states(data_a, y, mask, slots)
        s2, l2, t2 = model._cached_init_triple_states(data_a, y, mask, slots)
        assert calls["n"] == 1  # second call is a cache hit
        torch.testing.assert_close(s1, s2)
        torch.testing.assert_close(l1, l2)
        torch.testing.assert_close(t1, t2)

        # In-place mutation of a returned tensor must not poison the cache
        s1[0, 0] = 123.0
        s3, _, _ = model._cached_init_triple_states(data_a, y, mask, slots)
        assert calls["n"] == 1
        assert s3[0, 0].item() != 123.0

        # A different dataset object recomputes
        model._cached_init_triple_states(data_b, y, mask, slots)
        assert calls["n"] == 2

        # Entries pin their dataset object, so a key hit can only come from
        # that exact, still-alive dataset (ids of live objects are unique).
        assert model._init_cache[id(data_a)]["data"] is data_a

    def test_padded_rows_do_not_distort_indices(self):
        """Masked (padded) observations must not contribute to the init."""
        m, cycles = 4, 5
        T_real = m * cycles
        pad_len = 6
        S = torch.tensor([0.8, 1.2, 1.1, 0.9])
        slots_real = torch.arange(T_real) % m
        y_real = 50.0 * S[slots_real]

        # Back-append garbage values that the mask flags as padding
        y = torch.cat([y_real, torch.full((pad_len,), 9999.0)]).view(1, -1)
        mask = torch.cat([torch.ones(T_real), torch.zeros(pad_len)]).view(1, -1)
        slots = (torch.arange(T_real + pad_len) % m).view(1, -1)

        model = HyperTreeETS(
            ets_type="triple", season_length=m, seasonality_feature="month"
        )
        model.n_series = 1
        seasonality, level0, _ = model._init_triple_states(y, mask, slots)

        torch.testing.assert_close(seasonality[0], S, atol=0.05, rtol=0.0)
        assert abs(level0.item() - 50.0) < 2.5


# ----------------------------------------------------------------------
# Additive-seasonality variant (ets_type="additive")
# ----------------------------------------------------------------------
K_A, T_A, M_A, FCST_H_A = 2, 72, 12, 6
LGB_PARAMS_A = {"learning_rate": 0.1, "min_data_in_leaf": 5}


def make_additive_panel(k=K_A, n_train=T_A, fcst_h=FCST_H_A, seed=0, offset=0.0):
    """Aligned panel with linear trend plus additive monthly seasonality.

    With ``offset`` the series can be shifted to cross zero, which the
    multiplicative variant cannot handle.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2015-01-01", periods=n_train + fcst_h, freq="MS")
    t = np.arange(1, len(dates) + 1)
    season = 10.0 * np.sin(2 * np.pi * dates.month / M_A)

    frames = []
    for i in range(k):
        values = offset + 50.0 * (i + 1) + 0.5 * t + season + 0.5 * rng.randn(len(dates))
        frames.append(pd.DataFrame({
            "series_id": f"s{i}",
            "date": dates,
            "value": values,
            "month": dates.month,
            "series_num": i,
        }))
    df = pd.concat(frames, ignore_index=True)
    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return train, test


def make_additive_model(fcst_h=FCST_H_A):
    return HyperTreeETS(
        ets_type="additive", seasonality_feature="month",
        season_length=M_A, freq="M", fcst_h=fcst_h,
    )


class FakeDataset:
    """Mimics the lgb.Dataset attributes used by the ETS forward pass."""

    def __init__(self, df):
        self.data = df


class TestAdditiveInitialization:
    def test_constructor(self):
        model = make_additive_model()
        assert model.ets_type == "additive"
        assert model.n_params == 4

    def test_seasonality_feature_required(self):
        with pytest.raises(ValueError, match="seasonality_feature must be provided"):
            HyperTreeETS(ets_type="additive", season_length=12)

    def test_recovers_clean_additive_components(self):
        """On noise-free linear-trend + additive-seasonal data, the classical
        additive initialization recovers the seasonal profile, trend, and the
        level (line evaluated at the first observation) exactly: the centered
        2 x m moving average removes the seasonality and reproduces the linear
        trend, so the per-slot detrended differences equal the true indices."""
        model = make_additive_model()
        n, level0_true, trend_true = 6 * M_A, 100.0, 0.5
        dates = pd.date_range("2015-01-01", periods=n, freq="MS")
        season_profile = 10.0 * np.sin(2 * np.pi * np.arange(1, M_A + 1) / M_A)
        season_profile -= season_profile.mean()  # mean-zero truth
        t = np.arange(1, n + 1)
        y = level0_true + trend_true * (t - 1) + season_profile[dates.month - 1]

        target = torch.tensor(y.reshape(1, -1), dtype=torch.float32)
        mask = torch.ones_like(target)
        idxs = torch.tensor(dates.month.to_numpy() - 1).reshape(1, -1)
        seasonality, level0, trend0 = model._init_additive_states(target, mask, idxs)

        np.testing.assert_allclose(
            seasonality.numpy().ravel(), season_profile, atol=1e-3
        )
        np.testing.assert_allclose(trend0.item(), trend_true, atol=1e-3)
        np.testing.assert_allclose(level0.item(), level0_true, atol=1e-2)


class TestAdditiveForward:
    def test_forward_matches_manual_recursion(self):
        """The additive forward pass must reproduce a hand-rolled Holt-Winters
        additive (damped) recursion seeded with the same initial states."""
        torch.manual_seed(0)
        model = make_additive_model()
        model.n_series = 1
        n = 4 * M_A
        dates = pd.date_range("2015-01-01", periods=n, freq="MS")
        rng = np.random.RandomState(0)
        months = np.asarray(dates.month)
        y = 100 + 0.5 * np.arange(n) + 10 * np.sin(2 * np.pi * months / M_A) + rng.randn(n)

        target = torch.tensor(y.reshape(1, -1), dtype=torch.float32)
        mask = torch.ones_like(target)
        idxs = torch.tensor(months - 1).reshape(1, -1)

        alpha, beta, gamma, phi = 0.3, 0.2, 0.1, 0.95
        params = torch.tensor([alpha, beta, gamma, phi]).repeat(1, n, 1)

        data = FakeDataset(pd.DataFrame({"month": months}))
        level_T, trend_T, seas_T, fits = model.forward(params, data, target, mask)
        fits = torch.stack(fits, dim=1).numpy().ravel()

        # Manual reference recursion with identical initial states
        seas, l, b = model._init_additive_states(target, mask, idxs)
        seas = seas.numpy().ravel().copy()
        l, b = l.item(), b.item()
        ref = [y[0]]
        for t in range(1, n):
            s = seas[months[t] - 1]
            base = l + phi * b
            ref.append(base + s)
            l_new = alpha * (y[t] - s) + (1 - alpha) * base
            b = beta * (l_new - l) + (1 - beta) * phi * b
            seas[months[t] - 1] = gamma * (y[t] - base) + (1 - gamma) * s
            l = l_new

        np.testing.assert_allclose(fits, np.array(ref), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(level_T.item(), l, rtol=1e-4)
        np.testing.assert_allclose(trend_T.item(), b, rtol=1e-4)


class TestAdditiveTraining:
    def test_train_forecast_shapes(self):
        train, test = make_additive_panel()
        model = make_additive_model()
        result = model.train(lgb_params=LGB_PARAMS_A, num_iterations=20, train_data=train)
        assert result.training_time is not None

        fcst = model.forecast(test_data=test)
        assert list(fcst.columns) == ["series_id", "date", "fcst", "model"]
        assert len(fcst) == K_A * FCST_H_A
        assert np.isfinite(fcst["fcst"]).all()
        assert (fcst["model"] == "Hyper-Tree-ETS(additive)").all()

    def test_handles_negative_values(self):
        """The additive variant must handle series crossing zero, where the
        multiplicative variant is undefined."""
        train, test = make_additive_panel(offset=-120.0)
        assert (train["value"] < 0).any() and (train["value"] > 0).any()
        model = make_additive_model()
        model.train(lgb_params=LGB_PARAMS_A, num_iterations=20, train_data=train)
        fcst = model.forecast(test_data=test)
        assert np.isfinite(fcst["fcst"]).all()

    def test_forecast_accuracy_on_seasonal_panel(self):
        """On clean additive-seasonal data the forecasts should track the
        actuals closely."""
        train, test = make_additive_panel(seed=3)
        model = make_additive_model()
        model.train(lgb_params=LGB_PARAMS_A, num_iterations=50, train_data=train)
        fcst = model.forecast(test_data=test)
        merged = fcst.merge(test[["series_id", "date", "value"]], on=["series_id", "date"])
        smape = np.mean(
            200.0 * np.abs(merged["value"] - merged["fcst"])
            / (np.abs(merged["value"]) + np.abs(merged["fcst"]))
        )
        assert smape < 10.0

    def test_parameters_output(self):
        train, test = make_additive_panel()
        model = make_additive_model()
        model.train(lgb_params=LGB_PARAMS_A, num_iterations=10, train_data=train)
        params = model.forecast(test_data=test, type="parameters")
        assert list(params.columns) == ["series_id", "date", "model", "alpha", "beta", "gamma", "phi"]
        for col in ["alpha", "beta", "gamma", "phi"]:
            assert params[col].between(0, 1).all()

    def test_conformal_intervals(self):
        train, test = make_additive_panel()
        model = make_additive_model()
        model.train(
            lgb_params=LGB_PARAMS_A, num_iterations=10, train_data=train,
            forecast_intervals=ForecastIntervals(n_windows=2, refit=False),
        )
        fcst = model.forecast(test_data=test, level=[80])
        lo, hi = "Hyper-Tree-ETS(additive)-lo-80", "Hyper-Tree-ETS(additive)-hi-80"
        assert lo in fcst.columns and hi in fcst.columns
        assert (fcst[lo] <= fcst["fcst"]).all()
        assert (fcst["fcst"] <= fcst[hi]).all()

    def test_validation_and_early_stopping(self):
        train, _ = make_additive_panel()
        model = make_additive_model()
        result = model.train(
            lgb_params=LGB_PARAMS_A, num_iterations=20, train_data=train,
            validation=True, early_stopping_round=5,
        )
        assert "MSELoss" in result.validation_metrics
        assert len(result.validation_metrics["MSELoss"]) > 0

    def test_masked_padding_does_not_change_states(self):
        """Back-padded pseudo-observations (mask == 0) must leave the final
        states untouched."""
        model = make_additive_model()
        model.n_series = 1
        n, pad = 3 * M_A, 5
        dates = pd.date_range("2015-01-01", periods=n + pad, freq="MS")
        rng = np.random.RandomState(1)
        months = np.asarray(dates.month)
        y = 100 + 10 * np.sin(2 * np.pi * months / M_A) + rng.randn(n + pad)
        params = torch.tensor([0.3, 0.2, 0.1, 0.95]).repeat(1, n + pad, 1)

        target_full = torch.tensor(y.reshape(1, -1), dtype=torch.float32)
        mask_padded = torch.ones_like(target_full)
        mask_padded[0, n:] = 0.0
        data = FakeDataset(pd.DataFrame({"month": months}))
        l_pad, b_pad, _, _ = model.forward(params, data, target_full, mask_padded)

        target_short = target_full[:, :n]
        mask_short = torch.ones_like(target_short)
        data_short = FakeDataset(pd.DataFrame({"month": months[:n]}))
        model._init_cache = {}
        l_short, b_short, _, _ = model.forward(
            params[:, :n], data_short, target_short, mask_short
        )

        np.testing.assert_allclose(l_pad.item(), l_short.item(), rtol=1e-5)
        np.testing.assert_allclose(b_pad.item(), b_short.item(), rtol=1e-5)


class TestHyperTreeETSRecursiveValidationMetric:
    """The validation metric is the recursive h-step forecast, not the
    degenerate in-sample one-step fit.

    For the seasonal variants with ``fcst_h <= season_length`` the in-sample
    validation loss collapses to a parameter-independent fixed point (~0 for any
    parameters), so early stopping would select noise. These tests pin the fix:
    the validation curve is on a meaningful scale and varies with the boosted
    parameters/states, and the recursive branch matches the shared
    ``_roll_forecast`` rollout used by ``forecast``.
    """

    LGB_PARAMS = {"learning_rate": 0.1, "num_leaves": 15,
                  "min_data_in_leaf": 1, "min_data_in_bin": 1}

    def _seasonal_panel(self, n_series=3, n_periods=48, freq="MS"):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2015-01-01", periods=n_periods, freq=freq)
        t = np.arange(n_periods)
        frames = []
        for sid in range(n_series):
            seasonal = 10.0 * np.sin(2 * np.pi * (dates.month - 1) / 12.0)
            value = 100.0 + 0.5 * t + seasonal + rng.standard_normal(n_periods)
            frames.append(pd.DataFrame({
                "series_id": sid, "date": dates, "value": value,
                "month": dates.month, "feature1": rng.standard_normal(n_periods),
            }))
        return pd.concat(frames, ignore_index=True)

    @pytest.mark.parametrize("ets_type", ["triple", "additive"])
    def test_validation_metric_is_not_degenerate(self, ets_type):
        """fcst_h == season_length is the exact-degenerate case; the recursive
        metric must be on the data scale and move across iterations."""
        train = self._seasonal_panel()
        model = HyperTreeETS(ets_type=ets_type, season_length=12,
                             seasonality_feature="month", freq="M", fcst_h=12)
        result = model.train(
            lgb_params=self.LGB_PARAMS, num_iterations=25, train_data=train,
            validation=True, early_stopping_round=999,
        )
        vals = np.asarray(next(iter(result.validation_metrics.values())), dtype=float)
        assert vals.size >= 5
        assert np.all(np.isfinite(vals))
        # Not the ~1e-12 degenerate in-sample metric: forecast errors on a
        # ~100-scale seasonal series give an MSE far above any rounding floor.
        assert vals.min() > 1e-2
        # The metric depends on the boosted parameters/states, so it moves.
        assert np.ptp(vals) > 1e-6

    def test_train_and_validation_metrics_differ(self):
        """The train metric stays teacher-forced in-sample; the validation
        metric is the recursive forecast, so the two curves are not identical."""
        train = self._seasonal_panel()
        model = HyperTreeETS(ets_type="triple", season_length=12,
                             seasonality_feature="month", freq="M", fcst_h=12)
        result = model.train(
            lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
            validation=True, early_stopping_round=999,
        )
        train_vals = np.asarray(next(iter(result.train_metrics.values())), dtype=float)
        val_vals = np.asarray(next(iter(result.validation_metrics.values())), dtype=float)
        assert not np.allclose(train_vals, val_vals)

    def test_eval_fn_recursive_branch_matches_roll_forecast(self):
        """The validation branch of eval_fn reproduces the shared rollout used
        by forecast (correct F-order predt reshape and masking)."""
        model = HyperTreeETS(ets_type="triple", season_length=4,
                             seasonality_feature="month", fcst_h=3)
        model.n_series = 2
        level = torch.tensor([100.0, 50.0])
        trend = torch.tensor([1.0, -0.5])
        seas = torch.tensor([[1.1, 0.9, 1.05, 0.95], [1.2, 0.8, 1.1, 0.9]])
        model._eval_boundary = (level, trend, seas)

        val_df = pd.DataFrame({"month": [1, 2, 3, 1, 2, 3],
                               "mask": [1, 1, 1, 1, 1, 1]})
        target = np.array([[101.0, 102.0, 103.0], [49.0, 48.0, 47.0]])
        mock_eval = Mock()
        mock_eval.data = val_df
        mock_eval.get_label.return_value = target.reshape(-1)
        model.dataset_references = {id(mock_eval): "validation"}

        rng = np.random.default_rng(0)
        predt = rng.standard_normal(model.n_series * model.fcst_h * model.n_params)
        name, loss_val, is_higher_better = model.eval_fn(predt, mock_eval)

        params = torch.clamp(
            torch.sigmoid(torch.tensor(
                predt.reshape(-1, model.n_params, order="F"), dtype=torch.float32
            ).reshape(model.n_series, -1, model.n_params)),
            min=model.eps, max=1 - model.eps,
        )
        idxs = (torch.tensor(val_df["month"].values).reshape(model.n_series, -1).long() - 1)
        expected = model._roll_forecast(level, trend, seas, params, idxs)
        target_t = torch.tensor(target, dtype=torch.float32)
        exp_loss = nn.MSELoss()(expected, target_t).item()

        assert name == "MSELoss"
        assert is_higher_better is False
        assert abs(loss_val - exp_loss) < 1e-5

    def test_roll_forecast_does_not_mutate_seasonality(self):
        """The shared rollout must clone seasonality so repeated calls (forecast
        plus the validation metric) are independent."""
        model = HyperTreeETS(ets_type="triple", season_length=4,
                             seasonality_feature="month", fcst_h=3)
        model.n_series = 1
        seas = torch.tensor([[1.1, 0.9, 1.05, 0.95]])
        seas_before = seas.clone()
        params = torch.full((1, 3, 4), 0.3)
        idxs = torch.tensor([[0, 1, 2]])
        model._roll_forecast(torch.tensor([100.0]), torch.tensor([1.0]),
                             seas, params, idxs)
        assert torch.allclose(seas, seas_before)
