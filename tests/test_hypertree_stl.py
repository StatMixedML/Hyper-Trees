import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from unittest.mock import Mock, patch

from hypertrees.models.HyperTreeSTL import HyperTreeSTL
from hypertrees.utils import TrainingResult


class TestHyperTreeSTLInitialization:
    """Test HyperTreeSTL initialization and parameter validation."""

    def test_default_initialization(self):
        model = HyperTreeSTL()
        assert model.period == 12
        assert model.num_seasonal_components == 1
        assert model.n_params == 2 + 2 * 1 + 1  # +1 for default type trend smoothing
        assert model.freq == "M"
        assert model.fcst_h == 12
        assert isinstance(model.loss_fn, nn.MSELoss)
        assert model.loss_name == "MSELoss"
        assert model.dtype == torch.float32
        assert model.forward_type == "default"
        assert model.model is None
        assert model.features is None
        assert model.is_trained is False

    def test_custom_initialization(self):
        loss_fn = nn.L1Loss()
        model = HyperTreeSTL(period=24, num_seasonal_components=2, freq="D", fcst_h=6, loss_fn=loss_fn)
        assert model.period == 24
        assert model.num_seasonal_components == 2
        assert model.n_params == 2 + 2 * 2 + 1  # +1 for default type trend smoothing
        assert model.freq == "D"
        assert model.fcst_h == 6
        assert model.loss_fn is loss_fn
        assert model.loss_name == "L1Loss"
        assert model.forward_type == "default"

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="Period must be a positive integer"):
            HyperTreeSTL(period=0)

    def test_invalid_fcst_h(self):
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeSTL(fcst_h=0)

    def test_invalid_loss_function(self):
        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeSTL(loss_fn="not_a_loss")

    def test_invalid_freq_type(self):
        with pytest.raises(TypeError, match="freq must be a string"):
            HyperTreeSTL(freq=123)

    def test_paper_type_initialization(self):
        model = HyperTreeSTL(type="paper")
        assert model.forward_type == "paper"
        assert model.n_params == 2 + 2 * 1  # No extra parameter for paper type
        assert model._forward == model._forward_paper

    def test_default_type_initialization(self):
        model = HyperTreeSTL(type="default")
        assert model.forward_type == "default"
        assert model.n_params == 2 + 2 * 1 + 1  # +1 extra parameter for default type
        assert model._forward == model._forward_default

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Type must be either 'default' or 'paper'"):
            HyperTreeSTL(type="invalid")

    def test_parameter_count_paper_type(self):
        """Test parameter count for paper type models."""
        model = HyperTreeSTL(period=12, num_seasonal_components=3, type="paper")
        expected_params = 2 + 2 * 3  # trend(2) + seasonal(2*num_components)
        assert model.n_params == expected_params

    def test_parameter_count_default_type(self):
        """Test parameter count for default type models."""
        model = HyperTreeSTL(period=12, num_seasonal_components=3, type="default")
        expected_params = 2 + 2 * 3 + 1  # trend(2) + seasonal(2*num_components) + smoothing(1)
        assert model.n_params == expected_params


class TestHyperTreeSTLTraining:
    """Test HyperTreeSTL training functionality."""

    @pytest.fixture
    def sample_train_data(self):
        np.random.seed(0)
        n_series = 1
        n_periods = 24
        dates = pd.date_range("2020-01-01", periods=n_periods, freq="ME")
        data = []
        for sid in range(n_series):
            df = pd.DataFrame(
                {
                    "series_id": sid,
                    "date": dates,
                    "value": (np.sin(np.arange(n_periods) / 6.0) * 5 + 50) + np.random.randn(n_periods),
                    "feature1": np.random.randn(n_periods),
                }
            )
            # STL expects a time index feature to be present/used
            df["time"] = np.arange(1, n_periods + 1)
            data.append(df)
        return pd.concat(data, ignore_index=True)

    def test_train_parameter_validation(self, sample_train_data):
        model = HyperTreeSTL()

        with pytest.raises(ValueError, match="train_data must be provided"):
            model.train(lgb_params={}, num_iterations=10)

        with pytest.raises(ValueError, match="lgb_params must be provided"):
            model.train(train_data=sample_train_data, num_iterations=10)

        with pytest.raises(TypeError, match="train_data must be a pandas DataFrame"):
            model.train(lgb_params={}, train_data="invalid", num_iterations=10)

        with pytest.raises(TypeError, match="lgb_params must be a dictionary"):
            model.train(lgb_params="invalid", train_data=sample_train_data, num_iterations=10)

        with pytest.raises(ValueError, match="num_iterations must be a positive integer"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=0)

        with pytest.raises(ValueError, match="early_stopping_round must be provided when validation is True"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, validation=True)

        with pytest.raises(ValueError, match="early_stopping_round can only be used when validation is True"):
            model.train(
                lgb_params={}, train_data=sample_train_data, num_iterations=10, validation=False, early_stopping_round=5
            )

    def test_train_type_validation(self, sample_train_data):
        """Test training parameter type validation for seed, verbose, validation, deterministic."""
        model = HyperTreeSTL()

        with pytest.raises(TypeError, match="seed must be an integer"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, seed="bad")

        with pytest.raises(TypeError, match="verbose must be an integer"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, verbose="bad")

        with pytest.raises(TypeError, match="validation must be a boolean"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, validation="yes")

        with pytest.raises(TypeError, match="deterministic must be a boolean"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, deterministic="yes")

        with pytest.raises(ValueError, match="early_stopping_round must be a positive integer"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10, validation=True, early_stopping_round=-1)

    def test_train_multi_series_raises(self):
        """Test that multi-series data raises NotImplementedError."""
        model = HyperTreeSTL()
        df = pd.DataFrame({
            'series_id': [0]*10 + [1]*10,
            'date': list(pd.date_range('2020-01-01', periods=10, freq='MS')) * 2,
            'value': np.random.randn(20),
            'time': list(range(1, 11)) * 2,
        })
        with pytest.raises(RuntimeError, match="only supports univariate training"):
            model.train(lgb_params={'learning_rate': 0.1}, train_data=df, num_iterations=10)

    def test_required_columns_validation(self):
        model = HyperTreeSTL()

        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "value": np.random.randn(10)})
        with pytest.raises(ValueError, match="Required column 'series_id' not found"):
            model.train(lgb_params={}, train_data=df, num_iterations=10)

        df = pd.DataFrame({"series_id": [0] * 10, "value": np.random.randn(10)})
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            model.train(lgb_params={}, train_data=df, num_iterations=10)

        df = pd.DataFrame({"series_id": [0] * 10, "date": pd.date_range("2020-01-01", periods=10), "time": range(1, 11)})
        with pytest.raises(ValueError, match="Required column 'value' not found"):
            model.train(lgb_params={}, train_data=df, num_iterations=10)

    @patch("hypertrees.models.HyperTreeSTL.lgb.train")
    @patch("hypertrees.models.HyperTreeSTL.prepare_datasets")
    @patch("hypertrees.models.HyperTreeSTL.TimeSeriesPreprocessor")
    def test_successful_training(self, mock_preprocessor, mock_prepare_datasets, mock_lgb_train, sample_train_data):
        model = HyperTreeSTL(period=12, num_seasonal_components=1)

        # Mock preprocessor to return the same data and expose features incl. time_idx
        mock_pre = Mock()
        mock_preprocessor.return_value = mock_pre
        mock_pre.create_lags.return_value = sample_train_data
        mock_pre.extract.return_value = {"features": pd.DataFrame({"feature1": [1, 2], "time": [3, 4]})}

        # Mock dataset preparation (no validation path)
        mock_valid_sets = [Mock()]
        mock_prepare_datasets.return_value = (
            mock_valid_sets,
            ["train"],
            None,
            None,
            None,
            None,
            {},
        )

        # Mock LightGBM training
        mock_model = Mock()
        mock_model.best_iteration = 11
        mock_lgb_train.return_value = mock_model

        result = model.train(lgb_params={"learning_rate": 0.1}, num_iterations=12, train_data=sample_train_data)

        assert model.is_trained is True
        assert model.model is mock_model
        assert model.features == ["feature1", "time"]
        assert isinstance(result, TrainingResult)
        assert result.train_metrics == {"loss": []}
        assert result.validation_metrics is None
        assert result.best_iteration == 10  # best_iteration-1
        assert result.training_time is not None


class TestHyperTreeSTLForecasting:
    """Test HyperTreeSTL forecasting functionality."""

    @pytest.fixture
    def trained_model(self):
        model = HyperTreeSTL(period=12, num_seasonal_components=1, fcst_h=24)
        model.is_trained = True
        model.n_series = 1
        model.train_series_id = 0
        model.features = ["feature1", "time"]

        n_params = model.n_params
        mock_pred = np.random.randn(24 * n_params)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = mock_pred
        model.model = mock_lgb_model
        return model

    @pytest.fixture
    def sample_test_data(self):
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        df = pd.DataFrame(
            {
                "series_id": [0] * 24,
                "date": dates,
                "feature1": np.random.randn(24),
                "time": np.arange(101, 125),
            }
        )
        return df

    def test_forecast_untrained_model(self):
        model = HyperTreeSTL()
        with pytest.raises(RuntimeError, match="Model has not been trained"):
            model.forecast(test_data=pd.DataFrame())

    def test_forecast_parameter_validation(self, trained_model):
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            trained_model.forecast(test_data=pd.DataFrame({"series_id": [0], "time": [1]}))

        with pytest.raises(ValueError, match="Required column 'time' not found"):
            trained_model.forecast(test_data=pd.DataFrame({"series_id": [0], "date": ["2023-01-01"]}))

        # Wrong series ID
        bad_series = pd.DataFrame({
            "series_id": [99] * 24,
            "date": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "feature1": np.random.randn(24),
            "time": np.arange(1, 25),
        })
        with pytest.raises(ValueError, match="test_data series_id must match"):
            trained_model.forecast(test_data=bad_series)

        # Missing training feature (must have fcst_h rows)
        bad = pd.DataFrame({
            "series_id": [0] * 24,
            "date": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "time": np.arange(1, 25),
            # missing 'feature1'
        })
        with pytest.raises(ValueError, match="Missing features in test_data"):
            trained_model.forecast(test_data=bad)

        # Invalid type
        ok = pd.DataFrame({
            "series_id": [0] * 24,
            "date": pd.date_range("2023-01-01", periods=24, freq="MS"),
            "feature1": np.random.randn(24),
            "time": np.arange(1, 25),
        })
        with pytest.raises(ValueError, match="Parameter 'type' must be either 'forecast', 'parameters', or 'components'"):
            trained_model.forecast(test_data=ok, type="invalid")

    def test_forecast_success(self, trained_model, sample_test_data):
        result = trained_model.forecast(test_data=sample_test_data, type="forecast")
        assert isinstance(result, pd.DataFrame)
        assert set(["series_id", "date", "fcst", "model"]).issubset(result.columns)
        assert len(result) == len(sample_test_data)
        assert result["model"].iloc[0] == f"Hyper-Tree-STL(period={trained_model.period})"

    def test_forecast_components_output(self, trained_model, sample_test_data):
        result = trained_model.forecast(test_data=sample_test_data, type="components")
        assert isinstance(result, pd.DataFrame)
        assert set(["series_id", "date", "trend", "seasonality", "model"]).issubset(result.columns)
        assert len(result) == len(sample_test_data)

    def test_forecast_parameters_output(self, trained_model, sample_test_data):
        result = trained_model.forecast(test_data=sample_test_data, type="parameters")
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "series_id",
            "date",
            "model",
            "trend_intercept",
            "trend_slope",
            "seasonal_sine1",
            "seasonal_cosine1",
        }
        assert expected_cols.issubset(result.columns)
        assert len(result) == len(sample_test_data)
        assert result["model"].iloc[0] == f"Hyper-Tree-STL(period={trained_model.period})"
        # Basic sanity: parameter columns contain numeric values and no NaNs
        params_df = result[[
            "trend_intercept",
            "trend_slope",
            "seasonal_sine1",
            "seasonal_cosine1",
        ]]
        assert params_df.apply(pd.api.types.is_numeric_dtype).all()
        assert not params_df.isna().any().any()

    @pytest.fixture
    def trained_model_paper(self):
        """Create a paper-type trained model for testing."""
        model = HyperTreeSTL(period=12, num_seasonal_components=1, fcst_h=24, type="paper")
        model.is_trained = True
        model.n_series = 1
        model.train_series_id = 0
        model.features = ["feature1", "time"]

        n_params = model.n_params
        mock_pred = np.random.randn(24 * n_params)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = mock_pred
        model.model = mock_lgb_model
        return model

    def test_forecast_paper_type_parameters(self, trained_model_paper, sample_test_data):
        """Test that paper type model generates correct parameter structure."""
        result = trained_model_paper.forecast(test_data=sample_test_data, type="parameters")
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "series_id",
            "date",
            "model",
            "trend_intercept",
            "trend_slope",
            "seasonal_sine1",
            "seasonal_cosine1",
        }
        assert expected_cols.issubset(result.columns)
        assert len(result) == len(sample_test_data)

    def test_forecast_parameters_default_values(self):
        """Verify default-type parameter output maps to correct parameter indices."""
        fcst_h = 24
        model = HyperTreeSTL(period=12, num_seasonal_components=1, fcst_h=fcst_h, type="default")
        model.is_trained = True
        model.n_series = 1
        model.train_series_id = 0
        model.features = ["feature1", "time"]

        # n_params = 2 (trend) + 1 (window) + 2 (sine+cosine) = 5
        # Fortran reshape: column-major from (fcst_h * n_series, n_params)
        raw = np.arange(fcst_h * 5, dtype=float)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = raw
        model.model = mock_lgb_model

        test_data = pd.DataFrame({
            "series_id": [0] * fcst_h,
            "date": pd.date_range("2022-01-01", periods=fcst_h, freq="MS"),
            "feature1": np.random.randn(fcst_h),
            "time": np.arange(101, 101 + fcst_h),
        })

        result = model.forecast(test_data=test_data, type="parameters")

        # params_fcst is reshaped from Fortran order: (3, 1, 5)
        params_fcst = torch.tensor(raw.reshape(-1, 1, 5, order="F"))
        np.testing.assert_allclose(result["trend_intercept"].values, params_fcst[:, 0, 0].numpy())
        np.testing.assert_allclose(result["trend_slope"].values, params_fcst[:, 0, 1].numpy())
        assert "trend_window_logit" in result.columns
        np.testing.assert_allclose(result["trend_window_logit"].values, params_fcst[:, 0, 2].numpy())
        np.testing.assert_allclose(result["seasonal_sine1"].values, params_fcst[:, 0, 3].numpy())
        np.testing.assert_allclose(result["seasonal_cosine1"].values, params_fcst[:, 0, 4].numpy())

    def test_forecast_parameters_paper_no_window_logit(self):
        """Verify paper-type omits trend_window_logit and uses correct seasonal offset."""
        fcst_h = 24
        model = HyperTreeSTL(period=12, num_seasonal_components=1, fcst_h=fcst_h, type="paper")
        model.is_trained = True
        model.n_series = 1
        model.train_series_id = 0
        model.features = ["feature1", "time"]

        # n_params = 2 (trend) + 2 (sine+cosine) = 4
        raw = np.arange(fcst_h * 4, dtype=float)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = raw
        model.model = mock_lgb_model

        test_data = pd.DataFrame({
            "series_id": [0] * fcst_h,
            "date": pd.date_range("2022-01-01", periods=fcst_h, freq="MS"),
            "feature1": np.random.randn(fcst_h),
            "time": np.arange(101, 101 + fcst_h),
        })

        result = model.forecast(test_data=test_data, type="parameters")

        assert "trend_window_logit" not in result.columns
        params_fcst = torch.tensor(raw.reshape(-1, 1, 4, order="F"))
        # Paper type: seasonal starts at index 2 (no window param)
        np.testing.assert_allclose(result["seasonal_sine1"].values, params_fcst[:, 0, 2].numpy())
        np.testing.assert_allclose(result["seasonal_cosine1"].values, params_fcst[:, 0, 3].numpy())

    def test_forecast_parameters_multiple_seasonal_components(self):
        """Verify parameter output with multiple Fourier seasonal components."""
        fcst_h = 24
        model = HyperTreeSTL(period=12, num_seasonal_components=3, fcst_h=fcst_h, type="default")
        model.is_trained = True
        model.n_series = 1
        model.train_series_id = 0
        model.features = ["feature1", "time"]

        # n_params = 2 (trend) + 1 (window) + 6 (3 sine + 3 cosine) = 9
        raw = np.arange(fcst_h * 9, dtype=float)
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = raw
        model.model = mock_lgb_model

        test_data = pd.DataFrame({
            "series_id": [0] * fcst_h,
            "date": pd.date_range("2022-01-01", periods=fcst_h, freq="MS"),
            "feature1": np.random.randn(fcst_h),
            "time": np.arange(101, 101 + fcst_h),
        })

        result = model.forecast(test_data=test_data, type="parameters")

        params_fcst = torch.tensor(raw.reshape(-1, 1, 9, order="F"))
        # seasonal_offset=3 for default type
        for i in range(3):
            np.testing.assert_allclose(
                result[f"seasonal_sine{i+1}"].values, params_fcst[:, 0, 3 + i].numpy()
            )
            np.testing.assert_allclose(
                result[f"seasonal_cosine{i+1}"].values, params_fcst[:, 0, 3 + 3 + i].numpy()
            )


class TestHyperTreeSTLForwardMethods:
    """Test both forward methods (_forward_default and _forward_paper)."""

    def test_forward_paper_method(self):
        model = HyperTreeSTL(period=12, num_seasonal_components=2, type="paper")
        n_samples = 36  # Use longer sequence for consistency
        n_series = 2
        n_params = model.n_params  # Should be 6 for paper type (2 + 2*2)

        # Create random parameters and time indices
        params = torch.randn(n_samples, n_series, n_params)
        time_idx = torch.arange(1, n_samples + 1, dtype=torch.float32).unsqueeze(1).repeat(1, n_series)

        trend, seasonality = model._forward_paper(params, time_idx)

        assert trend.shape == (n_samples, n_series)
        assert seasonality.shape == (n_samples, n_series)
        assert torch.allclose(torch.mean(seasonality, dim=0), torch.zeros(n_series), atol=1e-5)

    def test_forward_default_method(self):
        model = HyperTreeSTL(period=12, num_seasonal_components=2, type="default")
        n_samples = 36  # Use longer sequence to accommodate padding requirements
        n_series = 2
        n_params = model.n_params  # Should be 7 for default type (2 + 2*2 + 1)

        # Create random parameters and time indices
        params = torch.randn(n_samples, n_series, n_params)
        time_idx = torch.arange(1, n_samples + 1, dtype=torch.float32).unsqueeze(1).repeat(1, n_series)

        trend, seasonality = model._forward_default(params, time_idx)

        assert trend.shape == (n_samples, n_series)
        assert seasonality.shape == (n_samples, n_series)

    def test_forward_method_selection(self):
        """Test that the correct forward method is selected based on type."""
        model_paper = HyperTreeSTL(type="paper")
        model_default = HyperTreeSTL(type="default")

        assert model_paper._forward == model_paper._forward_paper
        assert model_default._forward == model_default._forward_default


class TestHyperTreeSTLCoreMethods:
    """Unit tests for core objective and gradient utilities."""

    def test_get_params_loss(self):
        model = HyperTreeSTL(period=6, num_seasonal_components=1, type="default")
        n_samples = 10
        n_params = model.n_params  # Should be 5 for default type (2 + 2*1 + 1)
        predt = np.random.randn(n_samples * n_params)
        target = torch.randn(n_samples, 1)
        model.n_series = 1
        time_idx = torch.arange(1, n_samples + 1, dtype=torch.float32).reshape(-1, 1)

        params, loss = model.get_params_loss(predt, target, time_idx, requires_grad=False)
        assert isinstance(params, torch.nn.Parameter)
        assert params.shape == (n_samples, n_params)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_get_params_loss_requires_grad(self):
        model = HyperTreeSTL(period=12, num_seasonal_components=2, type="default")
        n_samples = 12
        n_params = model.n_params  # Should be 7 for default type (2 + 2*2 + 1)
        predt = np.random.randn(n_samples * n_params)
        target = torch.randn(n_samples, 1)
        model.n_series = 1
        time_idx = torch.arange(1, n_samples + 1, dtype=torch.float32).reshape(-1, 1)

        params, loss = model.get_params_loss(predt, target, time_idx, requires_grad=True)
        assert params.requires_grad is True
        assert loss.requires_grad is True

    def test_calculate_gradients_and_hessians(self):
        model = HyperTreeSTL(period=6, type="default")
        n_params = model.n_params  # Should be 5 for default type
        params = torch.randn(5, n_params, requires_grad=True)
        loss = torch.sum(params ** 2)
        grad, hess = model._calculate_gradients_and_hessians(loss, params)
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (25,)  # 5 * 5 = 25
        assert hess.shape == (25,)

    @patch.object(HyperTreeSTL, "get_params_loss")
    def test_eval_fn_returns_metric(self, mock_get_params_loss):
        """Test eval_fn with unknown dataset (defaults to train time_idx)."""
        model = HyperTreeSTL(period=6, type="default")
        model.n_series = 1
        model.dataset_references = {}  # Initialize dataset references (empty = unknown)
        # Set time_idx_train as it's required by eval_fn
        model.time_idx_train = torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1)
        n_params = model.n_params  # Should be 5 for default type
        mock_get_params_loss.return_value = (torch.randn(8, n_params), torch.tensor(0.987))

        # Mock lgb.Dataset to supply labels
        mock_eval = Mock(spec=lgb.Dataset)
        mock_eval.get_label.return_value = np.random.randn(8)

        predt = np.random.randn(8 * n_params)
        name, value, is_higher_better = model.eval_fn(predt, mock_eval)
        assert isinstance(name, str)
        assert name == "MSELoss"
        assert isinstance(value, float)
        assert is_higher_better is False

        # Verify get_params_loss was called with time_idx_train (default for unknown)
        mock_get_params_loss.assert_called_once()
        call_args = mock_get_params_loss.call_args
        assert torch.equal(call_args[0][2], model.time_idx_train)  # time_idx argument

    @patch.object(HyperTreeSTL, "get_params_loss")
    def test_eval_fn_with_train_dataset(self, mock_get_params_loss):
        """Test eval_fn with train dataset reference."""
        model = HyperTreeSTL(period=6, type="default")
        model.n_series = 1
        # Set time_idx for both train and eval
        model.time_idx_train = torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1)
        model.time_idx_eval = torch.arange(9, 17, dtype=torch.float32).reshape(-1, 1)
        n_params = model.n_params
        mock_get_params_loss.return_value = (torch.randn(8, n_params), torch.tensor(0.123))

        # Mock lgb.Dataset and register it as "train"
        mock_train = Mock(spec=lgb.Dataset)
        mock_train.get_label.return_value = np.random.randn(8)
        model.dataset_references = {id(mock_train): "train"}

        predt = np.random.randn(8 * n_params)
        name, value, is_higher_better = model.eval_fn(predt, mock_train)

        assert name == "MSELoss"
        assert abs(value - 0.123) < 1e-6  # Use approximate equality for float precision
        assert is_higher_better is False

        # Verify get_params_loss was called with time_idx_train
        mock_get_params_loss.assert_called_once()
        call_args = mock_get_params_loss.call_args
        assert torch.equal(call_args[0][2], model.time_idx_train)  # time_idx argument
        assert call_args[1]["requires_grad"] is False

    @patch.object(HyperTreeSTL, "get_params_loss")
    def test_eval_fn_with_validation_dataset(self, mock_get_params_loss):
        """Test eval_fn with validation dataset reference."""
        model = HyperTreeSTL(period=6, type="default")
        model.n_series = 1
        # Set time_idx for both train and eval
        model.time_idx_train = torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1)
        model.time_idx_eval = torch.arange(9, 17, dtype=torch.float32).reshape(-1, 1)
        n_params = model.n_params
        mock_get_params_loss.return_value = (torch.randn(8, n_params), torch.tensor(0.456))

        # Mock lgb.Dataset and register it as "validation"
        mock_validation = Mock(spec=lgb.Dataset)
        mock_validation.get_label.return_value = np.random.randn(8)
        model.dataset_references = {id(mock_validation): "validation"}

        predt = np.random.randn(8 * n_params)
        name, value, is_higher_better = model.eval_fn(predt, mock_validation)

        assert name == "MSELoss"
        assert abs(value - 0.456) < 1e-6  # Use approximate equality for float precision
        assert is_higher_better is False

        # Verify get_params_loss was called with time_idx_eval
        mock_get_params_loss.assert_called_once()
        call_args = mock_get_params_loss.call_args
        assert torch.equal(call_args[0][2], model.time_idx_eval)  # time_idx argument
        assert call_args[1]["requires_grad"] is False

    @patch.object(HyperTreeSTL, "get_params_loss")
    def test_objective_fn_builds_tensors_and_returns_grad_hess(self, mock_get_params_loss):
        """Objective uses dataset labels and time_idx, then computes grad/hess."""
        model = HyperTreeSTL(period=6, type="default")
        model.n_series = 1
        N = 6
        # Set time_idx_train as it's required by objective_fn
        model.time_idx_train = torch.arange(1, N + 1, dtype=torch.float32).reshape(-1, 1)
        n_params = model.n_params  # Should be 5 for default type
        predt = np.random.randn(N * n_params)

        # Mock Dataset interface used by objective_fn
        labels = np.random.randn(N)
        df = pd.DataFrame({"time": np.arange(1, N + 1)})
        mock_data = Mock(spec=lgb.Dataset)
        mock_data.get_label.return_value = labels
        mock_data.get_data.return_value = df

        # Provide simple differentiable params and loss to enable gradient computation
        params = torch.randn(N, n_params, requires_grad=True)
        loss = torch.sum(params ** 2)

        def side_effect(predt_arg, target, time_idx, requires_grad):
            # Validate target and time_idx construction and dtype/shape
            assert isinstance(target, torch.Tensor)
            assert isinstance(time_idx, torch.Tensor)
            assert target.shape == (N, 1)
            assert time_idx.shape == (N, 1)
            assert target.dtype == model.dtype
            assert time_idx.dtype == model.dtype
            assert requires_grad is True
            return params, loss

        mock_get_params_loss.side_effect = side_effect

        grad, hess = model.objective_fn(predt, mock_data)
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (N * n_params,)
        assert hess.shape == (N * n_params,)
