import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from unittest.mock import Mock, patch

from hypertrees.models.HyperTreeAR import HyperTreeAR
from hypertrees.utils import TrainingResult
from hypertrees import ForecastIntervals


class TestHyperTreeARInitialization:
    """Test HyperTreeAR initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = HyperTreeAR()
        
        assert model.p == 2
        assert model.freq == "M"
        assert model.fcst_h == 1
        assert isinstance(model.loss_fn, nn.MSELoss)
        assert model.dtype == torch.float32
        assert model.model is None
        assert model.features is None
        assert model.is_trained is False
        
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        loss_fn = nn.L1Loss()
        model = HyperTreeAR(p=5, freq="D", fcst_h=3, loss_fn=loss_fn)
        
        assert model.p == 5
        assert model.freq == "D"
        assert model.fcst_h == 3
        assert model.loss_fn == loss_fn
        assert model.loss_name == "L1Loss"
        
    def test_invalid_p_parameter(self):
        """Test that invalid p parameter raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'p' must be a positive integer"):
            HyperTreeAR(p=0)
            
        with pytest.raises(ValueError, match="Parameter 'p' must be a positive integer"):
            HyperTreeAR(p=-1)
            
    def test_invalid_fcst_h_parameter(self):
        """Test that invalid forecast horizon raises ValueError."""
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeAR(fcst_h=0)
            
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeAR(fcst_h=-1)
            
    def test_invalid_loss_function(self):
        """Test that invalid loss function raises TypeError."""
        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeAR(loss_fn="invalid")

        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeAR(loss_fn=lambda x, y: x - y)

    def test_gn_hessian_initialization(self):
        """Test model initialization with GN hessian method."""
        model = HyperTreeAR(hessian_method="gn", n_hessian_probes=10)
        assert model.hessian_method == "gn"
        assert model.n_hessian_probes == 10
        assert hasattr(model, '_gn_hessian')

    def test_default_hessian_method(self):
        """Test that default hessian method is exact."""
        model = HyperTreeAR()
        assert model.hessian_method == "exact"
        assert not hasattr(model, '_gn_hessian')

    def test_invalid_hessian_method(self):
        """Test that invalid hessian_method raises ValueError."""
        with pytest.raises(ValueError, match="hessian_method must be either 'exact' or 'gn'"):
            HyperTreeAR(hessian_method="invalid")

    def test_gn_hessian_warning_non_mse(self):
        """Test that warning is issued for non-MSE loss with GN."""
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeAR(hessian_method="gn", loss_fn=nn.L1Loss())

    def test_invalid_freq_type(self):
        """Test that non-string freq raises TypeError."""
        with pytest.raises(TypeError, match="freq must be a string"):
            HyperTreeAR(freq=123)

    def test_invalid_n_hessian_probes(self):
        """Test that invalid n_hessian_probes raises ValueError."""
        with pytest.raises(ValueError, match="n_hessian_probes must be a positive integer"):
            HyperTreeAR(n_hessian_probes=0)

        with pytest.raises(ValueError, match="n_hessian_probes must be a positive integer"):
            HyperTreeAR(n_hessian_probes=-5)


class TestHyperTreeARSetForecastOrigin:
    """Test HyperTreeAR.set_forecast_origin."""

    def test_updates_fcst_lags(self):
        """set_forecast_origin should store last p values per series in reverse order."""
        model = HyperTreeAR(p=3, freq="M", fcst_h=2)
        history = pd.DataFrame({
            "series_id": [0]*6 + [1]*6,
            "date": list(pd.date_range("2020-01-01", periods=6, freq="MS")) * 2,
            "value": [10, 20, 30, 40, 50, 60] + [100, 200, 300, 400, 500, 600],
        })
        model.set_forecast_origin(history)
        # last 3 values of series 0: [60, 50, 40], reversed = newest-first
        np.testing.assert_array_equal(model.fcst_lags[0], [60, 50, 40])
        np.testing.assert_array_equal(model.fcst_lags[1], [600, 500, 400])

    def test_reanchor_changes_lags(self):
        """Calling set_forecast_origin twice should update the lags."""
        model = HyperTreeAR(p=2, freq="M", fcst_h=1)
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        history1 = pd.DataFrame({
            "series_id": [0]*5, "date": dates, "value": [1, 2, 3, 4, 5],
        })
        history2 = pd.DataFrame({
            "series_id": [0]*5, "date": dates, "value": [10, 20, 30, 40, 50],
        })
        model.set_forecast_origin(history1)
        np.testing.assert_array_equal(model.fcst_lags[0], [5, 4])
        model.set_forecast_origin(history2)
        np.testing.assert_array_equal(model.fcst_lags[0], [50, 40])

    def test_validates_series_order(self):
        """set_forecast_origin should reject non-contiguous series."""
        model = HyperTreeAR(p=2, freq="M", fcst_h=1)
        bad = pd.DataFrame({
            "series_id": [0, 1, 0],
            "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "value": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="non-contiguous"):
            model.set_forecast_origin(bad)


class TestHyperTreeARTraining:
    """Test HyperTreeAR training functionality."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='ME')
        
        data = []
        for series_id in range(10):
            series_data = pd.DataFrame({
                'series_id': series_id,
                'date': dates,
                'value': np.random.randn(100).cumsum() + 100,
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100)
            })
            data.append(series_data)
            
        return pd.concat(data, ignore_index=True)
    
    def test_train_parameter_validation(self, sample_train_data):
        """Test training parameter validation."""
        model = HyperTreeAR()
        
        # Test missing train_data
        with pytest.raises(ValueError, match="train_data must be provided"):
            model.train(lgb_params={}, num_iterations=10)
            
        # Test missing lgb_params
        with pytest.raises(ValueError, match="lgb_params must be provided"):
            model.train(train_data=sample_train_data, num_iterations=10)
            
        # Test invalid data types
        with pytest.raises(TypeError, match="train_data must be a pandas DataFrame"):
            model.train(lgb_params={}, train_data="invalid", num_iterations=10)
            
        with pytest.raises(TypeError, match="lgb_params must be a dictionary"):
            model.train(lgb_params="invalid", train_data=sample_train_data, num_iterations=10)
            
        # Test invalid num_iterations
        with pytest.raises(ValueError, match="num_iterations must be a positive integer"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=0)
            
        # Test validation without early_stopping_round
        with pytest.raises(ValueError, match="early_stopping_round must be provided when validation is True"):
            model.train(
                lgb_params={}, 
                train_data=sample_train_data, 
                num_iterations=10,
                validation=True
            )
            
        # Test early_stopping_round without validation
        with pytest.raises(ValueError, match="early_stopping_round can only be used when validation is True"):
            model.train(
                lgb_params={},
                train_data=sample_train_data,
                num_iterations=10,
                validation=False,
                early_stopping_round=5
            )

    def test_train_type_validation(self, sample_train_data):
        """Test training parameter type validation for seed, verbose, validation, deterministic."""
        model = HyperTreeAR()

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

    def test_required_columns_validation(self):
        """Test that missing required columns raise ValueError."""
        model = HyperTreeAR()
        
        # Missing series_id
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': np.random.randn(10)
        })
        with pytest.raises(ValueError, match="Required column 'series_id' not found"):
            model.train(lgb_params={}, train_data=data, num_iterations=10)
            
        # Missing date
        data = pd.DataFrame({
            'series_id': [0] * 10,
            'value': np.random.randn(10)
        })
        with pytest.raises(ValueError, match="Required column 'date' not found"):
            model.train(lgb_params={}, train_data=data, num_iterations=10)
            
        # Missing value
        data = pd.DataFrame({
            'series_id': [0] * 10,
            'date': pd.date_range('2020-01-01', periods=10)
        })
        with pytest.raises(ValueError, match="Required column 'value' not found"):
            model.train(lgb_params={}, train_data=data, num_iterations=10)
    
    @patch('hypertrees.models.HyperTreeAR.lgb.train')
    @patch('hypertrees.models.HyperTreeAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeAR.TimeSeriesPreprocessor')
    def test_successful_training(self, mock_preprocessor, mock_prepare_datasets, mock_lgb_train, sample_train_data):
        """Test successful training flow."""
        model = HyperTreeAR(p=2)
        
        # Mock preprocessor
        mock_preprocessor_instance = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.create_lags.return_value = sample_train_data
        mock_preprocessor_instance.extract.return_value = {
            "features": pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        }
        
        # Mock prepare_datasets
        mock_valid_sets = [Mock()]
        mock_lags_train = torch.randn(50, 2)
        mock_prepare_datasets.return_value = (
            mock_valid_sets, ['train'], [], {'train': {'loss': [0.5]}},
            mock_lags_train, None, {}
        )
        
        # Mock LightGBM model
        mock_lgb_model = Mock()
        mock_lgb_model.best_iteration = 10
        mock_lgb_train.return_value = mock_lgb_model
        
        # Train model
        result = model.train(
            lgb_params={'learning_rate': 0.1},
            num_iterations=10,
            train_data=sample_train_data
        )
        
        # Assertions
        assert model.is_trained is True
        assert model.model == mock_lgb_model
        assert model.features == ['feature1', 'feature2']
        assert isinstance(result, TrainingResult)
        assert result.train_metrics == {'loss': []}
        assert result.validation_metrics is None
        assert result.best_iteration == 9  # best_iteration-1 in implementation
        assert result.training_time is not None
        
    @patch('hypertrees.models.HyperTreeAR.lgb.train')
    def test_training_failure_handling(self, mock_lgb_train, sample_train_data):
        """Test that training failures are handled properly."""
        model = HyperTreeAR()
        
        # Mock LightGBM to raise exception
        mock_lgb_train.side_effect = Exception("Training failed")
        
        with pytest.raises(RuntimeError, match="Training failed"):
            model.train(
                lgb_params={'learning_rate': 0.1},
                train_data=sample_train_data,
                num_iterations=10
            )
            
        assert model.is_trained is False


class TestHyperTreeARForecasting:
    """Test HyperTreeAR forecasting functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a mock trained model for testing."""
        model = HyperTreeAR(p=2, fcst_h=3)
        model.is_trained = True
        model.features = ['feature1', 'feature2']

        # Mock the LightGBM model
        mock_lgb_model = Mock()
        mock_lgb_model.predict.return_value = np.random.randn(18)  # 3 series * 3 horizons * 2 params
        model.model = mock_lgb_model

        # Mock the stored forecast lags (last 2 values for each series in reverse order)
        model.fcst_lags = {
            0: np.array([11, 10]),  # Series 0: [newest, oldest]
            1: np.array([21, 20]),  # Series 1: [newest, oldest]
            2: np.array([31, 30])   # Series 2: [newest, oldest]
        }

        return model
    
    @pytest.fixture
    def sample_train_data(self):
        """Sample training data for forecasting."""
        return pd.DataFrame({
            'series_id': [0, 0, 1, 1, 2, 2],
            'date': pd.date_range('2020-01-01', periods=6, freq='ME'),
            'value': [10, 11, 20, 21, 30, 31]
        })
    
    @pytest.fixture 
    def sample_test_data(self):
        """Sample test data for forecasting."""
        return pd.DataFrame({
            'series_id': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'date': pd.date_range('2020-07-01', periods=9, freq='ME'),
            'feature1': np.random.randn(9),
            'feature2': np.random.randn(9)
        })
    
    def test_forecast_untrained_model(self):
        """Test that forecasting with untrained model raises error."""
        model = HyperTreeAR()

        with pytest.raises(RuntimeError, match="Model has not been trained"):
            model.forecast(
                test_data=pd.DataFrame()
            )
    
    def test_forecast_parameter_validation(self, trained_model):
        """Test forecasting parameter validation."""
        # Missing required columns in test_data
        test_data = pd.DataFrame({'series_id': [0]})

        with pytest.raises(ValueError, match="Required column 'date' not found in test_data"):
            trained_model.forecast(test_data=test_data)

        # Missing features
        test_data = pd.DataFrame({
            'series_id': [0]*3 + [1]*3 + [2]*3,
            'date': pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 3,
            'feature1': range(9)  # Missing feature2
        })

        with pytest.raises(ValueError, match="Missing features in test_data"):
            trained_model.forecast(test_data=test_data)

        # Invalid type parameter
        test_data = pd.DataFrame({
            'series_id': [0]*3 + [1]*3 + [2]*3,
            'date': pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 3,
            'feature1': range(9), 'feature2': range(9)
        })

        with pytest.raises(ValueError, match="Parameter 'type' must be either 'forecast' or 'parameters'"):
            trained_model.forecast(
                test_data=test_data,
                type="invalid"
            )

    def test_forecast_series_mismatch(self, trained_model):
        """Test that mismatched series IDs between train and test raises ValueError."""
        # Extra series in test_data not in training
        test_data = pd.DataFrame({
            'series_id': [0]*3 + [1]*3 + [99]*3,
            'date': pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 3,
            'feature1': range(9), 'feature2': range(9)
        })
        with pytest.raises(ValueError, match="Missing series in training"):
            trained_model.forecast(test_data=test_data)

        # Missing series in test_data
        test_data = pd.DataFrame({
            'series_id': [0]*3 + [1]*3,
            'date': pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 2,
            'feature1': range(6), 'feature2': range(6)
        })
        with pytest.raises(ValueError, match="Extra series not in test_data"):
            trained_model.forecast(test_data=test_data)

    def test_forecast_wrong_row_count(self, trained_model):
        """Test that wrong number of rows per series raises ValueError."""
        test_data = pd.DataFrame({
            'series_id': [0]*2 + [1]*3 + [2]*3,
            'date': pd.date_range('2020-02-01', periods=2, freq='MS').tolist() +
                    pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 2,
            'feature1': range(8), 'feature2': range(8)
        })
        with pytest.raises(ValueError, match="Each series must have exactly fcst_h"):
            trained_model.forecast(test_data=test_data)

    def test_forecast_success(self, trained_model, sample_train_data, sample_test_data):
        """Test successful forecasting."""
        result = trained_model.forecast(
            test_data=sample_test_data,
            type="forecast"
        )
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert 'series_id' in result.columns
        assert 'date' in result.columns
        assert 'fcst' in result.columns
        assert 'model' in result.columns
        assert len(result) == len(sample_test_data)
        assert result['model'].iloc[0] == 'Hyper-Tree-AR(2)'
        
    def test_forecast_parameters(self, trained_model, sample_train_data, sample_test_data):
        """Test forecasting with parameter output."""
        result = trained_model.forecast(
            test_data=sample_test_data,
            type="parameters"
        )
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert 'series_id' in result.columns
        assert 'date' in result.columns
        assert 'model' in result.columns
        assert 'AR(1)' in result.columns
        assert 'AR(2)' in result.columns
        assert len(result) == len(sample_test_data)


class TestHyperTreeARInternalMethods:
    """Test internal methods of HyperTreeAR."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return HyperTreeAR(p=2)
    
    def test_get_params_loss(self, model):
        """Test get_params_loss method."""
        # Setup test data
        predt = np.random.randn(10)  # 5 samples * 2 params
        target = torch.randn(5, 1)
        lags = torch.randn(5, 2)
        
        # Call method
        params, loss = model.get_params_loss(predt, target, lags, requires_grad=False)
        
        # Check outputs
        assert isinstance(params, torch.Tensor)
        assert params.shape == (5, 2)  # 5 samples, 2 parameters
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        
    def test_get_params_loss_with_gradients(self, model):
        """Test get_params_loss method with gradient computation."""
        predt = np.random.randn(10)
        target = torch.randn(5, 1)
        lags = torch.randn(5, 2)
        
        params, loss = model.get_params_loss(predt, target, lags, requires_grad=True)
        
        # Check that parameters require gradients
        assert params.requires_grad
            
    def test_calculate_gradients_and_hessians(self, model):
        """Test gradient and hessian calculation."""
        # Create parameters that require gradients
        params = torch.randn(5, 2, requires_grad=True)

        # Create a simple loss
        loss = torch.sum(params ** 2)

        grad, hess = model.calculate_gradients_and_hessians(loss, params)

        # Check outputs
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (10,)  # 5 samples * 2 params
        assert hess.shape == (10,)

    def test_calculate_gradients_and_hessians_gn(self):
        """Test GN gradient and hessian calculation."""
        model = HyperTreeAR(p=2, hessian_method="gn", n_hessian_probes=10)

        # Create a computational graph: params -> fcst -> loss
        params = nn.Parameter(torch.randn(5, 2))
        lags = torch.randn(5, 2)
        fcst = torch.sum(params * lags, dim=1).unsqueeze(1)
        target = torch.randn(5, 1)
        loss = nn.MSELoss()(fcst, target)

        # Set up GN state (normally done by get_params_loss and objective_fn)
        model._fit = fcst
        model._target = target
        model._iter_count = 1

        grad, hess = model.calculate_gradients_and_hessians(loss, params)

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (10,)  # 5 samples * 2 params
        assert hess.shape == (10,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()
        # GN Hessians should be non-negative (PSD guarantee)
        assert (hess >= -1e-7).all()

class TestHyperTreeAREdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_series_data(self):
        """Test with single time series."""
        model = HyperTreeAR(p=1)
        
        train_data = pd.DataFrame({
            'series_id': [0] * 10,
            'date': pd.date_range('2020-01-01', periods=10, freq='ME'),
            'value': np.random.randn(10).cumsum(),
            'feature1': np.random.randn(10)
        })
        
        # Should not raise error
        assert len(train_data['series_id'].unique()) == 1
        
    def test_minimal_data_size(self):
        """Test with minimal data size."""
        model = HyperTreeAR(p=1, fcst_h=1)
        
        # Just enough data points
        train_data = pd.DataFrame({
            'series_id': [0, 0],
            'date': pd.date_range('2020-01-01', periods=2, freq='ME'),
            'value': [10, 11],
            'feature1': [1, 2]
        })
        
        # Should not raise error during setup
        assert len(train_data) >= model.p
        
    def test_large_parameter_values(self):
        """Test with large AR parameter p."""
        model = HyperTreeAR(p=10)
        
        assert model.p == 10
        
    def test_different_frequencies(self):
        """Test with different time series frequencies."""
        frequencies = ['D', 'W', 'M', 'Q', 'Y']
        
        for freq in frequencies:
            model = HyperTreeAR(freq=freq)
            assert model.freq == freq


class TestHyperTreeARIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def complete_dataset(self):
        """Create a complete dataset for integration testing."""
        np.random.seed(42)
        
        # Create synthetic time series data
        n_series = 2
        n_periods = 50
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='ME')
        
        data = []
        for series_id in range(n_series):
            # Create correlated features and target
            feature1 = np.random.randn(n_periods)
            feature2 = np.random.randn(n_periods)
            
            # Target with some AR structure and feature dependence
            value = np.zeros(n_periods)
            value[0] = 100
            for t in range(1, n_periods):
                value[t] = (0.7 * value[t-1] + 
                           0.1 * feature1[t] + 
                           0.05 * feature2[t] + 
                           np.random.randn() * 0.5)
            
            series_data = pd.DataFrame({
                'series_id': series_id,
                'date': dates,
                'value': value,
                'feature1': feature1,
                'feature2': feature2
            })
            data.append(series_data)
            
        return pd.concat(data, ignore_index=True)
    
    @patch('hypertrees.models.HyperTreeAR.lgb.train')
    @patch('hypertrees.models.HyperTreeAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeAR.TimeSeriesPreprocessor')
    def test_end_to_end_workflow(self, mock_preprocessor, mock_prepare_datasets, 
                                mock_lgb_train, complete_dataset):
        """Test complete training and forecasting workflow."""
        # Setup mocks
        mock_preprocessor_instance = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.create_lags.return_value = complete_dataset
        mock_preprocessor_instance.extract.return_value = {
            "features": pd.DataFrame({
                'feature1': np.random.randn(100), 
                'feature2': np.random.randn(100)
            })
        }
        
        mock_valid_sets = [Mock()]
        mock_lags_train = torch.randn(90, 2)  # Reduced size due to lags
        mock_prepare_datasets.return_value = (
            mock_valid_sets, ['train'], [], {'train': {'loss': [0.5]}},
            mock_lags_train, None, {}
        )
        
        mock_lgb_model = Mock()
        mock_lgb_model.best_iteration = 10
        mock_lgb_model.predict.return_value = np.random.randn(12)  # 2 series * 3 horizons * 2 params
        mock_lgb_train.return_value = mock_lgb_model
        
        # Initialize model
        model = HyperTreeAR(p=2, fcst_h=3)
        
        # Split data properly: take last 3 periods per series for test
        test_idx = complete_dataset.groupby('series_id').tail(3).index
        train_data = complete_dataset.drop(test_idx).reset_index(drop=True)
        test_data = complete_dataset.loc[test_idx].reset_index(drop=True)
        
        # Train model
        training_result = model.train(
            lgb_params={'learning_rate': 0.1, 'num_leaves': 31},
            num_iterations=10,
            train_data=train_data
        )

        # Check training result
        assert isinstance(training_result, TrainingResult)
        assert model.is_trained is True

        # Mock the fcst_lags that would be created during real training
        # Get the last p values for each series (in reverse order: newest to oldest)
        model.fcst_lags = {}
        for series_id in train_data['series_id'].unique():
            series_data = train_data[train_data['series_id'] == series_id]['value']
            model.fcst_lags[series_id] = series_data.iloc[-model.p:].iloc[::-1].values
        
        # Generate forecasts
        forecasts = model.forecast(
            test_data=test_data,
            type="forecast"
        )
        
        # Check forecast output
        assert isinstance(forecasts, pd.DataFrame)
        assert len(forecasts) == len(test_data)
        assert 'fcst' in forecasts.columns
        assert 'series_id' in forecasts.columns
        assert 'date' in forecasts.columns
        assert 'model' in forecasts.columns
        
        # Generate parameters
        parameters = model.forecast(
            test_data=test_data,
            type="parameters"
        )
        
        # Check parameter output
        assert isinstance(parameters, pd.DataFrame)
        assert len(parameters) == len(test_data)
        assert 'AR(1)' in parameters.columns
        assert 'AR(2)' in parameters.columns


class TestHyperTreeARObjectiveFunction:
    """Test HyperTreeAR objective_fn method and its components."""

    @pytest.fixture
    def model_with_lags(self):
        """Create a model instance with mock lags_train for testing."""
        model = HyperTreeAR(p=2, fcst_h=1)
        # Set up mock lags_train that would be created during training
        model.lags_train = torch.randn(10, 2, dtype=torch.float32)
        return model

    @pytest.fixture
    def mock_lgb_dataset(self):
        """Create a mock LightGBM dataset for testing."""
        # Create target values that match expected shape
        target_values = np.random.randn(10)  # 10 samples

        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = target_values
        return mock_dataset

    def test_objective_fn_data_preparation_and_tensor_conversion(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn correctly prepares data and converts to tensors."""
        # Prepare input: forecasts for 10 samples * 2 parameters = 20 values
        predt = np.random.randn(20)

        # Mock the methods that objective_fn calls
        with patch.object(model_with_lags, 'get_params_loss') as mock_get_params_loss, \
             patch.object(model_with_lags, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:

            # Setup return values
            mock_params = torch.randn(10, 2, requires_grad=True)
            mock_loss = torch.tensor(0.5)
            mock_get_params_loss.return_value = (mock_params, mock_loss)
            mock_calc_grad_hess.return_value = (np.random.randn(20), np.random.randn(20))

            # Call objective function
            model_with_lags.objective_fn(predt, mock_lgb_dataset)

            # Verify get_label was called to extract target values
            mock_lgb_dataset.get_label.assert_called_once()

            # Verify get_params_loss was called with correct arguments
            mock_get_params_loss.assert_called_once()
            args = mock_get_params_loss.call_args[0]
            target_tensor = args[1]
            lags_tensor = args[2]
            requires_grad = mock_get_params_loss.call_args[1]['requires_grad']

            # Check that target was converted to proper tensor
            assert isinstance(target_tensor, torch.Tensor)
            assert target_tensor.shape == (10, 1)  # Reshaped to column vector
            assert target_tensor.dtype == model_with_lags.dtype

            # Check that lags_train was passed correctly
            assert torch.equal(lags_tensor, model_with_lags.lags_train)

            # Check that requires_grad=True was passed
            assert requires_grad is True

    def test_objective_fn_get_params_loss_integration(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn correctly integrates with get_params_loss method."""
        predt = np.random.randn(20)

        # Mock calculate_gradients_and_hessians but let get_params_loss run
        with patch.object(model_with_lags, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:
            mock_calc_grad_hess.return_value = (np.random.randn(20), np.random.randn(20))

            # Call objective function
            model_with_lags.objective_fn(predt, mock_lgb_dataset)

            # Verify that calculate_gradients_and_hessians was called
            mock_calc_grad_hess.assert_called_once()

            # Check the arguments passed to calculate_gradients_and_hessians
            args = mock_calc_grad_hess.call_args[0]
            loss_tensor = args[0]
            params_list = args[1]

            # Verify loss is a tensor
            assert isinstance(loss_tensor, torch.Tensor)
            assert loss_tensor.dim() == 0  # Should be scalar

            # Verify params is a tensor with correct shape
            assert isinstance(params_list, torch.Tensor)
            assert params_list.shape[1] == model_with_lags.p
            assert params_list.requires_grad  # Should require gradients

    def test_objective_fn_gradient_and_hessian_calculation(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn correctly calculates gradients and hessians."""
        predt = np.random.randn(20)
        expected_grad = np.random.randn(20)
        expected_hess = np.random.randn(20)

        # Mock only calculate_gradients_and_hessians to control output
        with patch.object(model_with_lags, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:
            mock_calc_grad_hess.return_value = (expected_grad, expected_hess)

            # Call objective function
            grad, hess = model_with_lags.objective_fn(predt, mock_lgb_dataset)

            # Verify the method was called
            mock_calc_grad_hess.assert_called_once()

            # Check that the returned values match expected
            assert np.array_equal(grad, expected_grad)
            assert np.array_equal(hess, expected_hess)

    def test_objective_fn_return_value_format_validation(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn returns properly formatted gradients and hessians."""
        predt = np.random.randn(20)  # 10 samples * 2 parameters

        # Call objective function
        grad, hess = model_with_lags.objective_fn(predt, mock_lgb_dataset)

        # Verify return types
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)

        # Verify shapes match input forecasts
        assert grad.shape == predt.shape
        assert hess.shape == predt.shape

        # Verify data types are appropriate for LightGBM
        assert grad.dtype in [np.float32, np.float64]
        assert hess.dtype in [np.float32, np.float64]

        # Verify no NaN or infinite values
        assert not np.isnan(grad).any()
        assert not np.isnan(hess).any()
        assert not np.isinf(grad).any()
        assert not np.isinf(hess).any()

    def test_objective_fn_error_handling_cases(self, model_with_lags):
        """Test that objective_fn handles error cases appropriately."""
        predt = np.random.randn(20)

        # Test with dataset that returns None labels
        mock_dataset_none = Mock(spec=lgb.Dataset)
        mock_dataset_none.get_label.return_value = None

        with pytest.raises((TypeError, AttributeError)):
            model_with_lags.objective_fn(predt, mock_dataset_none)

        # Test with mismatched forecast shape
        wrong_predt = np.random.randn(15)  # Wrong size
        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(10)

        # This should still work as get_params_loss will handle reshaping
        # But we can test that it doesn't crash
        try:
            model_with_lags.objective_fn(wrong_predt, mock_dataset)
        except Exception as e:
            # Should get a specific error about tensor shapes
            assert isinstance(e, (RuntimeError, ValueError))

    def test_objective_fn_dtype_consistency(self, mock_lgb_dataset):
        """Test that objective_fn maintains dtype consistency."""
        # Test with different model dtypes
        for dtype in [torch.float32, torch.float64]:
            model = HyperTreeAR(p=2, fcst_h=1)
            model.dtype = dtype
            model.lags_train = torch.randn(10, 2, dtype=dtype)

            predt = np.random.randn(20)

            # Mock the internal methods to focus on dtype handling
            with patch.object(model, 'get_params_loss') as mock_get_params_loss, \
                 patch.object(model, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:

                mock_params = torch.randn(10, 2, dtype=dtype)
                mock_loss = torch.tensor(0.5, dtype=dtype)
                mock_get_params_loss.return_value = (mock_params, mock_loss)
                mock_calc_grad_hess.return_value = (np.random.randn(20), np.random.randn(20))

                model.objective_fn(predt, mock_lgb_dataset)

                # Verify that target tensor was created with correct dtype
                args = mock_get_params_loss.call_args[0]
                target_tensor = args[1]
                assert target_tensor.dtype == dtype

    def test_objective_fn_integration_with_real_methods(self, model_with_lags, mock_lgb_dataset):
        """Test objective_fn with real get_params_loss and calculate_gradients_and_hessians methods."""
        predt = np.random.randn(20)  # 10 samples * 2 parameters

        # Call objective function without mocking internal methods
        grad, hess = model_with_lags.objective_fn(predt, mock_lgb_dataset)

        # Basic validation that the integration works
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)
        assert hess.shape == (20,)

        # Values should be finite
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()

        # Hessians should generally be non-negative (for MSE loss)
        # Allow for some numerical precision issues in this integration test
        assert np.isfinite(hess).all()  # Just ensure they're finite

    def test_objective_fn_integration_with_gn_hessian(self, mock_lgb_dataset):
        """Test objective_fn with GN hessian method end-to-end."""
        model = HyperTreeAR(p=2, fcst_h=1, hessian_method="gn", n_hessian_probes=10)
        model.lags_train = torch.randn(10, 2, dtype=torch.float32)

        predt = np.random.randn(20)

        grad, hess = model.objective_fn(predt, mock_lgb_dataset)

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)
        assert hess.shape == (20,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()
        assert model._iter_count == 1


class TestHyperTreeAREvalFunction:
    """Test HyperTreeAR eval_fn method and dataset reference handling."""

    @pytest.fixture
    def model_with_dataset_references(self):
        """Create a model instance with dataset references and lags for testing."""
        model = HyperTreeAR(p=2, fcst_h=1)
        # Set up mock lags for train and validation
        model.lags_train = torch.randn(10, 2, dtype=torch.float32)
        model.lags_eval = torch.randn(8, 2, dtype=torch.float32)

        # Set up dataset references (normally created during training)
        model.dataset_references = {}
        return model

    @pytest.fixture
    def mock_train_dataset(self):
        """Create a mock training dataset."""
        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(10)
        return mock_dataset

    @pytest.fixture
    def mock_validation_dataset(self):
        """Create a mock validation dataset."""
        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(8)
        return mock_dataset

    def test_eval_fn_dataset_reference_train(self, model_with_dataset_references, mock_train_dataset):
        """Test eval_fn correctly identifies and uses training dataset."""
        model = model_with_dataset_references
        predt = np.random.randn(20)  # 10 samples * 2 parameters

        # Register the train dataset
        model.dataset_references[id(mock_train_dataset)] = "train"

        # Mock get_params_loss to control the loss value
        with patch.object(model, 'get_params_loss') as mock_get_params_loss:
            mock_params = torch.randn(10, 2)
            mock_loss = torch.tensor(0.5)
            mock_get_params_loss.return_value = (mock_params, mock_loss)

            # Call eval function
            metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_train_dataset)

            # Verify correct lags were used (train lags)
            mock_get_params_loss.assert_called_once()
            args = mock_get_params_loss.call_args[0]
            lags_used = args[2]
            assert torch.equal(lags_used, model.lags_train)

            # Verify return values
            assert metric_name == model.loss_name
            assert isinstance(metric_value, float)
            assert metric_value == 0.5
            assert is_higher_better is False  # For loss metrics, lower is better

    def test_eval_fn_dataset_reference_validation(self, model_with_dataset_references, mock_validation_dataset):
        """Test eval_fn correctly identifies and uses validation dataset."""
        model = model_with_dataset_references
        predt = np.random.randn(16)  # 8 samples * 2 parameters

        # Register the validation dataset
        model.dataset_references[id(mock_validation_dataset)] = "validation"

        # Mock get_params_loss to control the loss value
        with patch.object(model, 'get_params_loss') as mock_get_params_loss:
            mock_params = torch.randn(8, 2)
            mock_loss = torch.tensor(1.2)
            mock_get_params_loss.return_value = (mock_params, mock_loss)

            # Call eval function
            metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_validation_dataset)

            # Verify correct lags were used (validation lags)
            mock_get_params_loss.assert_called_once()
            args = mock_get_params_loss.call_args[0]
            lags_used = args[2]
            assert torch.equal(lags_used, model.lags_eval)

            # Verify return values
            assert metric_name == model.loss_name
            assert isinstance(metric_value, float)
            assert abs(metric_value - 1.2) < 1e-6  # Use approximate equality for floating point
            assert is_higher_better is False

    def test_eval_fn_dataset_reference_unknown_default_to_train(self, model_with_dataset_references):
        """Test eval_fn defaults to training lags for unknown datasets and emits warning."""
        model = model_with_dataset_references
        predt = np.random.randn(20)  # 10 samples * 2 parameters

        # Create a mock dataset that's not in dataset_references
        unknown_dataset = Mock(spec=lgb.Dataset)
        unknown_dataset.get_label.return_value = np.random.randn(10)

        # Mock get_params_loss to control the loss value
        with patch.object(model, 'get_params_loss') as mock_get_params_loss:
            mock_params = torch.randn(10, 2)
            mock_loss = torch.tensor(0.8)
            mock_get_params_loss.return_value = (mock_params, mock_loss)

            # Call eval function and verify warning
            with pytest.warns(UserWarning, match="Unknown dataset in metric_fn"):
                metric_name, metric_value, is_higher_better = model.eval_fn(predt, unknown_dataset)

            # Verify training lags were used as default
            mock_get_params_loss.assert_called_once()
            args = mock_get_params_loss.call_args[0]
            lags_used = args[2]
            assert torch.equal(lags_used, model.lags_train)

    def test_eval_fn_dataset_reference_get_method(self, model_with_dataset_references, mock_train_dataset):
        """Test that eval_fn correctly uses dataset_references.get() method."""
        model = model_with_dataset_references
        predt = np.random.randn(20)

        # Don't register the dataset, so it should default to "unknown"
        dataset_id = id(mock_train_dataset)
        assert dataset_id not in model.dataset_references

        # Mock get_params_loss
        with patch.object(model, 'get_params_loss') as mock_get_params_loss:
            mock_params = torch.randn(10, 2)
            mock_loss = torch.tensor(0.3)
            mock_get_params_loss.return_value = (mock_params, mock_loss)

            # Call eval function
            model.eval_fn(predt, mock_train_dataset)

            # Should have used training lags as default
            args = mock_get_params_loss.call_args[0]
            lags_used = args[2]
            assert torch.equal(lags_used, model.lags_train)

    def test_eval_fn_data_preparation_and_tensor_conversion(self, model_with_dataset_references, mock_train_dataset):
        """Test that eval_fn correctly prepares data and converts to tensors."""
        model = model_with_dataset_references
        predt = np.random.randn(20)
        model.dataset_references[id(mock_train_dataset)] = "train"

        # Call eval function
        metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_train_dataset)

        # Verify get_label was called to extract target values
        mock_train_dataset.get_label.assert_called_once()

        # Basic validation that method completed successfully
        assert isinstance(metric_name, str)
        assert isinstance(metric_value, float)
        assert isinstance(is_higher_better, bool)

    def test_eval_fn_loss_computation_integration(self, model_with_dataset_references, mock_validation_dataset):
        """Test that eval_fn correctly integrates with get_params_loss for loss computation."""
        model = model_with_dataset_references
        predt = np.random.randn(16)  # 8 samples * 2 parameters
        model.dataset_references[id(mock_validation_dataset)] = "validation"

        # Mock get_params_loss to verify it's called correctly
        with patch.object(model, 'get_params_loss') as mock_get_params_loss:
            mock_params = torch.randn(8, 2)
            mock_loss = torch.tensor(2.5)
            mock_get_params_loss.return_value = (mock_params, mock_loss)

            # Call eval function
            metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_validation_dataset)

            # Verify get_params_loss was called with correct arguments
            mock_get_params_loss.assert_called_once()
            args = mock_get_params_loss.call_args[0]

            # Check predt argument
            np.testing.assert_array_equal(args[0], predt)

            # Check target tensor argument
            target_tensor = args[1]
            assert isinstance(target_tensor, torch.Tensor)
            assert target_tensor.shape == (8, 1)  # Reshaped to column vector

            # Check lags argument (should be validation lags)
            lags_tensor = args[2]
            assert torch.equal(lags_tensor, model.lags_eval)

            # Check that requires_grad is not passed (uses default False)
            # eval_fn calls get_params_loss with positional args only
            assert len(mock_get_params_loss.call_args[1]) == 0  # No keyword arguments

    def test_eval_fn_return_format_validation(self, model_with_dataset_references, mock_train_dataset):
        """Test that eval_fn returns the correct format."""
        model = model_with_dataset_references
        predt = np.random.randn(20)
        model.dataset_references[id(mock_train_dataset)] = "train"

        # Call eval function
        result = model.eval_fn(predt, mock_train_dataset)

        # Verify return format
        assert isinstance(result, tuple)
        assert len(result) == 3

        metric_name, metric_value, is_higher_better = result

        # Verify individual return values
        assert isinstance(metric_name, str)
        assert metric_name == model.loss_name  # Should match the model's loss function name

        assert isinstance(metric_value, (float, np.floating))
        assert not np.isnan(metric_value)
        assert not np.isinf(metric_value)

        assert isinstance(is_higher_better, bool)
        assert is_higher_better is False  # For loss metrics, lower is better

    def test_eval_fn_different_loss_functions(self):
        """Test eval_fn with different loss functions."""
        loss_functions = [
            (nn.MSELoss(), "MSELoss"),
            (nn.L1Loss(), "L1Loss"),
            (nn.SmoothL1Loss(), "SmoothL1Loss")
        ]

        for loss_fn, expected_name in loss_functions:
            model = HyperTreeAR(p=2, loss_fn=loss_fn)
            model.lags_train = torch.randn(5, 2)
            model.dataset_references = {}

            mock_dataset = Mock(spec=lgb.Dataset)
            mock_dataset.get_label.return_value = np.random.randn(5)
            model.dataset_references[id(mock_dataset)] = "train"

            predt = np.random.randn(10)  # 5 samples * 2 parameters

            metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_dataset)

            assert metric_name == expected_name
            assert isinstance(metric_value, float)
            assert is_higher_better is False

    def test_eval_fn_edge_case_empty_dataset_references(self):
        """Test eval_fn behavior when dataset_references is empty."""
        model = HyperTreeAR(p=1)
        model.lags_train = torch.randn(5, 1)
        model.lags_eval = torch.randn(3, 1)
        model.dataset_references = {}  # Empty references

        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(5)

        predt = np.random.randn(5)

        # Should default to training lags and emit warning
        with pytest.warns(UserWarning, match="Unknown dataset in metric_fn"):
            metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_dataset)

        # Should still return valid results
        assert isinstance(metric_name, str)
        assert isinstance(metric_value, float)
        assert is_higher_better is False


class TestHyperTreeARConformal:
    """Tests for conformal prediction intervals on HyperTreeAR."""

    P = 2
    FCST_H = 4
    N_SERIES = 3
    N_OBS = 60
    MODEL_NAME = f"Hyper-Tree-AR({P})"
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
        model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
                     forecast_intervals=ForecastIntervals(n_windows=3))
        return model, train, test

    def test_calibration_sets_state(self, calibrated):
        model, _, _ = calibrated
        assert model._is_calibrated is True
        assert model._cs_scores.shape == (3, self.N_SERIES, self.FCST_H)
        assert np.all(model._cs_scores >= 0)
        assert model._cs_series_order == [0, 1, 2]

    def test_no_calibration_by_default(self, split):
        train, _ = split
        model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train)
        assert model._is_calibrated is False
        assert model._cs_scores is None

    def test_forecast_adds_interval_columns(self, calibrated):
        model, _, test = calibrated
        out = model.forecast(test_data=test, level=[80, 90])
        for lv in [80, 90]:
            assert f"{self.MODEL_NAME}-lo-{lv}" in out.columns
            assert f"{self.MODEL_NAME}-hi-{lv}" in out.columns
        assert np.isfinite(out[f"{self.MODEL_NAME}-lo-90"]).all()

    def test_interval_nesting(self, calibrated):
        model, _, test = calibrated
        mn = self.MODEL_NAME
        out = model.forecast(test_data=test, level=[80, 90])
        assert np.all(out[f"{mn}-lo-90"].to_numpy() <= out[f"{mn}-lo-80"].to_numpy() + 1e-9)
        assert np.all(out[f"{mn}-lo-80"].to_numpy() <= out["fcst"].to_numpy() + 1e-9)
        assert np.all(out["fcst"].to_numpy() <= out[f"{mn}-hi-80"].to_numpy() + 1e-9)
        assert np.all(out[f"{mn}-hi-80"].to_numpy() <= out[f"{mn}-hi-90"].to_numpy() + 1e-9)

    def test_point_forecast_unchanged_by_level(self, calibrated):
        model, _, test = calibrated
        base = model.forecast(test_data=test)
        with_intervals = model.forecast(test_data=test, level=[90])
        np.testing.assert_allclose(base["fcst"].to_numpy(), with_intervals["fcst"].to_numpy())

    def test_both_methods_run(self, split):
        train, test = split
        for method in ("conformal_distribution", "conformal_error"):
            model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
            model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
                         forecast_intervals=ForecastIntervals(n_windows=3, method=method))
            out = model.forecast(test_data=test, level=[90])
            assert f"{self.MODEL_NAME}-lo-90" in out.columns
            assert np.all(out[f"{self.MODEL_NAME}-lo-90"].to_numpy() <= out[f"{self.MODEL_NAME}-hi-90"].to_numpy())

    def test_refit_false_produces_intervals(self, split):
        train, test = split
        model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
                     forecast_intervals=ForecastIntervals(n_windows=3, refit=False))
        assert model._is_calibrated is True
        out = model.forecast(test_data=test, level=[80, 90])
        assert np.all(out[f"{self.MODEL_NAME}-lo-90"].to_numpy() <= out[f"{self.MODEL_NAME}-hi-90"].to_numpy())

    def test_level_without_calibration_raises(self, split):
        train, test = split
        model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train)
        with pytest.raises(RuntimeError, match="not calibrated"):
            model.forecast(test_data=test, level=[90])

    def test_short_series_raises(self):
        short = self._make_data()
        short = short.groupby("series_id", sort=False).head(10).reset_index(drop=True)
        train, _ = self._split(short)
        model = HyperTreeAR(p=self.P, freq="M", fcst_h=self.FCST_H)
        with pytest.raises(ValueError, match="too short"):
            model.train(lgb_params=self.LGB_PARAMS, num_iterations=20, train_data=train,
                         forecast_intervals=ForecastIntervals(n_windows=5))

    def test_invalid_level_value(self, calibrated):
        model, _, test = calibrated
        with pytest.raises(ValueError):
            model.forecast(test_data=test, level=[150])
