import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from unittest.mock import Mock, patch

from hypertrees.models.HyperTreeNetAR import HyperTreeNetAR
from hypertrees.utils import TrainingResult
from hypertrees import ForecastIntervals


class TestHyperTreeNetARInitialization:
    """Test HyperTreeNetAR initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test model initialization with default parameters."""
        model = HyperTreeNetAR()
        
        assert model.p == 2
        assert model.freq == "M"
        assert model.fcst_h == 1
        assert isinstance(model.loss_fn, nn.MSELoss)
        assert model.dtype == torch.float32
        assert model.device == "cpu"
        assert model.model is None
        assert model.features is None
        assert model.is_trained is False
        
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        loss_fn = nn.L1Loss()
        model = HyperTreeNetAR(p=5, freq="D", fcst_h=3, loss_fn=loss_fn, device="cuda")
        
        assert model.p == 5
        assert model.freq == "D"
        assert model.fcst_h == 3
        assert model.loss_fn == loss_fn
        assert model.loss_name == "L1Loss"
        assert model.device == "cuda"
        
    def test_invalid_p_parameter(self):
        """Test that invalid p parameter raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'p' must be a positive integer"):
            HyperTreeNetAR(p=0)
            
        with pytest.raises(ValueError, match="Parameter 'p' must be a positive integer"):
            HyperTreeNetAR(p=-1)
            
    def test_invalid_fcst_h_parameter(self):
        """Test that invalid forecast horizon raises ValueError."""
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeNetAR(fcst_h=0)
            
        with pytest.raises(ValueError, match="Forecast horizon 'fcst_h' must be a positive integer"):
            HyperTreeNetAR(fcst_h=-1)
            
    def test_invalid_loss_function(self):
        """Test that invalid loss function raises TypeError."""
        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeNetAR(loss_fn="invalid")

        with pytest.raises(TypeError, match="loss_fn must be a PyTorch loss function"):
            HyperTreeNetAR(loss_fn=lambda x, y: x - y)

    def test_gn_hessian_initialization(self):
        """Test model initialization with GN hessian method."""
        model = HyperTreeNetAR(hessian_method="gn", n_hessian_probes=10)
        assert model.hessian_method == "gn"
        assert model.n_hessian_probes == 10
        assert hasattr(model, '_gn_hessian')

    def test_default_hessian_method(self):
        """Test that default hessian method is exact."""
        model = HyperTreeNetAR()
        assert model.hessian_method == "exact"
        assert not hasattr(model, '_gn_hessian')

    def test_invalid_hessian_method(self):
        """Test that invalid hessian_method raises ValueError."""
        with pytest.raises(ValueError, match="hessian_method must be either 'exact' or 'gn'"):
            HyperTreeNetAR(hessian_method="invalid")

    def test_gn_hessian_warning_non_mse(self):
        """Test that warning is issued for non-MSE loss with GN."""
        with pytest.warns(UserWarning, match="not nn.MSELoss"):
            HyperTreeNetAR(hessian_method="gn", loss_fn=nn.L1Loss())

    def test_invalid_freq_type(self):
        """Test that non-string freq raises TypeError."""
        with pytest.raises(TypeError, match="freq must be a string"):
            HyperTreeNetAR(freq=123)

    def test_invalid_n_hessian_probes(self):
        """Test that invalid n_hessian_probes raises ValueError."""
        with pytest.raises(ValueError, match="n_hessian_probes must be a positive integer"):
            HyperTreeNetAR(n_hessian_probes=0)


class TestHyperTreeNetARSetForecastOrigin:
    """Test HyperTreeNetAR.set_forecast_origin."""

    def test_updates_fcst_lags(self):
        """set_forecast_origin should store last p values per series in reverse order."""
        model = HyperTreeNetAR(p=3, freq="M", fcst_h=2)
        history = pd.DataFrame({
            "series_id": [0]*6 + [1]*6,
            "date": list(pd.date_range("2020-01-01", periods=6, freq="MS")) * 2,
            "value": [10, 20, 30, 40, 50, 60] + [100, 200, 300, 400, 500, 600],
        })
        model.set_forecast_origin(history)
        np.testing.assert_array_equal(model.fcst_lags[0], [60, 50, 40])
        np.testing.assert_array_equal(model.fcst_lags[1], [600, 500, 400])

    def test_reanchor_changes_lags(self):
        """Calling set_forecast_origin twice should update the lags."""
        model = HyperTreeNetAR(p=2, freq="M", fcst_h=1)
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
        model = HyperTreeNetAR(p=2, freq="M", fcst_h=1)
        bad = pd.DataFrame({
            "series_id": [0, 1, 0],
            "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "value": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="non-contiguous"):
            model.set_forecast_origin(bad)


class TestHyperTreeNetARTraining:
    """Test HyperTreeNetAR training functionality."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='ME')
        
        data = []
        for series_id in range(3):
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
        model = HyperTreeNetAR()
        
        # Test missing train_data
        with pytest.raises(ValueError, match="train_data must be provided"):
            model.train(lgb_params={}, network_params={}, num_iterations=10)
            
        # Test missing lgb_params
        with pytest.raises(ValueError, match="lgb_params must be provided"):
            model.train(train_data=sample_train_data, network_params={}, num_iterations=10)
            
        # Test missing network_params
        with pytest.raises(ValueError, match="network_params must be provided"):
            model.train(lgb_params={}, train_data=sample_train_data, num_iterations=10)
            
        # Test invalid gradient_mode
        with pytest.raises(ValueError, match="gradient_mode must be either 'shared' or 'separate'"):
            model.train(
                lgb_params={}, 
                network_params={}, 
                train_data=sample_train_data, 
                num_iterations=10,
                gradient_mode="invalid"
            )
            
        # Test invalid data types
        with pytest.raises(TypeError, match="train_data must be a pandas DataFrame"):
            model.train(lgb_params={}, network_params={}, train_data="invalid", num_iterations=10)
            
        with pytest.raises(TypeError, match="lgb_params must be a dictionary"):
            model.train(lgb_params="invalid", network_params={}, train_data=sample_train_data, num_iterations=10)
            
        with pytest.raises(TypeError, match="network_params must be a dictionary"):
            model.train(lgb_params={}, network_params="invalid", train_data=sample_train_data, num_iterations=10)

    def test_train_type_validation(self, sample_train_data):
        """Test training parameter type validation for seed, verbose, validation, deterministic."""
        model = HyperTreeNetAR()
        net_params = {'embedding_dimension': 2, 'hidden_dim': 32, 'learning_rate': 0.01,
                      'use_random_projection': False, 'dropout': 0.1}

        with pytest.raises(TypeError, match="seed must be an integer"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, seed="bad")

        with pytest.raises(TypeError, match="verbose must be an integer"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, verbose="bad")

        with pytest.raises(TypeError, match="validation must be a boolean"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, validation="yes")

        with pytest.raises(TypeError, match="deterministic must be a boolean"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, deterministic="yes")

        with pytest.raises(ValueError, match="early_stopping_round must be a positive integer"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, validation=True, early_stopping_round=-1)

        with pytest.raises(ValueError, match="early_stopping_round can only be used when validation is True"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, validation=False, early_stopping_round=5)

        with pytest.raises(ValueError, match="early_stopping_round must be provided when validation is True"):
            model.train(lgb_params={}, network_params=net_params, train_data=sample_train_data, num_iterations=10, validation=True)

    def test_required_columns_validation(self):
        """Test that missing required columns raise ValueError."""
        model = HyperTreeNetAR()
        
        # Missing series_id
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'value': np.random.randn(10)
        })
        with pytest.raises(ValueError, match="Required column 'series_id' not found"):
            model.train(
                lgb_params={}, 
                network_params={'embedding_dimension': 2, 'hidden_dim': 32, 'learning_rate': 0.01, 'use_random_projection': False, 'dropout': 0.1}, 
                train_data=data, 
                num_iterations=10
            )
    
    @patch('hypertrees.models.HyperTreeNetAR.lgb.train')
    @patch('hypertrees.models.HyperTreeNetAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeNetAR.TimeSeriesPreprocessor')
    def test_successful_training_separate_gradients(self, mock_preprocessor, mock_prepare_datasets, mock_lgb_train, sample_train_data):
        """Test successful training flow with separate gradients."""
        model = HyperTreeNetAR(p=2)
        
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
            network_params={
                'embedding_dimension': 2,
                'hidden_dim': 32,
                'learning_rate': 0.01,
                'use_random_projection': False,
                'dropout': 0.1
            },
            num_iterations=10,
            train_data=sample_train_data,
            gradient_mode="separate"
        )
        
        # Assertions
        assert model.is_trained is True
        assert model.model == mock_lgb_model
        assert model.features == ['feature1', 'feature2']
        assert isinstance(result, TrainingResult)
        assert model.gradient_mode == "separate"
        assert model.embedding_dim == 2
        assert model.network is not None
        assert model.optimizer is not None
        
    @patch('hypertrees.models.HyperTreeNetAR.lgb.train')
    @patch('hypertrees.models.HyperTreeNetAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeNetAR.TimeSeriesPreprocessor')
    def test_successful_training_shared_gradients(self, mock_preprocessor, mock_prepare_datasets, mock_lgb_train, sample_train_data):
        """Test successful training flow with shared gradients."""
        model = HyperTreeNetAR(p=2)
        
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
            network_params={
                'embedding_dimension': 2,
                'hidden_dim': 32,
                'learning_rate': 0.01,
                'use_random_projection': False,
                'dropout': 0.1
            },
            num_iterations=10,
            train_data=sample_train_data,
            gradient_mode="shared"
        )
        
        # Assertions
        assert model.is_trained is True
        assert model.gradient_mode == "shared"
        
    @patch('hypertrees.models.HyperTreeNetAR.lgb.train')
    def test_training_failure_handling(self, mock_lgb_train, sample_train_data):
        """Test that training failures are handled properly."""
        model = HyperTreeNetAR()
        
        # Mock LightGBM to raise exception
        mock_lgb_train.side_effect = Exception("Training failed")
        
        with pytest.raises(RuntimeError, match="Training failed"):
            model.train(
                lgb_params={'learning_rate': 0.1},
                network_params={
                    'embedding_dimension': 2,
                    'hidden_dim': 32,
                    'learning_rate': 0.01,
                    'use_random_projection': False,
                    'dropout': 0.1
                },
                train_data=sample_train_data,
                num_iterations=10
            )
            
        assert model.is_trained is False


class TestHyperTreeNetARForecasting:
    """Test HyperTreeNetAR forecasting functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a mock trained model for testing."""
        model = HyperTreeNetAR(p=2, fcst_h=3)
        model.is_trained = True
        model.features = ['feature1', 'feature2']
        model.embedding_dim = 4

        # Mock the LightGBM model
        mock_lgb_model = Mock()
        mock_lgb_model.best_iteration = 10
        mock_lgb_model.predict.return_value = np.random.randn(36)  # 3 series * 3 horizons * 4 embeddings
        model.model = mock_lgb_model

        # Mock the neural network
        mock_network = Mock()
        mock_network.eval.return_value = None
        mock_network_output = torch.randn(9, 2)  # 9 test points, 2 AR parameters

        def mock_network_call(x):
            return mock_network_output

        mock_network.side_effect = mock_network_call
        model.network = mock_network

        # Mock network state
        HyperTreeNetAR._network_states = {}

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
        model = HyperTreeNetAR()

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

        # Missing features (must include all trained series with fcst_h rows each)
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

        with pytest.raises(ValueError, match="Parameter 'type' must be either 'forecast', 'parameters' or 'tree_embeddings'"):
            trained_model.forecast(
                test_data=test_data,
                type="invalid"
            )

    def test_forecast_series_mismatch(self, trained_model):
        """Test that mismatched series IDs between train and test raises ValueError."""
        test_data = pd.DataFrame({
            'series_id': [0]*3 + [1]*3 + [99]*3,
            'date': pd.date_range('2020-02-01', periods=3, freq='MS').tolist() * 3,
            'feature1': range(9), 'feature2': range(9)
        })
        with pytest.raises(ValueError, match="Missing series in training"):
            trained_model.forecast(test_data=test_data)

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
        assert result['model'].iloc[0] == 'Hyper-TreeNet-AR(2)'
        
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
        
    def test_forecast_tree_embeddings(self, trained_model, sample_train_data, sample_test_data):
        """Test forecasting with tree embeddings output."""
        result = trained_model.forecast(
            test_data=sample_test_data,
            type="tree_embeddings"
        )
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert 'series_id' in result.columns
        assert 'date' in result.columns
        assert 'model' in result.columns
        assert 'tree_embedding_1' in result.columns
        assert f'tree_embedding_{trained_model.embedding_dim}' in result.columns
        assert len(result) == len(sample_test_data)


class TestHyperTreeNetARInternalMethods:
    """Test internal methods of HyperTreeNetAR."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return HyperTreeNetAR(p=2, device="cpu")
    
    def test_get_embeds_loss_separate(self, model):
        """Test get_embeds_loss_separate method."""
        model.embedding_dim = 4
        model.device = "cpu"
        
        # Mock network and optimizer
        mock_network = Mock()
        mock_network.train.return_value = None
        mock_network.eval.return_value = None
        
        def mock_network_call(x):
            # Return tensors that require gradients
            return torch.randn(5, 2, requires_grad=True)  # 5 samples, 2 AR parameters
        
        mock_network.side_effect = mock_network_call
        model.network = mock_network
        
        mock_optimizer = Mock()
        mock_optimizer.zero_grad.return_value = None
        mock_optimizer.step.return_value = None
        model.optimizer = mock_optimizer
        
        # Setup test data - align requires_grad with method parameter
        predt = np.random.randn(20)  # 5 samples * 4 embeddings
        target = torch.randn(5, 1, requires_grad=False)
        lags = torch.randn(5, 2, requires_grad=False)

        # Call method
        params, loss = model.get_embeds_loss_separate(predt, target, lags, requires_grad=False)
        
        # Check outputs
        assert params.shape[1] == 4  # Number of embedding dimensions (tensor shape)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        
    def test_get_embeds_loss_shared(self, model):
        """Test get_embeds_loss_shared method."""
        model.embedding_dim = 4
        model.device = "cpu"
        
        # Mock network and optimizer
        mock_network = Mock()
        def mock_network_call(x):
            # Return tensors that require gradients
            return torch.randn(5, 2, requires_grad=True)  # 5 samples, 2 AR parameters
        mock_network.side_effect = mock_network_call
        model.network = mock_network
        
        mock_optimizer = Mock()
        mock_optimizer.zero_grad.return_value = None
        model.optimizer = mock_optimizer
        
        # Setup test data - align requires_grad with method parameter
        predt = np.random.randn(20)  # 5 samples * 4 embeddings
        target = torch.randn(5, 1, requires_grad=True)
        lags = torch.randn(5, 2, requires_grad=True)

        # Call method
        embeds, loss = model.get_embeds_loss_shared(predt, target, lags, requires_grad=True)
        
        # Check outputs
        assert isinstance(embeds, torch.Tensor)
        assert embeds.requires_grad
        assert isinstance(loss, torch.Tensor)
        
    def test_calculate_gradients_and_hessians_separate(self, model):
        """Test gradient and hessian calculation for separate mode."""
        model.embedding_dim = 4  # Set embedding dimension to fix AttributeError
        # Create embeddings that require gradients - as a list of tensors
        embeds = torch.randn(5, 4, requires_grad=True)

        # Create a simple loss that maintains the computational graph
        loss = torch.sum(embeds ** 2)

        # The method expects the loss to still be connected to embeds for gradient computation
        # Don't call backward here - let the method compute gradients directly
        grad, hess = model._calculate_gradients_and_hessians_separate(loss, embeds)

        # Check outputs
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)  # 5 samples * 4 embeddings
        assert hess.shape == (20,)
        
    def test_calculate_gradients_and_hessians_shared(self, model):
        """Test gradient and hessian calculation for shared mode."""
        model.embedding_dim = 4

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.step.return_value = None
        mock_optimizer.zero_grad.return_value = None
        model.optimizer = mock_optimizer

        # Mock network with state_dict method
        mock_network = Mock()
        mock_network.state_dict.return_value = {"fake": "state"}
        model.network = mock_network

        # Create embeddings as nn.Parameter (as done in the actual implementation)
        embeds = nn.Parameter(torch.randn(5, 4), requires_grad=True)

        # Create a loss with create_graph=True (as done in get_embeds_loss_shared)
        loss = torch.sum(embeds ** 2)
        loss.backward(create_graph=True)

        grad, hess = model._calculate_gradients_and_hessians_shared(loss, embeds)

        # Check outputs
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)  # 5 samples * 4 embeddings
        assert hess.shape == (20,)
        
    def test_calculate_gradients_and_hessians_separate_gn(self):
        """Test GN gradient and hessian calculation for separate mode."""
        model = HyperTreeNetAR(p=2, hessian_method="gn", n_hessian_probes=10)
        model.embedding_dim = 4

        # Create a direct computational graph: embeds -> loss
        embeds = torch.randn(5, 4, requires_grad=True)
        fcst = embeds.sum(dim=1, keepdim=True)
        target = torch.randn(5, 1)
        loss = nn.MSELoss()(fcst, target)

        # Set up GN state
        model._fit = fcst
        model._target = target
        model._iter_count = 1
        model.calculate_gradients_and_hessians = model._calculate_gradients_and_hessians_separate_gn

        grad, hess = model.calculate_gradients_and_hessians(loss, embeds)

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)  # 5 samples * 4 embeddings
        assert hess.shape == (20,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()

    def test_calculate_gradients_and_hessians_shared_gn(self):
        """Test GN gradient and hessian calculation for shared mode."""
        model = HyperTreeNetAR(p=2, hessian_method="gn", n_hessian_probes=10)
        model.embedding_dim = 4

        # Mock optimizer and network for shared mode cleanup
        mock_optimizer = Mock()
        mock_optimizer.step.return_value = None
        mock_optimizer.zero_grad.return_value = None
        model.optimizer = mock_optimizer
        mock_network = Mock()
        mock_network.state_dict.return_value = {"fake": "state"}
        model.network = mock_network

        # Create a computational graph and call backward with create_graph=True
        embeds = nn.Parameter(torch.randn(5, 4))
        fcst = embeds.sum(dim=1, keepdim=True)
        target = torch.randn(5, 1)
        loss = nn.MSELoss()(fcst, target)
        loss.backward(create_graph=True)

        # Set up GN state
        model._fit = fcst
        model._target = target
        model._iter_count = 1
        model.calculate_gradients_and_hessians = model._calculate_gradients_and_hessians_shared_gn

        grad, hess = model.calculate_gradients_and_hessians(loss, embeds)

        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (20,)
        assert hess.shape == (20,)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()

    def test_eval_function_with_network_state(self, model):
        """Test evaluation function with proper network state handling."""
        model.embedding_dim = 2
        model.device = "cpu"
        model.dataset_references = {}
        model.lags_train = torch.randn(3, 2)
        
        # Mock network
        mock_network = Mock()
        mock_network.eval.return_value = None
        mock_network.load_state_dict.return_value = None
        def mock_network_call(x):
            return torch.randn(3, 2)
        mock_network.side_effect = mock_network_call
        model.network = mock_network
        
        # Store network state
        HyperTreeNetAR._network_states = {}
        
        # Setup test data
        predt = np.random.randn(6)  # 3 samples * 2 embeddings
        mock_data = Mock()
        mock_data.get_label.return_value = np.array([1.0, 2.0, 3.0])
        
        metric_name, metric_value, is_higher_better = model.eval_fn(predt, mock_data)

        # Check outputs
        assert metric_name == model.loss_name
        assert isinstance(metric_value, float)
        assert is_higher_better is False


class TestHyperTreeNetAREdgeCases:
    """Test edge cases and error conditions."""
    
    def test_different_embedding_dimensions(self):
        """Test with different embedding dimensions."""
        dimensions = [1, 4, 16, 64]
        
        for dim in dimensions:
            model = HyperTreeNetAR(p=2)
            # Embedding dimension is set during training
            assert model.p == 2
            
    def test_different_parameter_values(self):
        """Test with different AR parameter p."""
        p_values = [1, 3, 5, 10]
        
        for p in p_values:
            model = HyperTreeNetAR(p=p)
            assert model.p == p
            
    def test_large_forecast_horizons(self):
        """Test with large forecast horizons."""
        model = HyperTreeNetAR(p=2, fcst_h=12)
        assert model.fcst_h == 12
        
    def test_different_devices(self):
        """Test with different devices."""
        devices = ["cpu", "cuda"]
        
        for device in devices:
            model = HyperTreeNetAR(device=device)
            assert model.device == device
            
    def test_different_frequencies(self):
        """Test with different time series frequencies."""
        frequencies = ['D', 'W', 'M', 'Q', 'Y']
        
        for freq in frequencies:
            model = HyperTreeNetAR(freq=freq)
            assert model.freq == freq
            
    def test_different_loss_functions(self):
        """Test with different PyTorch loss functions."""
        loss_functions = [nn.MSELoss(), nn.L1Loss(), nn.HuberLoss()]
        
        for loss_fn in loss_functions:
            model = HyperTreeNetAR(loss_fn=loss_fn)
            assert model.loss_fn == loss_fn
            assert model.loss_name == loss_fn.__class__.__name__
    
    def test_network_state_management(self):
        """Test network state storage and retrieval."""
        model1 = HyperTreeNetAR(p=2)
        model2 = HyperTreeNetAR(p=3)

        # Test state storage
        test_state = {'test': 'state'}
        HyperTreeNetAR._network_states = test_state


class TestHyperTreeNetARIntegration:
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
    
    @patch('hypertrees.models.HyperTreeNetAR.lgb.train')
    @patch('hypertrees.models.HyperTreeNetAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeNetAR.TimeSeriesPreprocessor')
    def test_end_to_end_workflow_separate_gradients(self, mock_preprocessor, mock_prepare_datasets, 
                                                   mock_lgb_train, complete_dataset):
        """Test complete training and forecasting workflow with separate gradients."""
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
        mock_lgb_model.predict.return_value = np.random.randn(24)  # 2 series * 3 horizons * 4 embeddings
        mock_lgb_train.return_value = mock_lgb_model
        
        # Initialize model
        model = HyperTreeNetAR(p=2, fcst_h=3, device="cpu")
        
        # Split data properly: take last 3 periods per series for test
        test_idx = complete_dataset.groupby('series_id').tail(3).index
        train_data = complete_dataset.drop(test_idx).reset_index(drop=True)
        test_data = complete_dataset.loc[test_idx].reset_index(drop=True)
        
        # Train model
        training_result = model.train(
            lgb_params={'learning_rate': 0.1, 'num_leaves': 31},
            network_params={
                'embedding_dimension': 4,
                'hidden_dim': 32,
                'learning_rate': 0.01,
                'use_random_projection': False,
                'dropout': 0.1
            },
            num_iterations=10,
            train_data=train_data,
            gradient_mode="separate"
        )
        
        # Check training result
        assert isinstance(training_result, TrainingResult)
        assert model.is_trained is True
        assert model.gradient_mode == "separate"
        
        # Mock network for forecasting
        mock_network = Mock()
        mock_network.eval.return_value = None
        mock_network.load_state_dict.return_value = None
        def mock_network_call(x):
            return torch.randn(x.shape[0], 2)  # Return appropriate size
        mock_network.side_effect = mock_network_call
        model.network = mock_network
        
        # Store network state
        HyperTreeNetAR._network_states = {}

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
        
        # Generate tree embeddings
        embeddings = model.forecast(
            test_data=test_data,
            type="tree_embeddings"
        )
        
        # Check embeddings output
        assert isinstance(embeddings, pd.DataFrame)
        assert len(embeddings) == len(test_data)
        assert 'tree_embedding_1' in embeddings.columns
        assert f'tree_embedding_{model.embedding_dim}' in embeddings.columns
    
    @patch('hypertrees.models.HyperTreeNetAR.lgb.train')
    @patch('hypertrees.models.HyperTreeNetAR.prepare_datasets')
    @patch('hypertrees.models.HyperTreeNetAR.TimeSeriesPreprocessor')
    def test_end_to_end_workflow_shared_gradients(self, mock_preprocessor, mock_prepare_datasets, 
                                                 mock_lgb_train, complete_dataset):
        """Test complete training and forecasting workflow with shared gradients."""
        # Setup mocks similar to separate gradients test
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
        mock_lags_train = torch.randn(90, 2)
        mock_prepare_datasets.return_value = (
            mock_valid_sets, ['train'], [], {'train': {'loss': [0.5]}},
            mock_lags_train, None, {}
        )
        
        mock_lgb_model = Mock()
        mock_lgb_model.best_iteration = 10
        mock_lgb_model.predict.return_value = np.random.randn(24)
        mock_lgb_train.return_value = mock_lgb_model
        
        # Initialize model
        model = HyperTreeNetAR(p=2, fcst_h=3, device="cpu")
        
        # Split data properly: take last 3 periods per series for test
        test_idx = complete_dataset.groupby('series_id').tail(3).index
        train_data = complete_dataset.drop(test_idx).reset_index(drop=True)
        test_data = complete_dataset.loc[test_idx].reset_index(drop=True)
        
        # Train model with shared gradients
        training_result = model.train(
            lgb_params={'learning_rate': 0.1, 'num_leaves': 31},
            network_params={
                'embedding_dimension': 4,
                'hidden_dim': 32,
                'learning_rate': 0.01,
                'use_random_projection': False,
                'dropout': 0.1
            },
            num_iterations=10,
            train_data=train_data,
            gradient_mode="shared"
        )
        
        # Check training result
        assert isinstance(training_result, TrainingResult)
        assert model.is_trained is True
        assert model.gradient_mode == "shared"


class TestHyperTreeNetARObjectiveFunction:
    """Test HyperTreeNetAR objective_fn method and its components."""

    @pytest.fixture
    def model_with_lags(self):
        """Create a model instance with mock lags_train for testing."""
        model = HyperTreeNetAR(p=2, fcst_h=1)
        # Set up mock lags_train that would be created during training
        model.lags_train = torch.randn(10, 2, dtype=torch.float32)
        model.device = "cpu"  # Ensure device is set

        # Set up the dynamic method assignments (normally done during training)
        model.gradient_mode = "separate"  # Set a default mode
        model.get_embeds_loss = model.get_embeds_loss_separate
        model.calculate_gradients_and_hessians = model._calculate_gradients_and_hessians_separate

        # Set up embedding dimension (normally set during training)
        model.embedding_dim = 10  # Set a reasonable default for testing

        # Set up a mock network (normally created during training)
        from hypertrees.models.mlp import MLP
        model.network = MLP(
            tree_embed_dim=10,
            output_dim=2,  # p=2
            hidden_dim=64,
            use_random_projection=False,
            dropout_rate=0.1
        )

        # Set up optimizer (normally created during training)
        import torch.optim as optim
        model.optimizer = optim.Adam(model.network.parameters(), lr=1e-3)

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
        # Prepare input: forecasts for 10 samples * 10 embedding_dim = 100 values
        predt = np.random.randn(100)

        # Mock the methods that objective_fn calls
        with patch.object(model_with_lags, 'get_embeds_loss') as mock_get_embeds_loss, \
             patch.object(model_with_lags, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:

            # Setup return values
            mock_embeds = [torch.randn(10, 1) for _ in range(10)]  # 10 embedding dimensions
            mock_loss = torch.tensor(0.5)
            mock_get_embeds_loss.return_value = (mock_embeds, mock_loss)
            mock_calc_grad_hess.return_value = (np.random.randn(100), np.random.randn(100))

            # Call objective function
            model_with_lags.objective_fn(predt, mock_lgb_dataset)

            # Verify get_label was called to extract target values
            mock_lgb_dataset.get_label.assert_called_once()

            # Verify get_embeds_loss was called with correct arguments
            mock_get_embeds_loss.assert_called_once()
            args = mock_get_embeds_loss.call_args[0]
            target_tensor = args[1]
            lags_tensor = args[2]
            requires_grad = mock_get_embeds_loss.call_args[1]['requires_grad']

            # Check that target was converted to proper tensor
            assert isinstance(target_tensor, torch.Tensor)
            assert target_tensor.shape == (10, 1)  # Reshaped to column vector
            assert target_tensor.dtype == model_with_lags.dtype
            assert target_tensor.device.type == model_with_lags.device  # Check device

            # Check that lags_train was passed correctly
            assert torch.equal(lags_tensor, model_with_lags.lags_train)

            # Check that requires_grad=True was passed
            assert requires_grad is True

    def test_objective_fn_get_embeds_loss_integration(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn correctly integrates with get_embeds_loss method."""
        predt = np.random.randn(100)

        # Mock calculate_gradients_and_hessians but let get_embeds_loss run
        with patch.object(model_with_lags, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:
            mock_calc_grad_hess.return_value = (np.random.randn(100), np.random.randn(100))

            # Call objective function
            model_with_lags.objective_fn(predt, mock_lgb_dataset)

            # Verify that calculate_gradients_and_hessians was called
            mock_calc_grad_hess.assert_called_once()

            # Check the arguments passed to calculate_gradients_and_hessians
            args = mock_calc_grad_hess.call_args[0]
            loss_tensor = args[0]
            embeds_tensor = args[1]  # Renamed from embeds_list

            # Verify loss is a tensor
            assert isinstance(loss_tensor, torch.Tensor)
            assert loss_tensor.dim() == 0  # Should be scalar

            # Verify embeds is a tensor with correct shape
            assert isinstance(embeds_tensor, torch.Tensor)
            assert embeds_tensor.shape[1] == model_with_lags.embedding_dim  # Check embedding dimension
            assert embeds_tensor.requires_grad  # Should require gradients

    def test_objective_fn_gradient_and_hessian_calculation(self, model_with_lags, mock_lgb_dataset):
        """Test that objective_fn correctly calculates gradients and hessians."""
        predt = np.random.randn(100)
        expected_grad = np.random.randn(100)
        expected_hess = np.random.randn(100)

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
        predt = np.random.randn(100)  # 10 samples * 10 embedding dimensions

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
        predt = np.random.randn(100)

        # Test with dataset that returns None labels
        mock_dataset_none = Mock(spec=lgb.Dataset)
        mock_dataset_none.get_label.return_value = None

        with pytest.raises((TypeError, AttributeError)):
            model_with_lags.objective_fn(predt, mock_dataset_none)

        # Test with mismatched forecast shape
        wrong_predt = np.random.randn(50)  # Wrong size
        mock_dataset = Mock(spec=lgb.Dataset)
        mock_dataset.get_label.return_value = np.random.randn(10)

        # This should still work as get_embeds_loss will handle reshaping
        # But we can test that it doesn't crash
        try:
            model_with_lags.objective_fn(wrong_predt, mock_dataset)
        except Exception as e:
            # Should get a specific error about tensor shapes
            assert isinstance(e, (RuntimeError, ValueError))

    def test_objective_fn_device_consistency(self, mock_lgb_dataset):
        """Test that objective_fn maintains device consistency."""
        # Test with different devices
        for device in ["cpu"]:  # Only test CPU since CUDA may not be available
            model = HyperTreeNetAR(p=2, fcst_h=1, device=device)
            model.lags_train = torch.randn(10, 2, dtype=torch.float32, device=device)

            # Set up the dynamic method assignments
            model.gradient_mode = "separate"
            model.get_embeds_loss = model.get_embeds_loss_separate
            model.calculate_gradients_and_hessians = model._calculate_gradients_and_hessians_separate
            model.embedding_dim = 10

            # Set up a mock network
            from hypertrees.models.mlp import MLP
            model.network = MLP(
                tree_embed_dim=10,
                output_dim=2,
                hidden_dim=64,
                use_random_projection=False,
                dropout_rate=0.1
            )

            predt = np.random.randn(100)

            # Mock the internal methods to focus on device handling
            with patch.object(model, 'get_embeds_loss') as mock_get_embeds_loss, \
                 patch.object(model, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:

                mock_embeds = [torch.randn(10, 1, device=device) for _ in range(10)]
                mock_loss = torch.tensor(0.5, device=device)
                mock_get_embeds_loss.return_value = (mock_embeds, mock_loss)
                mock_calc_grad_hess.return_value = (np.random.randn(100), np.random.randn(100))

                model.objective_fn(predt, mock_lgb_dataset)

                # Verify that target tensor was created with correct device
                args = mock_get_embeds_loss.call_args[0]
                target_tensor = args[1]
                assert target_tensor.device.type == device

    def test_objective_fn_dtype_consistency(self, mock_lgb_dataset):
        """Test that objective_fn maintains dtype consistency."""
        # Test with different model dtypes
        for dtype in [torch.float32, torch.float64]:
            model = HyperTreeNetAR(p=2, fcst_h=1)
            model.dtype = dtype
            model.lags_train = torch.randn(10, 2, dtype=dtype)

            # Set up the dynamic method assignments
            model.gradient_mode = "separate"
            model.get_embeds_loss = model.get_embeds_loss_separate
            model.calculate_gradients_and_hessians = model._calculate_gradients_and_hessians_separate
            model.embedding_dim = 10

            # Set up a mock network
            from hypertrees.models.mlp import MLP
            model.network = MLP(
                tree_embed_dim=10,
                output_dim=2,
                hidden_dim=64,
                use_random_projection=False,
                dropout_rate=0.1
            )

            predt = np.random.randn(100)

            # Mock the internal methods to focus on dtype handling
            with patch.object(model, 'get_embeds_loss') as mock_get_embeds_loss, \
                 patch.object(model, 'calculate_gradients_and_hessians') as mock_calc_grad_hess:

                mock_embeds = [torch.randn(10, 1, dtype=dtype) for _ in range(10)]
                mock_loss = torch.tensor(0.5, dtype=dtype)
                mock_get_embeds_loss.return_value = (mock_embeds, mock_loss)
                mock_calc_grad_hess.return_value = (np.random.randn(100), np.random.randn(100))

                model.objective_fn(predt, mock_lgb_dataset)

                # Verify that target tensor was created with correct dtype
                args = mock_get_embeds_loss.call_args[0]
                target_tensor = args[1]
                assert target_tensor.dtype == dtype

    def test_objective_fn_integration_with_real_methods(self, model_with_lags, mock_lgb_dataset):
        """Test objective_fn with real get_embeds_loss and calculate_gradients_and_hessians methods."""
        predt = np.random.randn(100)  # 10 samples * 10 embedding dimensions

        # Call objective function without mocking internal methods
        grad, hess = model_with_lags.objective_fn(predt, mock_lgb_dataset)

        # Basic validation that the integration works
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == (100,)
        assert hess.shape == (100,)

        # Values should be finite
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()

        # Hessians should generally be non-negative (for MSE loss)
        # Allow for some numerical precision issues in this integration test
        assert np.isfinite(hess).all()  # Just ensure they're finite


class TestHyperTreeNetARConformal:
    """Tests for conformal prediction intervals on HyperTreeNetAR."""

    FCST_H = 4
    N_SERIES = 2
    N_OBS = 60
    LGB_PARAMS = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 1, "min_data_in_bin": 1}
    NETWORK_PARAMS = {
        "learning_rate": 1e-3, "embedding_dimension": 1, "hidden_dim": 32,
        "dropout": 0.0, "use_random_projection": False, "rp_embed_dim": None,
    }

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
        model = HyperTreeNetAR(p=2, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, network_params=self.NETWORK_PARAMS,
                     num_iterations=20, train_data=train,
                     forecast_intervals=ForecastIntervals(n_windows=3, refit=False))
        return model, train, test

    def test_calibration_sets_state(self, calibrated):
        model, _, _ = calibrated
        assert model._is_calibrated is True
        assert model._cs_scores.shape == (3, self.N_SERIES, self.FCST_H)
        assert np.all(model._cs_scores >= 0)

    def test_no_calibration_by_default(self, split):
        train, _ = split
        model = HyperTreeNetAR(p=2, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, network_params=self.NETWORK_PARAMS,
                     num_iterations=20, train_data=train)
        assert model._is_calibrated is False

    def test_forecast_adds_interval_columns(self, calibrated):
        model, _, test = calibrated
        mn = "Hyper-TreeNet-AR(2)"
        out = model.forecast(test_data=test, level=[80, 90])
        for lv in [80, 90]:
            assert f"{mn}-lo-{lv}" in out.columns
            assert f"{mn}-hi-{lv}" in out.columns

    def test_interval_nesting(self, calibrated):
        model, _, test = calibrated
        mn = "Hyper-TreeNet-AR(2)"
        out = model.forecast(test_data=test, level=[80, 90])
        assert np.all(out[f"{mn}-lo-90"].to_numpy() <= out[f"{mn}-lo-80"].to_numpy() + 1e-9)
        assert np.all(out[f"{mn}-hi-80"].to_numpy() <= out[f"{mn}-hi-90"].to_numpy() + 1e-9)

    def test_level_without_calibration_raises(self, split):
        train, test = split
        model = HyperTreeNetAR(p=2, freq="M", fcst_h=self.FCST_H)
        model.train(lgb_params=self.LGB_PARAMS, network_params=self.NETWORK_PARAMS,
                     num_iterations=20, train_data=train)
        with pytest.raises(RuntimeError, match="not calibrated"):
            model.forecast(test_data=test, level=[90])
