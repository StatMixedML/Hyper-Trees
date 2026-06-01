import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch
from hypertrees.utils import TrainingResult, CustomLogger, TimeSeriesPreprocessor, prepare_datasets


class TestTrainingResult:
    """Test the TrainingResult dataclass."""

    def test_default_initialization(self):
        """Test TrainingResult with only required fields."""
        train_metrics = {"loss": [0.5, 0.3, 0.2]}
        result = TrainingResult(train_metrics=train_metrics)

        assert result.train_metrics == train_metrics
        assert result.validation_metrics is None
        assert result.best_iteration is None
        assert result.training_time is None

    def test_full_initialization(self):
        """Test TrainingResult with all fields."""
        train_metrics = {"loss": [0.5, 0.3, 0.2]}
        val_metrics = {"val_loss": [0.6, 0.4, 0.3]}

        result = TrainingResult(
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            best_iteration=2,
            training_time=120.5
        )

        assert result.train_metrics == train_metrics
        assert result.validation_metrics == val_metrics
        assert result.best_iteration == 2
        assert result.training_time == 120.5


class TestCustomLogger:
    """Test the CustomLogger class."""

    def test_logger_initialization(self):
        """Test CustomLogger is properly initialized."""
        logger = CustomLogger()
        assert logger.logger.name == 'lightgbm_custom'

    def test_warning_suppression(self):
        """Test that warning method does nothing."""
        logger = CustomLogger()
        # Should not raise any exceptions
        logger.warning("This warning should be suppressed")

    @patch('logging.getLogger')
    def test_info_method(self, mock_get_logger):
        """Test info method calls logger.info."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        logger = CustomLogger()
        logger.info("Test info message")

        mock_logger.info.assert_called_once_with("Test info message")

    @patch('logging.getLogger')
    def test_error_method(self, mock_get_logger):
        """Test error method calls logger.error."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        logger = CustomLogger()
        logger.error("Test error message")

        mock_logger.error.assert_called_once_with("Test error message")


class TestTimeSeriesPreprocessor:
    """Test the TimeSeriesPreprocessor class."""

    def test_frequency_conversion(self):
        """Test frequency conversion mapping."""
        processor = TimeSeriesPreprocessor(freq='M', lags=[1, 2])
        assert processor.freq == 'MS'  # After conversion

        processor = TimeSeriesPreprocessor(freq='Q', lags=[1, 2])
        assert processor.freq == 'QS'

        processor = TimeSeriesPreprocessor(freq='Y', lags=[1, 2])
        assert processor.freq == 'YS-JAN'

        processor = TimeSeriesPreprocessor(freq='D', lags=[1, 2])
        assert processor.freq == 'D'  # No conversion needed

    def test_initialization(self):
        """Test TimeSeriesPreprocessor initialization."""
        lags = [1, 2, 3]
        processor = TimeSeriesPreprocessor(freq='M', lags=lags)

        assert processor.lags == lags
        assert processor.freq == 'MS'

    def test_preprocess_missing_columns(self):
        """Test preprocessing with missing required columns."""
        processor = TimeSeriesPreprocessor(freq='M', lags=[1, 2])

        # Missing 'value' column
        df = pd.DataFrame({
            'series_id': [1, 1],
            'date': pd.date_range('2020-01-01', periods=2, freq='MS')
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            processor.preprocess(df)

    def test_preprocess_lag_values_correct(self):
        """Lag columns should contain the expected shifted target values."""
        processor = TimeSeriesPreprocessor(freq='D', lags=[1, 2])

        input_df = pd.DataFrame({
            'series_id': [1, 1, 1, 1, 1],
            'date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'value': [10, 12, 14, 16, 18],
        })

        result = processor.preprocess(input_df)

        # First max(lags)=2 rows per series are dropped; 3 remain
        assert len(result) == 3
        # lag1 at row i should be value at row i-1
        assert list(result['lag1']) == [12, 14, 16]
        # lag2 at row i should be value at row i-2
        assert list(result['lag2']) == [10, 12, 14]
        # Current value should be preserved
        assert list(result['value']) == [14, 16, 18]

    def test_preprocess_lag_column_order_matches_lags_list(self):
        """Lag columns should be added in the order of the lags list."""
        processor = TimeSeriesPreprocessor(freq='D', lags=[3, 1, 2])

        input_df = pd.DataFrame({
            'series_id': [1] * 5,
            'date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'value': [10, 12, 14, 16, 18],
        })

        result = processor.preprocess(input_df)

        # Order of lag columns must follow the lags list: [3, 1, 2]
        lag_cols = [c for c in result.columns if c.startswith('lag')]
        assert lag_cols == ['lag3', 'lag1', 'lag2']

    def test_preprocess_multiple_series_independence(self):
        """Lags should be computed per-series, not across series."""
        processor = TimeSeriesPreprocessor(freq='D', lags=[1])

        input_df = pd.DataFrame({
            'series_id': [1, 1, 1, 2, 2, 2],
            'date': list(pd.date_range('2020-01-01', periods=3, freq='D')) * 2,
            'value': [10, 20, 30, 100, 200, 300],
        })

        result = processor.preprocess(input_df)

        # First row of each series dropped; 2 remain per series = 4 total
        assert len(result) == 4
        # Series 1's lag1 values should only come from series 1
        series_1 = result[result['series_id'] == 1].sort_values('date')
        assert list(series_1['lag1']) == [10, 20]
        # Series 2's lag1 values should only come from series 2
        series_2 = result[result['series_id'] == 2].sort_values('date')
        assert list(series_2['lag1']) == [100, 200]

    def test_preprocess_preserves_features(self):
        """Non-required columns should be carried through as features."""
        processor = TimeSeriesPreprocessor(freq='D', lags=[1])

        input_df = pd.DataFrame({
            'series_id': [1, 1, 1],
            'date': pd.date_range('2020-01-01', periods=3, freq='D'),
            'value': [10, 12, 14],
            'feature1': [100, 200, 300],
            'feature2': [0.1, 0.2, 0.3],
        })

        result = processor.preprocess(input_df)

        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        # Features detected and stored on the processor
        assert sorted(processor.features) == ['feature1', 'feature2']

    def test_extract_features(self):
        """Test feature extraction from a preprocessed DataFrame."""
        processor = TimeSeriesPreprocessor(freq='M', lags=[1, 2])
        processor.features = ['feature1']

        preprocessed_df = pd.DataFrame({
            'series_id': [1, 1],
            'date': pd.date_range('2020-01-01', periods=2, freq='MS'),
            'value': [10, 12],
            'feature1': [1, 2],
            'lag1': [0, 10],
            'lag2': [0, 0]
        })

        result = processor.extract(preprocessed_df)

        assert 'date' in result
        assert 'lags_target' in result
        assert 'target' in result
        assert 'features' in result
        assert result['target'].shape == (2, 1)  # Reshaped to column vector
        # lags_target should have 2 columns (lag1, lag2)
        assert result['lags_target'].shape == (2, 2)

    def test_create_lags_returns_preprocessed_dataframe(self):
        """create_lags should be a thin wrapper returning the preprocessed DataFrame."""
        processor = TimeSeriesPreprocessor(freq='D', lags=[1])

        input_df = pd.DataFrame({
            'series_id': [1, 1, 1],
            'date': pd.date_range('2020-01-01', periods=3, freq='D'),
            'value': [10, 12, 14],
            'feature1': [1, 2, 3]
        })

        result = processor.create_lags(input_df)

        assert isinstance(result, pd.DataFrame)
        assert 'lag1' in result.columns
        # First row dropped due to NaN lag
        assert len(result) == 2


class TestPrepareDatasets:
    """Test the prepare_datasets function."""

    @pytest.fixture
    def sample_ts_data(self):
        """Sample time series data for testing."""
        return pd.DataFrame({
            'series_id': [0, 0, 0, 0, 1, 1, 1, 1],
            'value': [10, 12, 14, 16, 20, 22, 24, 26],
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8]
        })

    @pytest.fixture
    def mock_preprocessor(self):
        """Mock preprocessor for testing."""
        mock = Mock()
        return mock

    def test_prepare_datasets_no_validation(self, sample_ts_data, mock_preprocessor):
        """Test prepare_datasets without validation."""
        def mock_extract(data):
            return {
                'features': pd.DataFrame({'feature1': [1, 2, 3, 4]}),
                'target': np.array([[10], [12], [14], [16]]),
                'lags_target': np.array([[0], [10], [12], [14]])
            }

        mock_preprocessor.extract.return_value = mock_extract(sample_ts_data)

        with patch('lightgbm.Dataset') as mock_dataset:
            result = prepare_datasets(
                full_ts=sample_ts_data,
                preprocessor=mock_preprocessor,
                fcst_h=1,
                dtype=torch.float32,
                validation=False
            )

            valid_sets, valid_names, callbacks, evals_result, lags_train, lags_eval, dataset_refs = result

            assert len(valid_sets) == 1
            assert valid_names == ["train"]
            assert callbacks is None
            assert evals_result is None
            assert lags_eval is None
            assert isinstance(lags_train, torch.Tensor)

    def test_prepare_datasets_with_validation(self, sample_ts_data, mock_preprocessor):
        """Test prepare_datasets with validation."""
        def mock_extract(data):
            if len(data) == 2:  # eval data
                return {
                    'features': pd.DataFrame({'feature1': [4, 8]}),
                    'target': np.array([[16], [26]]),
                    'lags_target': np.array([[14], [24]])
                }
            else:  # train data
                return {
                    'features': pd.DataFrame({'feature1': [1, 2, 3, 5, 6, 7]}),
                    'target': np.array([[10], [12], [14], [20], [22], [24]]),
                    'lags_target': np.array([[0], [10], [12], [0], [20], [22]])
                }

        mock_preprocessor.extract.side_effect = mock_extract

        with patch('lightgbm.Dataset') as mock_dataset:
            with patch('lightgbm.record_evaluation') as mock_record_eval:
                result = prepare_datasets(
                    full_ts=sample_ts_data,
                    preprocessor=mock_preprocessor,
                    fcst_h=1,
                    dtype=torch.float32,
                    validation=True
                )

                valid_sets, valid_names, callbacks, evals_result, lags_train, lags_eval, dataset_refs = result

                assert len(valid_sets) == 2
                assert valid_names == ["train", "validation"]
                assert callbacks is not None
                assert evals_result is not None
                assert isinstance(lags_train, torch.Tensor)
                assert isinstance(lags_eval, torch.Tensor)

    def test_prepare_datasets_forecast_horizon_too_large(self, sample_ts_data, mock_preprocessor):
        """Test that large forecast horizon raises error."""
        with pytest.raises(ValueError, match="Forecast horizon .* must be smaller than the minimum series length"):
            prepare_datasets(
                full_ts=sample_ts_data,
                preprocessor=mock_preprocessor,
                fcst_h=5,  # Larger than series length (4)
                dtype=torch.float32,
                validation=True
            )

    def test_prepare_datasets_early_stopping(self, sample_ts_data, mock_preprocessor):
        """Test early stopping callback configuration."""
        def mock_extract(data):
            return {
                'features': pd.DataFrame({'feature1': [1, 2]}),
                'target': np.array([[10], [12]]),
                'lags_target': np.array([[0], [10]])
            }

        mock_preprocessor.extract.side_effect = mock_extract

        with patch('lightgbm.Dataset'):
            with patch('lightgbm.record_evaluation'):
                with patch('lightgbm.early_stopping') as mock_early_stop:
                    prepare_datasets(
                        full_ts=sample_ts_data,
                        preprocessor=mock_preprocessor,
                        fcst_h=1,
                        dtype=torch.float32,
                        validation=True,
                        early_stopping_round=10
                    )

                    mock_early_stop.assert_called_once_with(
                        stopping_rounds=10,
                        verbose=False
                    )