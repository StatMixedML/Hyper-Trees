import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Tuple, List, Callable, Optional
import time
import random
from ..utils import CustomLogger
lgb.register_logger(CustomLogger())

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, GaussNewtonHessian
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)

class HyperTreeETS:
    """
    Class that implements a Hyper-Tree-ETS model for time series forecasting.

    The Hyper-Tree-ETS model extends traditional Exponential Smoothing models by allowing
    the smoothing parameters (alpha, beta, gamma, phi) to be time-varying and estimated
    by gradient boosted trees. This creates an adaptive exponential smoothing model that can capture complex
    temporal dependencies. For cross-series learning, the model assumes that all series are of the same length and
    have the same frequency. The user must ensure that the training data is properly formatted with all series having
    the same number of observations and a mask column indicating valid observations. Valid observations
    are expected to have a mask value of 1, while padded values should have a mask value of 0.

    Key features:
    - Combines tree-based models (LightGBM) with exponential smoothing time series modeling
    - Allows ETS parameters to vary based on features
    - Provides ETS parameters that can vary over time
    - Supports both triple exponential smoothing (with seasonality) and trend-only models

    Use this model when:
    - You have relevant features that might influence the smoothing structure
    - You want more flexibility than traditional ETS models

    Example usage:
    ```python
    # Imports
    from hypertrees.models import HyperTreeETS
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 12
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeETS(ets_type='triple', seasonality_feature="month", season_length=12, freq=frequency, fcst_h=fcst_h)

    # Data
    # The data needs to have the following columns: 'date', 'series_id', 'value'. All other columns are automatically treated as features.
    df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv', parse_dates=['ds'])
    df.rename(columns={'unique_id': 'series_id', 'ds': 'date', 'y': 'value'}, inplace=True)
    df['month'] = df['date'].dt.month
    df["quarter"] = df['date'].dt.quarter
    test = df.tail(fcst_h)
    train = df.drop(test.index)

    # Train model
    model.train(
        lgb_params={'learning_rate': 0.1},
        num_iterations=100,
        train_data=train
    )

    # Generate forecasts
    forecasts = model.forecast(test_data=test)

    # Plot results
    datasets = [
            (df, 'date', 'value', 'Actual', '#2E86AB', '-'),
            (forecasts, 'date', 'fcst', 'Forecast', '#F18F01', '--')
        ]

    for data, x_col, y_col, label, color, style in datasets:
        plt.plot(data[x_col], data[y_col], label=label, color=color,
                linestyle=style, linewidth=2, alpha=0.8)

    plt.title('AirPassengers - Forecast', fontsize=14)
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ```
    """

    def __init__(
            self,
            ets_type: str = "triple",
            season_length: int = 12,
            seasonality_feature: str = None,
            freq: str = "M",
            fcst_h: int = 12,
            loss_fn: Callable = nn.MSELoss(),
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-Tree-ETS model.

        Arguments
        ----------
        ets_type : str
            Type of ETS model to use. Either "triple" (with seasonality) or "trend" (linear trend-only).
        season_length : int
            Seasonal length of the time series (e.g., 12 for monthly data, 4 for quarterly).
        seasonality_feature : str
            Feature name for seasonality. This is used to create seasonal indices. Must be present in the dataset.
            For example, "month" for monthly data, "quarter" for quarterly data, etc. This is required when
            ets_type is "triple".
        freq : str
            Frequency of the time series (e.g., 'D' for daily, 'M' for monthly,
            'Q' for quarterly, 'Y' for yearly).
        fcst_h : int
            Forecast horizon (number of periods to forecast ahead).
        loss_fn : Callable
            Loss function for optimization. Must be a PyTorch loss function.
            Default is MSE loss. Must be twice-differentiable for the
            Gauss-Newton Hessian; non-smooth losses (e.g., L1Loss) have
            zero or undefined second derivatives, causing degenerate Hessians.
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            More probes reduce variance but increase computation. Default is 5.
        """
        # Validate inputs
        if ets_type not in ["triple", "trend"]:
            raise ValueError("ets_type must be either 'triple' or 'trend'.")
        if season_length <= 0:
            raise ValueError("season_length must be a positive integer.")
        if not isinstance(season_length, int):
            raise TypeError("season_length must be an integer.")
        if fcst_h <= 0:
            raise ValueError("Forecast horizon 'fcst_h' must be a positive integer.")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a PyTorch loss function.")
        if not isinstance(loss_fn, nn.MSELoss):
            import warnings
            warnings.warn(
                f"Loss {type(loss_fn).__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )
        if not isinstance(freq, str):
            raise TypeError("freq must be a string representing the frequency of the time series.")
        if seasonality_feature is None and ets_type == "triple":
            raise ValueError("seasonality_feature must be provided for triple ETS type.")

        self.ets_type = ets_type
        self.season_length = season_length
        self.seasonality_feature = seasonality_feature
        self.freq = freq
        self.n_params = 4 if ets_type == "triple" else 2  # alpha, beta, gamma, phi OR alpha, beta
        self.fcst_h = fcst_h
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.dtype = torch.float32
        self.model = None
        self.features = None            # Stores feature names after training
        self.is_trained = False         # Flag to track if model has been trained
        self.dataset_references = {}    # Store references to LightGBM datasets
        self.eps = 1e-6                 # Small constant to prevent numerical issues in sigmoid and division
        self.fcst_states = None         # Store final ETS states for forecasting
        self.n_hessian_probes = n_hessian_probes
        self._iter_count = 0            # Iteration counter for seeding Hessian probes

        # Shared Gauss-Newton Hessian estimator
        self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)

        # Conformal prediction interval state
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        # Activation function for parameter bounds
        self.sigmoid_fn = nn.Sigmoid()

        # Set the appropriate forward function based on ETS type
        if self.ets_type == "triple":
            self.forward = self._forward_triple
        elif self.ets_type == "trend":
            self.forward = self._forward_trend

    def _create_mask_from_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Create a mask for valid observations from the data.

        This function creates a mask that identifies valid (non-padded) observations
        in the dataset. If the data contains a 'mask' column, it uses that directly.
        Otherwise, it creates a mask based on non-null values.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the time series data

        Returns
        -------
        torch.Tensor
            Boolean mask indicating valid observations
        """
        if 'mask' in data.columns:
            # Use provided mask column
            mask = torch.tensor(
                data['mask'].values.reshape(self.n_series, -1),
                dtype=self.dtype
            )
        else:
            # Create mask with all ones based on shape of the data
            data_shape = data.shape[0]
            mask = torch.ones((data_shape, 1), dtype=self.dtype).reshape(self.n_series, -1)

        return mask

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts LightGBM forecasts to
        PyTorch tensors, computes the ETS forward pass, and then backpropagates to get gradients.

        Parameters
        ----------
        predt : np.ndarray
            Forecasts from LightGBM, representing the ETS parameters.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        self._iter_count += 1

        target = torch.tensor(
            data.get_label().reshape(self.n_series, -1),
            dtype=self.dtype
        )

        params, loss = self.get_params_loss(predt, target, data, requires_grad=True)
        grad, hess = self.calculate_gradients_and_hessians(loss, params)

        return grad, hess

    def eval_fn(self, predt: np.ndarray, eval_data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        Custom evaluation function for evaluating forecast accuracy on an evaluation dataset.

        This function computes the loss value to be monitored during evaluation.

        Parameters
        ----------
        predt : np.ndarray
            Outputs of LightGBM Model.
        eval_data : lgb.Dataset
            LightGBM dataset containing the evaluation data.

        Returns
        -------
        Tuple[str, float, bool]
            Name of the metric, value of the metric, and whether to maximize it.
        """
        # Calculate loss
        is_higher_better = False  # Lower loss is better, so we don't maximize
        target = torch.tensor(
            eval_data.get_label().reshape(self.n_series, -1),
            dtype=self.dtype
        )
        _, loss = self.get_params_loss(predt, target, eval_data)

        return self.loss_name, loss.item(), is_higher_better

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            data: lgb.Dataset = None,
            requires_grad: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Transform LightGBM outputs into ETS parameters and calculate loss.

        This function:
        1. Reshapes the raw outputs into ETS parameters
        2. Applies sigmoid transformation to ensure parameter bounds
        3. Runs the ETS forward pass to compute fitted values
        4. Calculates the loss between fitted values and actual values

        Parameters
        ----------
        predt : np.ndarray
            Outputs of LightGBM Model.
        target : torch.Tensor
            Target values (actual time series values).
        data : lgb.Dataset
            LightGBM dataset containing additional information.
        requires_grad : bool
            Whether to compute gradients (True during training).

        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            Parameters tensor list and loss value.
        """
        # Gradients must be w.r.t. raw (pre-sigmoid) outputs
        # differentiating w.r.t. post-sigmoid params would miss the sigmoid(predt) factor
        predt = nn.Parameter(
            torch.tensor(
                predt.reshape(-1, self.n_params, order="F"),
                dtype=self.dtype
            ),
            requires_grad=requires_grad
        )

        # Apply sigmoid transformation and reshape for ETS computation; clamp to avoid numerical issues
        params = torch.clamp(
            self.sigmoid_fn(predt.reshape(self.n_series, -1, self.n_params)),
            min=self.eps,
            max=1-self.eps
        )

        # Get mask
        if "mask" in data.data.columns:
            mask = torch.tensor(
                data.data["mask"].values.reshape(self.n_series, -1),
                dtype=self.dtype
            )
        else:
            series_len = target.shape[1]
            mask = torch.ones((self.n_series, series_len), dtype=self.dtype)

        # Forward pass to compute fitted values
        _, _, _, fit = self.forward(params, data, target, mask)

        # Stack fitted values and compute loss with masking
        fit = torch.stack(fit, dim=1)
        loss = self.loss_fn(fit * mask, target * mask)

        # Store for Gauss-Newton Hessian estimation
        self._fit = fit
        self._mask = mask
        self._target = target

        return predt, loss

    def _forward_triple(
            self,
            params: torch.Tensor,
            data: lgb.Dataset,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for triple exponential smoothing (with seasonality).

        This implements the ETS state space equations for triple exponential smoothing:
        - Level: l_t = α(y_t/s_{t-m}) + (1-α)(l_{t-1} + φb_{t-1})
        - Trend: b_t = β(l_t - l_{t-1}) + (1-β)φb_{t-1}
        - Seasonality: s_t = γ(y_t/(l_{t-1} + φb_{t-1})) + (1-γ)s_{t-m}
        - Fitted: ŷ_t = (l_{t-1} + φb_{t-1}) * s_{t-m}

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]
            Final level (n_series,), final trend (n_series,),
            seasonality matrix (n_series, season_length), and list of fitted values.
        """
        series_len = target.shape[1]

        # Unpack and pre-unbind parameters
        alpha, beta, gamma, phi = params.unbind(dim=2)
        alpha_t = alpha.unbind(dim=1)
        beta_t = beta.unbind(dim=1)
        gamma_t = gamma.unbind(dim=1)
        phi_t = phi.unbind(dim=1)

        # Initialize seasonality
        init_len = min(self.season_length, series_len)
        seasonal_avg = torch.zeros((self.n_series, init_len), dtype=self.dtype)

        # Compute initial seasonal averages
        for i in range(init_len):
            valid_obs = mask[:, i::init_len]
            seasonal_avg[:, i] = (target[:, i::init_len] * valid_obs.float()).sum(
                1) / valid_obs.float().sum(1).clamp(min=1)

        # Initialize seasonality as ratios since we are using multiplicative seasonality
        seasonality = target[:, :init_len] / (seasonal_avg + self.eps)
        if init_len < self.season_length:
            # Pad with ones for missing season positions
            pad = torch.ones((self.n_series, self.season_length - init_len), dtype=self.dtype)
            seasonality = torch.cat([seasonality, pad], dim=1)

        # ETS initialization using season-adjusted initialization
        season_adj = target[:, :init_len] / seasonality[:, :init_len]
        level_prev = season_adj[:, 0]
        trend_prev = season_adj[:, min(1, init_len - 1)] - season_adj[:, 0]
        fits = [target[:, 0]]

        # Get seasonal indices from features
        seasonality_idxs = torch.tensor(
            data.data[self.seasonality_feature].values - 1,
            dtype=torch.long
        ).reshape(self.n_series, -1)
        batch_idx = torch.arange(self.n_series, dtype=torch.long)

        # Triple ETS updates with masking for padded values
        for t in range(1, series_len):
            s_idx = seasonality_idxs[:, t]
            valid_mask = mask[:, t]

            fit_t = valid_mask * (
                    (level_prev + phi_t[t] * trend_prev) * seasonality[batch_idx, s_idx]
            ) + (1 - valid_mask) * fits[-1]

            level_new = valid_mask * (
                    alpha_t[t] * (target[:, t] / seasonality[batch_idx, s_idx]) +
                    (1 - alpha_t[t]) * (level_prev + phi_t[t] * trend_prev)
            ) + (1 - valid_mask) * level_prev

            trend_new = valid_mask * (
                    beta_t[t] * (level_new - level_prev) +
                    (1 - beta_t[t]) * phi_t[t] * trend_prev
            ) + (1 - valid_mask) * trend_prev

            seasonality[batch_idx, s_idx] = valid_mask * (
                    gamma_t[t] * (target[:, t] / (level_prev + phi_t[t] * trend_prev)) +
                    (1 - gamma_t[t]) * seasonality[batch_idx, s_idx]
            ) + (1 - valid_mask) * seasonality[batch_idx, s_idx]

            fits.append(fit_t)
            level_prev = level_new
            trend_prev = trend_new

        return level_prev, trend_prev, seasonality, fits

    def _forward_trend(
            self,
            params: torch.Tensor,
            data: lgb.Dataset,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, List[torch.Tensor]]:
        """
        Forward pass for trend-only exponential smoothing.

        This implements the ETS state space equations for trend-only model:
        - Level: l_t = αy_t + (1-α)(l_{t-1} + b_{t-1})
        - Trend: b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}
        - Fitted: ŷ_t = l_{t-1} + b_{t-1}

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, None, List[torch.Tensor]]
            Final level (n_series,), final trend (n_series,),
            None (no seasonality), and list of fitted values.
        """
        series_len = target.shape[1]

        # Unpack and pre-unbind parameters
        alpha, beta = params.unbind(dim=2)
        alpha_t = alpha.unbind(dim=1)
        beta_t = beta.unbind(dim=1)

        # Initialize states for trend model
        level_prev = target[:, 0]
        last_idx = min(2 * self.season_length - 1, series_len - 1)
        trend_prev = (target[:, last_idx] - target[:, 0]) / max(last_idx, 1)
        fits = [target[:, 0]]

        # Trend-only updates with masking for padded values
        for t in range(1, series_len):
            valid_mask = mask[:, t]

            fit_t = valid_mask * (level_prev + trend_prev) + (1 - valid_mask) * fits[-1]

            level_new = valid_mask * (
                    alpha_t[t] * target[:, t] +
                    (1 - alpha_t[t]) * (level_prev + trend_prev)
            ) + (1 - valid_mask) * level_prev

            trend_new = valid_mask * (
                    beta_t[t] * (level_new - level_prev) +
                    (1 - beta_t[t]) * trend_prev
            ) + (1 - valid_mask) * trend_prev

            fits.append(fit_t)
            level_prev = level_new
            trend_prev = trend_new

        return level_prev, trend_prev, None, fits

    def calculate_gradients_and_hessians(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute gradients and Generalized Gauss-Newton Hessians for LightGBM.

        Uses exact first-order gradients and the Generalized Gauss-Newton (GGN)
        approximation for the Hessian diagonal, estimated via Hutchinson probing.
        The GGN replaces the MSE-specific (2/N) scaling with per-observation
        loss curvature d^2 L / d(y_hat)^2, computed via autograd. This supports
        any twice-differentiable element-wise loss function.

        The Gauss-Newton approximation drops the residual-curvature term from the
        exact Hessian, retaining only H_GN = J^T B J where J is the Jacobian of
        forecasts w.r.t. parameters and B is the diagonal loss curvature matrix.
        This avoids second-order differentiation through the ETS state-space
        recurrence and guarantees positive semi-definite Hessians [1]. The diagonal
        of J^T B J is estimated via Hutchinson's stochastic trace estimator [2]
        using K random Gaussian probe vectors.

        References
        ----------
        [1] Martens, J. (2020). New Insights and Perspectives on the Natural
            Gradient Method. Journal of Machine Learning Research, 21(146), 1-76.
        [2] Hutchinson, M. F. (1990). A Stochastic Estimator of the Trace of the
            Influence Matrix for Laplacian Smoothing Splines. Communications in
            Statistics - Simulation and Computation, 19(2), 433-450.
        [3] Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
            Springer.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (pre-sigmoid LightGBM outputs, nn.Parameter).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        grad = autograd(loss, params, retain_graph=True)[0]

        fit_masked = self._fit * self._mask
        target_masked = self._target * self._mask
        rng = torch.Generator().manual_seed(self._iter_count)
        hess = self._gn_hessian.estimate(fit_masked, target_masked, params, rng)

        # Release graph references to prevent accumulation between iterations
        self._fit = None
        self._mask = None
        self._target = None

        # Convert to numpy arrays and reshape as expected by LightGBM
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")

        return grad, hess

    def train(
            self,
            lgb_params: dict = None,
            num_iterations: int = 100,
            train_data: pd.DataFrame = None,
            validation: bool = False,
            early_stopping_round: Optional[int] = None,
            seed: int = 123,
            verbose: int = -1,
            deterministic: bool = True,
            forecast_intervals: Optional[ForecastIntervals] = None,
    ) -> TrainingResult:
        """
        Train the Hyper-Tree-ETS model on time series data.

        This method:
        1. Preprocesses the time series data to create features and handle variable lengths
        2. Sets up LightGBM datasets with proper masking
        3. Trains the model using gradient boosting
        4. Stores final ETS states for future forecasting

        The training data must contain columns:
        - 'series_id': Identifier for each time series
        - 'date': Timestamp for each observation
        - 'value': Target value to forecast
        - Additional feature columns used for forecasting

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters like 'learning_rate', 'num_leaves', etc.
        num_iterations : int
            Number of boosting rounds for training
        train_data : pd.DataFrame
            Training data containing series_id, date, value and feature columns. All series must have the same length.
            The data should be preprocessed to ensure that all series are of the same length and padded with 1
            in the 'mask' column for valid observations. Padded values should have a mask value of 0.
        validation : bool
            If True, a validation set will be created for evaluation.
        early_stopping_round : int, optional
            If provided, training will stop if the validation loss does not improve for this many rounds.
        seed : int
            Random seed for reproducibility
        verbose : int
            Verbosity level for LightGBM training
        deterministic : bool
            If True, sets LightGBM's ``deterministic`` and ``force_row_wise`` parameters to ensure
            reproducible results. May slow down training. See
            https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic
        forecast_intervals : ForecastIntervals, optional
            If provided, calibrate conformal prediction intervals via rolling-window
            cross-validation after the main model is trained. The collected conformity
            scores are then used by ``forecast(..., level=[...])`` to produce
            ``<model>-lo-<level>`` / ``<model>-hi-<level>`` columns. See
            :class:`hypertrees.conformal.ForecastIntervals`.

        Returns
        -------
        TrainingResult
            Object containing evaluation results and training information.
        """
        # Validate inputs
        if train_data is None:
            raise ValueError("train_data must be provided.")
        if lgb_params is None:
            raise ValueError("lgb_params must be provided.")
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must be a pandas DataFrame.")
        if not isinstance(lgb_params, dict):
            raise TypeError("lgb_params must be a dictionary.")
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if not isinstance(verbose, int):
            raise TypeError("verbose must be an integer.")
        if early_stopping_round is not None and (
                not isinstance(early_stopping_round, int) or early_stopping_round <= 0):
            raise ValueError("early_stopping_round must be a positive integer.")
        if not isinstance(validation, bool):
            raise TypeError("validation must be a boolean.")
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean.")
        if forecast_intervals is not None and not isinstance(forecast_intervals, ForecastIntervals):
            raise TypeError("forecast_intervals must be a ForecastIntervals instance.")
        if early_stopping_round is not None and not validation:
            raise ValueError("early_stopping_round can only be used when validation is True.")
        if validation and early_stopping_round is None:
            raise ValueError("early_stopping_round must be provided when validation is True.")

        if deterministic:
            lgb_params = {**lgb_params, "deterministic": True, "force_row_wise": True}

        # Check required columns first
        required_columns = ['series_id', 'date', 'value']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")

        # Validate row ordering: each series must be a contiguous block with
        # monotonic dates so the ETS reshape to (n_series, T, n_params) aligns.
        validate_series_order(train_data, name="train_data")

        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals,
                min_train=self.season_length + 1,
            )

        # Check if all series in train_data have the same length
        unique_lengths = train_data.groupby('series_id')['date'].nunique()
        if len(unique_lengths.unique()) > 1:
            raise ValueError("All series in train_data must have the same length. Found multiple lengths.")

        # General model parameters
        self.lgb_params = {
            "num_class": self.n_params,
            "objective": self.objective_fn,
            "metric": "None",
            "random_seed": seed,
            "verbose": verbose
        }

        # Reset states
        self._iter_count = 0
        self._fit = None
        self._mask = None
        self._target = None
        self.model = None
        self.dataset_references = {}
        self.is_trained = False
        self.fcst_states = None
        self.features = None
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Copy to avoid modifying the caller's DataFrame across repeated train() calls
        train_data = train_data.copy()

        # Update with user-provided LightGBM parameters
        self.lgb_params.update(lgb_params)

        try:
            # Initialize TimeSeriesPreprocessor for ETS-specific preprocessing
            preprocessor = TimeSeriesPreprocessor(
                freq=self.freq,
                lags=[],  # ETS doesn't need lag features like AR models
            )

            # Process dataset - this should handle the ETS-specific preprocessing
            # including creating proper masking for variable-length series
            full_ts = preprocessor.create_lags(train_data)
            full_dict = preprocessor.extract(full_ts)

            # Store feature names and dataset dimensions
            self.features = full_dict["features"].columns.tolist()
            self.n_series = len(train_data['series_id'].unique())

            # Prepare datasets (adapted for ETS)
            (valid_sets,
             valid_names,
             callbacks,
             evals_result,
             _,  # No lags for ETS
             _,  # No lags for ETS
             self.dataset_references) = (
                prepare_datasets(
                    full_ts=full_ts,
                    preprocessor=preprocessor,
                    fcst_h=self.fcst_h,
                    dtype=self.dtype,
                    validation=validation,
                    early_stopping_round=early_stopping_round,
                    free_raw_data=False,
                )
            )

            # Train LightGBM model
            start_time = time.time()
            self.model = lgb.train(
                self.lgb_params,
                valid_sets[0],
                num_boost_round=num_iterations,
                feval=self.eval_fn if validation else None,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            training_time = time.time() - start_time

            # Store final ETS states for forecasting
            self._store_final_states(train_data)

            # Set trained flag to True
            self.is_trained = True

            if forecast_intervals is not None:
                def _model_factory():
                    return HyperTreeETS(
                        ets_type=self.ets_type,
                        season_length=self.season_length,
                        seasonality_feature=self.seasonality_feature,
                        freq=self.freq,
                        fcst_h=self.fcst_h,
                        loss_fn=self.loss_fn,
                        n_hessian_probes=self.n_hessian_probes,
                    )

                cal_train_kwargs = dict(
                    lgb_params=lgb_params,
                    num_iterations=num_iterations,
                    validation=False,
                    seed=seed,
                    verbose=verbose,
                    deterministic=deterministic,
                )
                self._cs_scores, self._cs_series_order = rolling_origin_residuals(
                    model_factory=_model_factory,
                    train_data=train_data,
                    fcst_h=self.fcst_h,
                    forecast_intervals=forecast_intervals,
                    train_kwargs=cal_train_kwargs,
                )
                self._pi_config = forecast_intervals
                self._is_calibrated = True

            # Return results
            result = TrainingResult(
                train_metrics=evals_result["train"] if validation else {"loss": []},
                validation_metrics=evals_result["validation"] if validation else None,
                best_iteration=self.model.best_iteration-1 if hasattr(self.model, 'best_iteration') else num_iterations,
                training_time=training_time
            )

            return result

        except Exception as e:
            self.is_trained = False
            raise RuntimeError(f"Training failed: {str(e)}")

    def _store_final_states(self, train_data: pd.DataFrame):
        """
        Store final ETS states after training for use in forecasting.

        Runs a full forward pass on the training data to obtain the final
        level, trend, and (for triple ETS) seasonality states per series.
        Also stores the series ordering to ensure consistent state access.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data used to compute final ETS states.
        """
        # Store series ordering from training data
        self.series_order = train_data['series_id'].unique().tolist()

        # Get fitted parameters for the full training period
        params = torch.clamp(
            self.sigmoid_fn(torch.tensor(self.model.predict(train_data[self.features]), dtype=self.dtype)),
            min=self.eps,
            max=1-self.eps
        ).reshape(self.n_series, -1, self.n_params)

        # Create mask and target tensors
        train_mask = self._create_mask_from_data(train_data)
        target = torch.tensor(
            train_data["value"].values.reshape(self.n_series, -1),
            dtype=self.dtype
        )

        # Create LightGBM dataset for forward pass
        dfit = lgb.Dataset(
            data=train_data[self.features],
            label=train_data["value"].values.reshape(-1, ),
            free_raw_data=False,
        )

        # Forward pass to get final states
        last_level, last_trend, seasonality, fit = self.forward(params, dfit, target, train_mask)

        # Store final states as dictionary with series_id as keys
        self.fcst_states = {}
        for i, series_id in enumerate(self.series_order):
            self.fcst_states[series_id] = {
                'last_level': last_level[i],
                'last_trend': last_trend[i]
            }

        # Store seasonality for triple ETS
        if self.ets_type == "triple" and seasonality is not None:
            for i, series_id in enumerate(self.series_order):
                self.fcst_states[series_id]['seasonality'] = seasonality[i]

    def set_forecast_origin(self, history: pd.DataFrame) -> None:
        """Re-anchor ETS states to the end of *history* without retraining.

        Recomputes the terminal ``{level, trend, seasonality}`` states by
        running the full ETS forward recurrence over *history* using the
        already-trained GBDT parameters.  Used by conformal calibration with
        ``refit=False``.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with the same columns as the training data (including
            any ``mask`` / ``seasonality_feature`` columns), ordered by
            ``(series_id, date)`` with each series in a contiguous block and
            all series of equal length.
        """
        self._store_final_states(history)

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast ETS parameters for each test point
        2. Uses stored final states (level, trend, seasonality) for forecast initialization
        3. Iteratively generates forecasted values using ETS equations

        The forecasting process rolls the ETS state-space recursion forward,
        mirroring the training forward pass.

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the ETS parameters used for forecasting
        level : list of int, optional
            Confidence levels (in ``(0, 100)``, e.g. ``[80, 90]``) for conformal
            prediction intervals. Only valid with ``type="forecast"`` and requires
            the model to have been trained with ``forecast_intervals=...``. Adds
            ``<model>-lo-<level>`` / ``<model>-hi-<level>`` columns to the output.

        Returns
        -------
        pd.DataFrame
            Forecasted data with columns:
            - series_id: Identifier for each time series
            - date: Forecast date/time
            - fcst: Forecasted value (if type="forecast")
            - model: Model name identifier
            - alpha, beta, gamma, phi: ETS parameter values (if type="parameters")
            - <model>-lo-<level> / <model>-hi-<level>: prediction interval bounds
              (if type="forecast" and level is provided)
        """
        # Check if model is trained and states are stored
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained. Call train() before forecasting.")
        if self.fcst_states is None or self.series_order is None:
            raise RuntimeError("Final states not found. This should not happen after training.")

        # Validate input data
        required_cols = ['series_id', 'date']
        for col in required_cols:
            if col not in test_data.columns:
                raise ValueError(f"Required column '{col}' not found in test_data")

        # Validate row ordering: each series must be a contiguous block with
        # monotonic dates so the forecast reshape aligns with stored states.
        validate_series_order(test_data, name="test_data")

        # Validate that test_data series_ids match training series_ids
        test_series_ids = set(test_data['series_id'].unique())
        train_series_ids = set(self.series_order)
        if test_series_ids != train_series_ids:
            missing_in_test = train_series_ids - test_series_ids
            extra_in_test = test_series_ids - train_series_ids
            error_msg = []
            if missing_in_test:
                error_msg.append(f"Missing series in test_data: {missing_in_test}")
            if extra_in_test:
                error_msg.append(f"Extra series in test_data: {extra_in_test}")
            raise ValueError(". ".join(error_msg))

        # Validate rows per series matches forecast horizon (forecast only;
        # parameters can be requested for arbitrary-length input).
        if type == "forecast":
            rows_per_series = test_data.groupby("series_id", sort=False).size()
            bad = rows_per_series[rows_per_series != self.fcst_h]
            if not bad.empty:
                raise ValueError(
                    f"Each series must have exactly fcst_h={self.fcst_h} rows in test_data. "
                    f"Series with wrong counts: {bad.to_dict()}"
                )

        # Validate type parameter
        if type not in ["forecast", "parameters"]:
            raise ValueError("Parameter 'type' must be either 'forecast' or 'parameters'")

        if level is not None:
            if type != "forecast":
                raise ValueError("level is only supported with type='forecast'.")
            if not self._is_calibrated:
                raise RuntimeError(
                    "Prediction intervals were requested via level, but the model "
                    "was not calibrated. Pass forecast_intervals=ForecastIntervals(...) "
                    "to train() before forecasting with level."
                )
            if not isinstance(level, (list, tuple)) or len(level) == 0:
                raise ValueError("level must be a non-empty list of integers.")
            for lv in level:
                if not isinstance(lv, (int, np.integer)) or not 0 < lv < 100:
                    raise ValueError(f"level values must be integers in (0, 100); got {lv}.")

        try:
            # If mask was a training feature but is absent from test_data, add it (all test obs are valid)
            if 'mask' in self.features and 'mask' not in test_data.columns:
                test_data = test_data.copy()
                test_data['mask'] = np.ones_like(test_data['series_id'], dtype=np.int32)

            # Check that all features used during training exist in test_data
            missing_features = [f for f in self.features if f not in test_data.columns]
            if missing_features:
                raise ValueError(f"Missing features in test_data: {missing_features}")

            if type == "forecast":
                # Get ETS parameter forecasts from the LightGBM model
                fcst_params = torch.clamp(
                    self.sigmoid_fn(torch.tensor(self.model.predict(test_data[self.features]),dtype=self.dtype)
                    ),
                    min=self.eps,
                    max=1-self.eps
                ).reshape(self.n_series, self.fcst_h, self.n_params)

                # Extract stored final states in the correct order of the test data series
                test_series_ids = test_data['series_id'].unique()
                last_level = torch.stack([self.fcst_states[series_id]['last_level'] for series_id in test_series_ids])
                last_trend = torch.stack([self.fcst_states[series_id]['last_trend'] for series_id in test_series_ids])

                # Generate forecasts by rolling the ETS state forward, mirroring
                # the training forward pass and using each step's forecast as
                # the pseudo-observation in the state updates.
                if self.ets_type == "triple":
                    # Extract seasonality in test series order
                    seasonality = torch.stack(
                        [self.fcst_states[series_id]['seasonality'] for series_id in test_series_ids]
                    )

                    seasonality_idxs = torch.tensor(
                        test_data[self.seasonality_feature].values - 1
                    ).reshape(self.n_series, self.fcst_h)
                    batch_idx = torch.arange(self.n_series, dtype=torch.long)
                    alpha_fcst = fcst_params[:, :, 0]
                    beta_fcst = fcst_params[:, :, 1]
                    gamma_fcst = fcst_params[:, :, 2]
                    phi_fcst = fcst_params[:, :, 3]

                    fcsts = []
                    level_h = last_level
                    trend_h = last_trend
                    for h in range(self.fcst_h):
                        alpha = alpha_fcst[:, h]
                        beta = beta_fcst[:, h]
                        gamma = gamma_fcst[:, h]
                        phi = phi_fcst[:, h]
                        s_idx = seasonality_idxs[:, h].long()
                        s_h = seasonality[batch_idx, s_idx]

                        # One-step-ahead fit (pseudo-observation),
                        # structurally identical to _forward_triple
                        pseudo_y = (level_h + phi * trend_h) * s_h
                        fcsts.append(pseudo_y.reshape(-1, 1))

                        # State updates exactly as in _forward_triple.
                        level_new = (
                            alpha * (pseudo_y / s_h)
                            + (1 - alpha) * (level_h + phi * trend_h)
                        )
                        trend_new = (
                            beta * (level_new - level_h)
                            + (1 - beta) * phi * trend_h
                        )
                        seasonality[batch_idx, s_idx] = (
                            gamma * (pseudo_y / (level_h + phi * trend_h))
                            + (1 - gamma) * s_h
                        )

                        level_h = level_new
                        trend_h = trend_new

                elif self.ets_type == "trend":
                    alpha_fcst = fcst_params[:, :, 0]
                    beta_fcst = fcst_params[:, :, 1]

                    fcsts = []
                    level_h = last_level
                    trend_h = last_trend
                    for h in range(self.fcst_h):
                        alpha = alpha_fcst[:, h]
                        beta = beta_fcst[:, h]

                        # One-step-ahead fit (pseudo-observation),
                        # structurally identical to _forward_trend
                        pseudo_y = level_h + trend_h
                        fcsts.append(pseudo_y.reshape(-1, 1))

                        # State updates exactly as in _forward_trend
                        level_new = (
                            alpha * pseudo_y
                            + (1 - alpha) * (level_h + trend_h)
                        )
                        trend_new = (
                            beta * (level_new - level_h)
                            + (1 - beta) * trend_h
                        )

                        level_h = level_new
                        trend_h = trend_new

                # Create output dataframe
                model_name = f"Hyper-Tree-ETS({self.ets_type})"
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": torch.cat(fcsts, dim=1).flatten().numpy(),
                    "model": model_name,
                })

                if level is not None:
                    point = torch.cat(fcsts, dim=1).numpy()  # (n_series, fcst_h)
                    test_series_ids = test_data["series_id"].unique()
                    columns = interval_columns(
                        point=point,
                        scores=self._cs_scores,
                        levels=level,
                        method=self._pi_config.method,
                        model_name=model_name,
                        cal_order=self._cs_series_order,
                        target_order=list(test_series_ids),
                    )
                    for col_name, values in columns.items():
                        out_df[col_name] = values

            elif type == "parameters":
                fcst_params = torch.clamp(
                    self.sigmoid_fn(torch.tensor(self.model.predict(test_data[self.features]), dtype=self.dtype)
                    ),
                    min=self.eps,
                    max=1-self.eps
                ).reshape(self.n_series, -1, self.n_params)
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": f"Hyper-Tree-ETS({self.ets_type})",
                })
                param_names = ["alpha", "beta", "gamma", "phi"] if self.ets_type == "triple" else ["alpha", "beta"]
                for i, param_name in enumerate(param_names):
                    out_df[param_name] = fcst_params[:, :, i].flatten().numpy()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}")
