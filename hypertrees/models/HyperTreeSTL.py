import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Tuple, Callable, Optional
import time
from ..utils import CustomLogger
lgb.register_logger(CustomLogger())

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, NoDeepcopyObjective

class HyperTreeSTL:
    """
    Class that implements a Hyper-Tree-STL model for time series decomposition.

    The Hyper-Tree-STL model extends traditional STL (Seasonal and Trend decomposition using Loess)
    by allowing the decomposition parameters to be time-varying and estimated by gradient boosted trees.
    This creates an adaptive decomposition model that can capture complex temporal patterns in both
    trend and seasonal components.

    Key features:
    - Combines tree-based models (LightGBM) with STL decomposition
    - Allows decomposition parameters to vary based on features
    - Provides adaptive trend and seasonal components
    - Currently supports a single time series per model; train one instance per series.

    Use this model when:
    - You have relevant features that might influence the decomposition structure
    - You want more flexibility than traditional STL decomposition

    Example usage:
    ```python
    # Imports
    from hypertrees.models import HyperTreeSTL
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    frequency = 'M'
    fcst_h=12
    model = HyperTreeSTL(
        period=12,
        num_seasonal_components=1,
        freq=frequency,
        fcst_h=fcst_h
    )

    # Data
    # The data needs to have the following columns: 'date', 'series_id', 'value', 'time'. All other columns are automatically treated as features.
    df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv', parse_dates=['ds'])
    df.rename(columns={'unique_id': 'series_id', 'ds': 'date', 'y': 'value'}, inplace=True)
    df['month'] = df['date'].dt.month
    df["quarter"] = df['date'].dt.quarter
    df['time'] = df.groupby("series_id").cumcount() + 1
    test = df.tail(fcst_h)
    train = df.drop(test.index)

    # Train model
    model.train(
        lgb_params={'learning_rate': 0.3},
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
            period: int = 12,
            num_seasonal_components: int = 1,
            freq: str = "M",
            fcst_h: int = 12,
            loss_fn: Callable = nn.MSELoss(),
            type: str = "default"
    ):
        """
        Initialize the Hyper-Tree-STL model.

        Arguments
        ----------
        period : int
            Seasonal period of the time series (e.g., 12 for monthly data, 4 for quarterly).
            Must be a positive integer.
        num_seasonal_components : int
            Number of seasonal harmonics to include in the decomposition.
        freq : str
            Frequency of the time series (e.g., 'D' for daily, 'M' for monthly,
            'Q' for quarterly, 'Y' for yearly).
        fcst_h : int
            Forecast horizon (number of periods to forecast ahead).
        loss_fn : Callable
            Loss function for optimization. Must be a PyTorch loss function.
            Default is MSE loss. Losses other than nn.MSELoss are not
            recommended, as they have not been systematically tested yet.
            nn.L1Loss is rejected (zero second derivative almost everywhere
            breaks Newton boosting).
        type : str
            Type of model variant to use. Currently, "default" and "paper" are supported:
            - "paper" uses the original method from the paper
            - "default" uses an updated method with improved trend smoothing.
        """
        # Validate inputs
        if not isinstance(period, int) or period <= 0:
            raise ValueError("Period must be a positive integer.")
        if not isinstance(num_seasonal_components, int) or num_seasonal_components <= 0:
            raise ValueError("num_seasonal_components must be a positive integer.")
        if fcst_h <= 0:
            raise ValueError("Forecast horizon 'fcst_h' must be a positive integer.")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a PyTorch loss function.")
        if isinstance(loss_fn, nn.L1Loss):
            raise ValueError(
                "nn.L1Loss is not supported: its second derivative is zero almost "
                "everywhere, so LightGBM's Newton boosting receives all-zero Hessians "
                "and cannot grow trees. Use nn.HuberLoss or nn.SmoothL1Loss for an "
                "MAE-like loss with usable curvature."
            )
        if not isinstance(freq, str):
            raise TypeError("freq must be a string representing the frequency of the time series.")
        if type not in ["default", "paper"]:
            raise ValueError("Type must be either 'default' or 'paper'.")

        self.period = period
        self.freq = freq
        self.fcst_h = fcst_h
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.dtype = torch.float32
        self.forward_type = type

        # Calculate number of parameters based on period
        # 2 for trend (intercept + slope) + 2 * number of seasonal harmonics
        self.num_seasonal_components = num_seasonal_components
        self.n_params = 2 + 2 * num_seasonal_components

        if self.forward_type == "paper":
            self._forward = self._forward_paper
        elif self.forward_type == "default":
            self._forward = self._forward_default
            self.n_params += 1  # Extra parameter for trend smoothing window


        self.model = None
        self.features = None  # Stores feature names after training
        self.is_trained = False  # Flag to track if model has been trained
        self.dataset_references = {}  # Store references to LightGBM datasets

    def objective_fn(
            self,
            predt: np.ndarray,
            data: lgb.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts the raw LightGBM outputs to
        PyTorch tensors, calculates the STL decomposition, and then backpropagates to get gradients.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the STL decomposition parameters.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        # Target values
        target = torch.tensor(
            data.get_label(),
            dtype=self.dtype
        ).reshape(-1, self.n_series)

        # Calculate gradients and hessians
        params, loss = self.get_params_loss(predt, target, self.time_idx_train, requires_grad=True)
        grad, hess = self._calculate_gradients_and_hessians(loss, params)

        return grad, hess

    def eval_fn(
            self,
            predt: np.ndarray,
            eval_data: lgb.Dataset
    ) -> Tuple[str, float, bool]:
        """
        Custom evaluation function for evaluating forecast accuracy on an evaluation dataset.

        This function calculates the loss value to be monitored during evaluation.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM.
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
            eval_data.get_label(),
            dtype=self.dtype
        ).reshape(-1, self.n_series)

        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        if dataset_name == "train":
            time_idx = self.time_idx_train
        elif dataset_name == "validation":
            time_idx = self.time_idx_eval
        else:
            # Default to training if unknown
            time_idx = self.time_idx_train
            warnings.warn("Unknown dataset in metric_fn. Using training time_idx.")

        _, loss = self.get_params_loss(predt, target, time_idx, requires_grad=False)

        return self.loss_name, loss.item(), is_higher_better

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            time_idx: torch.Tensor,
            requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into STL parameters and calculate loss.

        This function:
        1. Reshapes the raw outputs into STL parameters
        2. Calculates trend and seasonal components
        3. Calculates the loss between components and target decomposition
        4. Applies smoothing penalties to the trend component

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM.
        target : torch.Tensor
            Target values (actual time series values).
        time_idx : torch.Tensor
            Time indices for the observations.
        requires_grad : bool
            Whether to calculate gradients (True during training).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Parameters tensor and loss value.
        """
        # Reshape outputs into parameter matrix (samples × n_params)
        # The 'F' order means Fortran-style ordering (column-major)
        predt = nn.Parameter(
            torch.tensor(
                predt.reshape(-1, self.n_params, order="F"),
                dtype=self.dtype
            ),
            requires_grad=requires_grad
        )

        # Reshape to (seq_len, n_series, n_params)
        params = predt.reshape(-1, self.n_series, self.n_params)

        # Forward pass to calculate trend and seasonal components
        trend, seasonality = self._forward(params, time_idx)

        # Decompose target into trend and seasonal components
        y_trend = target - seasonality
        y_seasonality = target - trend

        # Calculate losses for trend and add smoothing penalties
        loss_trend = self.loss_fn(trend, y_trend)
        smooth_d1 = torch.nanmean(torch.diff(trend, dim=0, n=1) ** 2, dim=0)
        smooth_d2 = torch.nanmean(torch.diff(trend, dim=0, n=2) ** 2, dim=0)
        smooth_penalty = torch.nanmean(torch.cat([smooth_d1, smooth_d2], dim=0))
        loss_trend += smooth_penalty

        # Loss for seasonal component
        loss_seasonality = self.loss_fn(seasonality, y_seasonality)

        # Combine losses
        loss = (loss_trend + loss_seasonality) / 2

        return predt, loss

    def _calculate_gradients_and_hessians(
            self,
            loss: torch.Tensor,
            params: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for LightGBM optimization.

        This function calculates first and second-order derivatives needed for
        gradient boosting optimization in LightGBM.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (STL decomposition parameters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        # Backpropagate to compute gradients
        loss.backward(create_graph=True)

        # Compute gradients
        grad = params.grad

        # Compute hessians. We compute the diagonal of the Hessian matrix for each parameter separately
        hess = [
            autograd(grad[:, i].sum(), params, retain_graph=True)[0][:, i:(i + 1)]
            for i in range(self.n_params)
        ]

        # Convert to numpy arrays and reshape as expected by LightGBM
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).cpu().detach().numpy().ravel(order="F")

        # Clear existing gradients to prevent accumulation
        params.grad = None

        return grad, hess

    def _forward_paper(
            self,
            params: torch.Tensor,
            time_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute the trend and seasonality from STL parameters.
        This implementation follows the original method from the paper.

        Parameters
        ----------
        params : torch.Tensor
            STL decomposition parameters.
        time_idx : torch.Tensor
            Time indices for the observations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Trend and seasonal components.
        """
        # Trend component with intercept and slope
        trend = params[:, :, 0] + params[:, :, 1] * time_idx

        # Seasonal component: Fourier series representation
        H = self.num_seasonal_components
        seasonal_weights_sine = params[:, :, 2:2 + H]
        seasonal_weights_cosine = params[:, :, 2 + H:2 + 2 * H]

        seasonality = torch.sum(
            torch.cat(
                [
                    (
                            seasonal_weights_sine[:, :, i] * torch.sin(
                        time_idx * (i + 1) * (2 * torch.pi / self.period))
                    ).unsqueeze(-1)
                    +
                    (
                            seasonal_weights_cosine[:, :, i] * torch.cos(
                        time_idx * (i + 1) * (2 * torch.pi / self.period))
                    ).unsqueeze(-1)
                    for i in range(H)
                ], dim=2
            ),
            dim=2
        )

        # Center the seasonal component (remove mean)
        seasonality = seasonality - torch.mean(seasonality, dim=0, keepdim=True)

        return trend, seasonality

    def _forward_default(
            self,
            params: torch.Tensor,
            time_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to calculate the trend and seasonality from STL parameters.
        This implementation includes an updated trend smoothing method and more efficient seasonality calculation.

        The trend smoothing window (params[:, :, 2]) is learnable: it enters
        through a differentiable soft-boxcar kernel so gradients reach the GBDT.

        Parameters
        ----------
        params : torch.Tensor
            STL decomposition parameters.
        time_idx : torch.Tensor
            Time indices for the observations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Trend and seasonal components.
        """
        dtype = time_idx.dtype
        T, N = time_idx.shape
        m = self.period

        # Trend: linear + learned moving-average smoothing
        intercept = params[:, :, 0]
        slope = params[:, :, 1]
        trend_raw = intercept + slope * time_idx

        # Map logit -> effective window width in [3, max_w_model] per series.
        # The width enters through a *soft* boxcar kernel so gradients flow back
        # to the window parameter; a hard int(median(...).item()) cut would
        # sever the autograd graph, giving zero gradient AND zero Hessian, and
        # LightGBM would grow zero-valued trees for it (the window would stay
        # frozen at its sigmoid(0) midpoint forever).
        max_w_model = min(2 * m + 1, 101)
        w_logit = params[:, :, 2]
        w_eff = (max_w_model - 3.0) * torch.sigmoid(w_logit.mean(dim=0)) + 3.0  # (N,)

        # Kernel support: reflect padding requires pad <= T - 1, so cap the
        # support at 2T - 1 (short forecast horizons used to crash here when
        # W // 2 exceeded T - 1). Both arguments are odd, so K stays odd.
        K = min(max_w_model, 2 * T - 1)

        if K >= 3:
            w_eff = torch.clamp(w_eff, max=float(K))
            half = K // 2
            offsets = torch.arange(-half, half + 1, dtype=dtype).abs().view(1, -1)  # (1,K)
            # Soft boxcar: weight ~ 1 inside +-w_eff/2, smoothly decaying outside.
            k = torch.sigmoid(w_eff.view(-1, 1) / 2.0 - offsets)  # (N,K)
            k = (k / k.sum(dim=1, keepdim=True)).unsqueeze(1)  # (N,1,K)

            # Grouped conv expects channels divisible by groups.
            # Put series in the *channel* dimension: input (1, N, T), weight (N, 1, K), groups=N.
            xin = trend_raw.T.contiguous().unsqueeze(0)  # (1,N,T)
            xpad = torch.nn.functional.pad(xin, (half, half), mode="reflect")  # (1,N,T+2*half)
            trend = torch.nn.functional.conv1d(xpad, k, groups=N).squeeze(0).T  # (T,N)
        else:
            # Series too short to smooth (T == 1); keep the raw linear trend.
            trend = trend_raw

        # Seasonality: Fourier with per-cycle zero-mean centering
        H = (self.n_params - 3) // 2
        if H <= 0:
            seasonality = torch.zeros_like(trend)

            return trend, seasonality

        wsin = params[:, :, 3:3 + H]  # (T,N,H)
        wcos = params[:, :, 3 + H:3 + 2 * H]  # (T,N,H)

        k_h = torch.arange(1, H + 1, dtype=dtype).view(1, 1, H)  # (1,1,H)
        angle = time_idx.unsqueeze(-1) * k_h * (2.0 * torch.pi / m)  # (T,N,H)
        seasonality = (wsin * torch.sin(angle) + wcos * torch.cos(angle)).sum(dim=-1)  # (T,N)

        # Per-cycle centering (sum over a cycle ≈ 0)
        C = (T + m - 1) // m
        pad_T = C * m - T
        if pad_T > 0:
            # Extend by reflection. When the series is shorter than the padding
            # (T < pad_T, i.e. less than half a seasonal cycle observed), keep
            # ping-ponging the reflection until a full cycle can be assembled.
            tail = torch.flip(seasonality, dims=[0])
            while tail.shape[0] < pad_T:
                tail = torch.cat([tail, torch.flip(tail, dims=[0])], dim=0)
            S_ext = torch.cat([seasonality, tail[:pad_T]], dim=0)  # (C*m,N)
        else:
            S_ext = seasonality

        S_mcN = S_ext.view(C, m, N).transpose(0, 1).contiguous()  # (m,C,N)
        S_mcN = S_mcN - S_mcN.mean(dim=0, keepdim=True)  # zero-mean per cycle
        seasonality = S_mcN.transpose(0, 1).reshape(C * m, N)[:T, :]  # (T,N)

        return trend, seasonality

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
    ) -> TrainingResult:
        """
        Train the Hyper-Tree-STL model on time series data.

        This method:
        1. Preprocesses the time series data to create time features
        2. Sets up LightGBM datasets
        3. Trains the model using gradient boosting

        The training data must contain columns:
        - 'series_id': Identifier for each time series
        - 'date': Timestamp for each observation
        - 'value': Target value to forecast
        - 'time': Integer time index (e.g., 1, 2, ..., T) used for the Fourier basis
        - Additional feature columns used for forecasting

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters like 'learning_rate', 'num_leaves', etc.
        num_iterations : int
            Number of boosting rounds for training
        train_data : pd.DataFrame
            Training data containing series_id, date, value and feature columns
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
        if early_stopping_round is not None and (not isinstance(early_stopping_round, int) or early_stopping_round <= 0):
            raise ValueError("early_stopping_round must be a positive integer.")
        if not isinstance(validation, bool):
            raise TypeError("validation must be a boolean.")
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean.")
        if early_stopping_round is not None and not validation:
            raise ValueError("early_stopping_round can only be used when validation is True.")
        if validation and early_stopping_round is None:
            raise ValueError("early_stopping_round must be provided when validation is True.")

        # Reset state for re-training
        self.model = None
        self.dataset_references = {}
        self.is_trained = False
        self.features = None

        if deterministic:
            lgb_params = {**lgb_params, "deterministic": True, "force_row_wise": True}

        # Check required columns
        required_columns = ['series_id', 'date', 'time', 'value']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")

        # Validate row ordering: dates within the single series must be monotonic.
        validate_series_order(train_data, name="train_data")

        # Series Meta Data
        self.n_series = train_data['series_id'].nunique()
        if self.n_series > 1:
            raise NotImplementedError(f"You have provided {self.n_series} series. Currently, HyperTreeSTL only supports univariate training (1 series at a time). Please train separate models for each series.")
        self.train_series_id = train_data['series_id'].unique()[0]

        # General model parameters. The objective wrapper stops lgb.train's
        # params deepcopy from cloning this instance (see NoDeepcopyObjective).
        self.lgb_params = {
            "num_class": self.n_params,
            "objective": NoDeepcopyObjective(self.objective_fn),
            "metric": "None",
            "random_seed": seed,
            "verbose": verbose
        }

        # Update with user-provided LightGBM parameters
        self.lgb_params.update(lgb_params)

        try:
            # Initialize TimeSeriesPreprocessor for creating time features
            preprocessor = TimeSeriesPreprocessor(
                freq=self.freq,
                lags=[],  # STL doesn't need lag features
            )

            # Process full dataset to create time features
            full_ts = preprocessor.create_lags(train_data)
            full_dict = preprocessor.extract(full_ts)

            # Store feature names for later use
            self.features = full_dict["features"].columns.tolist()

            # Prepare datasets
            (valid_sets,
             valid_names,
             callbacks,
             evals_result,
             _,  # No lags for STL
             _,  # No lags for STL
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

            # Use the user-provided 'time' column directly so that training
            # and forecasting use the same Fourier basis.
            if validation:
                idx_eval = train_data.groupby("series_id").tail(self.fcst_h)
                self.time_idx_eval = torch.tensor(
                    idx_eval["time"].values,
                    dtype=self.dtype
                ).reshape(-1, self.n_series)

                idx_train = train_data[~train_data.index.isin(idx_eval.index)]
                self.time_idx_train = torch.tensor(
                    idx_train["time"].values,
                    dtype=self.dtype
                ).reshape(-1, self.n_series)
            else:
                self.time_idx_train = torch.tensor(
                    train_data["time"].values,
                    dtype=self.dtype
                ).reshape(-1, self.n_series)

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

            # Set trained flag to True
            self.is_trained = True

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
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast"
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast STL parameters for each test point
        2. Calculates trend and seasonal components
        3. Combines components to generate forecasted values

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the STL parameters used for forecasting
            - "components": Return the decomposed trend and seasonal components

        Returns
        -------
        pd.DataFrame
            Forecasted data with columns:
            - series_id: Identifier for each time series
            - date: Forecast date/time
            - fcst: Forecasted value (if type="forecast")
            - model: Model name identifier
            - trend, seasonality: Component values (if type="components")
            - trend_intercept, trend_slope, trend_window_logit (default only),
              seasonal_sine{i}, seasonal_cosine{i}: Parameter values (if type="parameters")
        """
        # Check if model is trained
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained. Call train() before forecasting.")

        # Validate input data
        required_cols = ['series_id', 'date', 'time']
        for col in required_cols:
            if col not in test_data.columns:
                raise ValueError(f"Required column '{col}' not found in test_data")

        # Validate row ordering: dates within the single series must be monotonic.
        validate_series_order(test_data, name="test_data")

        # Validate series ID matches training
        test_series_ids = test_data["series_id"].unique()
        if len(test_series_ids) != 1 or test_series_ids[0] != self.train_series_id:
            raise ValueError(
                f"test_data series_id must match the training series_id "
                f"({self.train_series_id}). Got: {test_series_ids.tolist()}"
            )

        # Validate rows per series matches forecast horizon (forecast only;
        # components/parameters can be requested for arbitrary-length input).
        if type == "forecast":
            rows_per_series = test_data.groupby("series_id", sort=False).size()
            bad = rows_per_series[rows_per_series != self.fcst_h]
            if not bad.empty:
                raise ValueError(
                    f"Each series must have exactly fcst_h={self.fcst_h} rows in test_data. "
                    f"Series with wrong counts: {bad.to_dict()}"
                )

        # Check that all features used during training exist in test_data
        missing_features = [f for f in self.features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in test_data: {missing_features}")

        # Validate type parameter
        if type not in ["forecast", "parameters", "components"]:
            raise ValueError("Parameter 'type' must be either 'forecast', 'parameters', or 'components'")

        # Number of series in the test data
        n_series_test = test_data['series_id'].nunique()

        try:
            # Get STL parameter forecasts from the LightGBM model
            params_fcst = torch.tensor(
                self.model.predict(
                    test_data[self.features]
                ).reshape(-1, n_series_test, self.n_params, order="F"),
                dtype=self.dtype
            )

            time_idx = torch.tensor(test_data["time"].to_numpy().reshape(-1, n_series_test), dtype=self.dtype)

            # Forward pass to calculate trend and seasonal components
            trend, seasonality = self._forward(params_fcst, time_idx)

            # Combine components to get forecasted values
            fcsts_stl = trend + seasonality

            # Create output dataframe based on requested type
            if type == "forecast":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": fcsts_stl.detach().numpy().flatten(),
                    "model": f"Hyper-Tree-STL(period={self.period})",
                })
            elif type == "components":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "trend": trend.detach().numpy().flatten(),
                    "seasonality": seasonality.detach().numpy().flatten(),
                    "model": f"Hyper-Tree-STL(period={self.period})",
                })
            elif type == "parameters":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": f"Hyper-Tree-STL(period={self.period})",
                })
                out_df["trend_intercept"] = params_fcst[:,:, 0].detach().numpy().flatten()
                out_df["trend_slope"] = params_fcst[:,:, 1].detach().numpy().flatten()
                # "default" type has a window logit at index 2; "paper" starts seasonality at 2
                seasonal_offset = 3 if self.forward_type == "default" else 2
                if self.forward_type == "default":
                    out_df["trend_window_logit"] = params_fcst[:,:, 2].detach().numpy().flatten()
                for i in range(self.num_seasonal_components):
                    out_df[f"seasonal_sine{i+1}"] = params_fcst[:,:, seasonal_offset + i].detach().numpy().flatten()
                    out_df[f"seasonal_cosine{i+1}"] = params_fcst[:,:, seasonal_offset + self.num_seasonal_components + i].detach().numpy().flatten()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
