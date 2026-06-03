import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Tuple, Callable, Optional, List
import time
from ..utils import CustomLogger
lgb.register_logger(CustomLogger())

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, GaussNewtonHessian
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)

class HyperTreeAR:
    """
    Class that implements a Hyper-Tree-AR(p) model for time series forecasting.

    The Hyper-Tree-AR(p) model extends traditional autoregressive models by allowing
    the AR coefficients to be time-varying and estimated by gradient boosted trees.
    This creates a non-linear, adaptive autoregressive model that can capture complex
    temporal dependencies.

    Key features:
    - Combines tree-based models (LightGBM) with autoregressive time series modeling
    - Allows AR coefficients to vary based on features
    - Provides AR coefficients that can vary over time

    Use this model when:
    - You have relevant features that might influence the autoregressive structure
    - You want more flexibility than traditional AR models

    Example usage:
    ```python
    # Imports
    from hypertrees.models import HyperTreeAR
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 12
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeAR(p=lag_p, freq=frequency, fcst_h=fcst_h)

    # Data
    # The data needs to have the following columns: 'date', 'series_id', 'value'. All other columns are automatically treated as features.
    # You don't have to add lag-values yourself, this happens automatically during training.
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
            p: int = 2,
            freq: str = "M",
            fcst_h: int = 1,
            loss_fn: Callable = nn.MSELoss(),
            hessian_method: str = "exact",
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-Tree-AR(p) model.

        Arguments
        ----------
        p : int
            Maximum number of AR(p) lags. Must be a positive integer.
        freq : str
            Frequency of the time series (e.g., 'D' for daily, 'M' for monthly,
            'Q' for quarterly, 'Y' for yearly).
        fcst_h : int
            Forecast horizon (number of periods to forecast ahead).
        loss_fn : Callable
            Loss function for optimization. Must be a PyTorch loss function.
            Default is MSE loss, but can be changed for different error metrics.
        hessian_method : str
            Method for computing the Hessian diagonal. Options:
            - "exact": Exact diagonal Hessian via per-parameter second-order autograd.
            - "gn": Gauss-Newton approximation estimated via Hutchinson probing.
              Guarantees positive semi-definite Hessians. Avoids second-order
              differentiation at the cost of Hutchinson estimation variance.
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            Only used when hessian_method="gn". More probes reduce variance but
            increase computation. Default is 5.
        """
        # Validate inputs
        if p <= 0:
            raise ValueError("Parameter 'p' must be a positive integer.")
        if fcst_h <= 0:
            raise ValueError("Forecast horizon 'fcst_h' must be a positive integer.")
        if not isinstance(freq, str):
            raise TypeError("freq must be a string.")
        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a PyTorch loss function.")
        if hessian_method not in ("exact", "gn"):
            raise ValueError("hessian_method must be either 'exact' or 'gn'.")
        if not isinstance(n_hessian_probes, int) or n_hessian_probes <= 0:
            raise ValueError("n_hessian_probes must be a positive integer.")

        if hessian_method == "gn" and not isinstance(loss_fn, nn.MSELoss):
            warnings.warn(
                f"Loss {type(loss_fn).__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )

        self.p = p
        self.freq = freq
        self.fcst_h = fcst_h
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.dtype = torch.float32
        self.model = None
        self.features = None  # Stores feature names after training
        self.is_trained = False  # Flag to track if model has been trained
        self.dataset_references = {} # Store references to LightGBM datasets
        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
        self._iter_count = 0
        self._fit = None
        self._target = None

        # Conformal prediction interval state (populated when train() is called
        # with forecast_intervals).
        self._is_calibrated = False
        self._cs_scores = None          # conformity scores (n_windows, n_series, fcst_h)
        self._cs_series_order = None    # series order along axis 1 of _cs_scores
        self._pi_config = None          # ForecastIntervals configuration

        # Bind Hessian computation strategy
        if hessian_method == "exact":
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_exact
        else:
            self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_gn

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts the raw LightGBM outputs to
        PyTorch tensors, computes the loss, and then backpropagates to get gradients.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the AR coefficients.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        self._iter_count += 1

        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=self.dtype)
        params, loss = self.get_params_loss(predt, target, self.lags_train, requires_grad=True)
        grad, hess = self.calculate_gradients_and_hessians(loss, params)

        return grad, hess


    def eval_fn(self, predt: np.ndarray, eval_data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        Custom evaluation function for evaluating forecast accuracy on an evaluation dataset.

        This function computes the loss value to be monitored during evaluation.

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
        # Use appropriate lags based on dataset name
        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        if dataset_name == "train":
            lags = self.lags_train
        elif dataset_name == "validation":
            lags = self.lags_eval
        else:
            # Default to training lags if unknown
            lags = self.lags_train
            warnings.warn("Unknown dataset in metric_fn. Using training lags.")

        # Calculate loss
        is_higher_better = False # Lower loss is better, so we don't maximize
        target = torch.tensor(eval_data.get_label().reshape(-1, 1), dtype=self.dtype)
        _, loss = self.get_params_loss(predt, target, lags)

        return self.loss_name, loss.item(), is_higher_better

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            lags: torch.Tensor = None,
            requires_grad: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into AR parameters and calculate loss.

        This function:
        1. Reshapes the raw outputs into AR parameters
        2. Multiplies these parameters with the lag values
        3. Computes the forecast by summing the weighted lags
        4. Calculates the loss between forecasts and actual values

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM.
        target : torch.Tensor
            Target values (actual time series values).
        lags : torch.Tensor
            Lagged values of the time series.
        requires_grad : bool
            Whether to compute gradients (True during training).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Parameters tensor and loss value.
        """
        # Reshape outputs into parameter matrix (samples × n_params)
        # The 'F' order means Fortran-style ordering (column-major)
        params = nn.Parameter(
            torch.tensor(
                predt.reshape(-1, self.p, order="F"),
                dtype=self.dtype
            ),
            requires_grad=requires_grad
        )

        # Forward pass: Compute forecasts by multiplying parameters with lags and summing
        fcst = torch.sum(params * lags, dim=1, dtype=torch.float32).unsqueeze(1)

        # Calculate loss between forecasts and actual values
        loss = self.loss_fn(fcst, target)

        if self.hessian_method == "gn":
            self._fit = fcst
            self._target = target

        return params, loss

    def _calculate_gradients_and_hessians_exact(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Exact diagonal Hessian via per-parameter second-order autograd."""
        loss.backward(create_graph=True)
        grad = params.grad
        hess = [
            autograd(grad[:, i].sum(), params, retain_graph=True)[0][:, i:(i + 1)]
            for i in range(self.p)
        ]

        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).cpu().detach().numpy().ravel(order="F")
        params.grad = None

        return grad, hess

    def _calculate_gradients_and_hessians_gn(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian diagonal estimated via Hutchinson probing."""
        grad = autograd(loss, params, retain_graph=True)[0]
        rng = torch.Generator().manual_seed(self._iter_count)
        hess = self._gn_hessian.estimate(self._fit, self._target, params, rng)
        self._fit = None
        self._target = None
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
        Train the Hyper-Tree-AR model on time series data.

        This method:
        1. Preprocesses the time series data to create lag features
        2. Sets up LightGBM datasets
        3. Trains the model using gradient boosting

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
            Training data containing series_id, date, value and feature columns
        validation : bool
            If True, a validation set will be created for evaluation. It splits the last fcst_h values of each
            series for validation.
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
        if early_stopping_round is not None and (not isinstance(early_stopping_round, int) or early_stopping_round <= 0):
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

        # Check required columns
        required_columns = ['series_id', 'date', 'value']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")

        # Validate row ordering: each series must be a contiguous block with
        # monotonic dates so the training reshape and fcst_lags extraction align.
        validate_series_order(train_data, name="train_data")

        # Fail fast if any series is too short for the requested conformal calibration.
        # An AR(p) model needs at least p + 1 rows to retain one training sample.
        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals, min_train=self.p + 1
            )

        # General model parameters
        self.lgb_params = {
            "num_class": self.p,
            "objective": self.objective_fn,
            "metric": "None",
            "random_seed": seed,
            "verbose": verbose
        }

        # Update with user-provided LightGBM parameters
        self.lgb_params.update(lgb_params)

        # Reset state for re-training
        self._iter_count = 0
        self._fit = None
        self._target = None
        self.model = None
        self.dataset_references = {}
        self.is_trained = False
        self.features = None
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        try:
            # Initialize TimeSeriesPreprocessor for creating lagged dataframe
            preprocessor = TimeSeriesPreprocessor(
                freq=self.freq,
                lags=[i for i in range(1, self.p + 1)],
            )

            # Process full dataset to create lagged dataframe
            full_ts = preprocessor.create_lags(train_data)
            full_dict = preprocessor.extract(full_ts)

            # Store feature names for later use
            self.features = full_dict["features"].columns.tolist()

            # Prepare datasets
            (valid_sets,
             valid_names,
             callbacks,
             evals_result,
             lags_train,
             lags_eval,
             self.dataset_references) = (
                prepare_datasets(
                    full_ts=full_ts,
                    preprocessor=preprocessor,
                    fcst_h=self.fcst_h,
                    dtype=self.dtype,
                    validation=validation,
                    early_stopping_round=early_stopping_round
                )
            )

            # Store lagged values for training and evaluation
            self.lags_train = lags_train
            self.lags_eval = lags_eval

            # Store lagged train values to be used in the forecast method
            self.fcst_lags = (
                train_data.groupby(["series_id"], sort=False)
                .apply(lambda x: x["value"][-self.p:][::-1].values)
                .to_dict()
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

            # Set trained flag to True
            self.is_trained = True

            # Calibrate conformal prediction intervals via rolling-window CV.
            # Fresh model instances are trained per window (no forecast_intervals
            # passed, so there is no recursion) using the same hyper-parameters.
            if forecast_intervals is not None:
                def _model_factory():
                    return HyperTreeAR(
                        p=self.p,
                        freq=self.freq,
                        fcst_h=self.fcst_h,
                        loss_fn=self.loss_fn,
                        hessian_method=self.hessian_method,
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

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast AR coefficients for each test point
        2. Recursively generates forecasts using the forecasted AR coefficients

        The forecasting process implements an autoregressive model where:
        y_t = φ₁(x)y_{t-1} + φ₂(x)y_{t-2} + ... + φₚ(x)y_{t-p}

        However, unlike traditional AR models, the φ(x) coefficients are not constant
        but determined by the LightGBM model based on features x.

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the AR(p) coefficients used for forecasting
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
            - AR(i): AR coefficient values (if type="parameters")
            - <model>-lo-<level> / <model>-hi-<level>: prediction interval bounds
              (if type="forecast" and level is provided)
        """
        # Check if model is trained
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained. Call train() before forecasting.")

        # Validate input data
        required_cols = ['series_id', 'date']
        for col in required_cols:
            if col not in test_data.columns:
                raise ValueError(f"Required column '{col}' not found in test_data")

        # Validate row ordering: each series must be a contiguous block with
        # monotonic dates so the forecast reshape aligns forecasts with lags.
        validate_series_order(test_data, name="test_data")

        # Validate series IDs match training data
        test_series_ids = test_data["series_id"].unique()
        train_series_ids = set(self.fcst_lags.keys())
        missing = set(test_series_ids) - train_series_ids
        extra = train_series_ids - set(test_series_ids)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"Missing series in training: {missing}")
            if extra:
                parts.append(f"Extra series not in test_data: {extra}")
            raise ValueError(". ".join(parts))

        # Validate rows per series matches fcst_h (forecast only; parameters
        # can be requested for arbitrary-length input).
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
        if type not in ["forecast", "parameters"]:
            raise ValueError("Parameter 'type' must be either 'forecast' or 'parameters'")

        # Validate conformal interval request
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

            if type == "forecast":
                # Get AR parameter forecasts from the LightGBM model
                # Shape: (n_series, fcst_h, n_params)
                n_series_test = len(test_series_ids)
                params_fcst = self.model.predict(test_data[self.features]).reshape(n_series_test, self.fcst_h, self.p)

                # Reconstruct lags array in the same order as test data
                lags = np.array([self.fcst_lags[series_id] for series_id in test_series_ids])

                # Generate multi-step forecasts
                forecasts = []
                for h in range(self.fcst_h):
                    # Compute next value using AR equation: y_t = φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p}
                    next_val = np.sum(params_fcst[:, h, :] * lags, axis=1).reshape(-1, 1)
                    forecasts.append(next_val)

                    # Update lags for next step by adding new forecast and removing oldest lag
                    lags = np.concatenate([next_val, lags[:, :-1]], axis=1)

                # Create output dataframe based on requested type
                model_name = f"Hyper-Tree-AR({self.p})"
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": np.hstack(forecasts).flatten(),
                    "model": model_name,
                })

                # Append conformal prediction intervals if requested.
                if level is not None:
                    point = np.hstack(forecasts)  # (n_series_test, fcst_h)
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
                params_fcst = np.asarray(self.model.predict(test_data[self.features]))
                # LightGBM may return 1D (column-major) or 2D depending on version/objective.
                # Normalize to (n_test, p) before indexing.
                if params_fcst.ndim == 1:
                    params_fcst = params_fcst.reshape(-1, self.p, order="F")
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": f"Hyper-Tree-AR({self.p})",
                })
                # Add AR parameters to the dataframe
                for i in range(self.p):
                    out_df[f"AR({i + 1})"] = params_fcst[:, i].flatten()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}")
