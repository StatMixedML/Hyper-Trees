import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Tuple, List, Callable, Optional
import time
import random
import warnings
from ..utils import CustomLogger
lgb.register_logger(CustomLogger())

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, GaussNewtonHessian, NoDeepcopyObjective
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)

class HyperTreeTSB:
    """
    Class that implements a Hyper-Tree-TSB model for intermittent demand forecasting.

    The Teunter-Syntetos-Babai (TSB) method forecasts intermittent demand as the
    product of two exponentially smoothed components: the demand *probability*
    ``p_t`` (smoothed occurrence indicator, updated every period) and the demand
    *size* ``z_t`` (smoothed over nonzero demands only). The Hyper-Tree variant
    makes both smoothing parameters time-varying functions of features, so the
    responsiveness of probability and size estimates can adapt to e.g.
    promotions, listings, or seasonality. The recursion follows the reference
    implementation in Nixtla's statsforecast:

    - Occurrence: d_t = 1 if y_t != 0 else 0
    - Probability: p_t = p_{t-1} + alpha_p,t * (d_t - p_{t-1})
    - Size: z_t = z_{t-1} + alpha_d,t * (y_t - z_{t-1}) if d_t = 1 else z_{t-1}
    - Fitted: y_hat_t = p_{t-1} * z_{t-1}
    - Forecast: y_hat_{T+h} = p_T * z_T (flat over the horizon, as in the
      classical method: future demand occurrence is unobserved and the expected
      states propagate unchanged)

    States are initialized as in statsforecast: ``p_0`` with the first
    occurrence indicator and ``z_0`` with the first nonzero demand (0 for
    all-zero series, which therefore forecast 0). For cross-series learning,
    all series must have the same length; datasets with varying lengths should
    be padded and carry a ``mask`` column (1 = valid observation, 0 = padding).

    Key features:
    - Designed for intermittent (sporadic, zero-inflated) demand
    - Combines tree-based models (LightGBM) with the TSB method
    - Allows the smoothing parameters to vary based on features
    - Handles obsolescence: the probability estimate decays during zero-demand periods

    Use this model when:
    - Series contain frequent zeros (intermittent demand), where AR and ETS
      target models are structurally misspecified
    - You have features that signal when demand probability or size shifts

    References
    ----------
    [1] Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent
        demand: Linking forecasting to inventory obsolescence. European
        Journal of Operational Research, 214(3), 606-615.
    [2] Recursion and state-initialization conventions follow the TSB
        implementation in Nixtla's statsforecast (Apache-2.0):
        https://github.com/Nixtla/statsforecast

    Example usage:
    ```python
    # Imports
    from hypertrees.models.HyperTreeTSB import HyperTreeTSB
    import numpy as np
    import pandas as pd

    # Initialize model
    frequency = 'W'
    fcst_h = 8
    model = HyperTreeTSB(freq=frequency, fcst_h=fcst_h)

    # Data: intermittent demand with columns 'date', 'series_id', 'value'.
    # All other columns are automatically treated as features.
    rng = np.random.RandomState(1)
    dates = pd.date_range("2022-01-03", periods=104 + fcst_h, freq="W-MON")
    demand = rng.binomial(1, 0.3, len(dates)) * rng.poisson(5, len(dates))
    df = pd.DataFrame({
        "series_id": "sku_1",
        "date": dates,
        "value": demand.astype(float),
        "month": dates.month,
    })
    test = df.tail(fcst_h)
    train = df.drop(test.index)

    # Train model
    model.train(
        lgb_params={'learning_rate': 0.1},
        num_iterations=100,
        train_data=train
    )

    # Generate forecasts and inspect the time-varying smoothing parameters
    forecasts = model.forecast(test_data=test)
    parameters = model.forecast(test_data=test, type="parameters")
    ```
    """

    def __init__(
            self,
            freq: str = "M",
            fcst_h: int = 12,
            loss_fn: Callable = nn.MSELoss(),
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-Tree-TSB model.

        Arguments
        ----------
        freq : str
            Frequency of the time series (e.g., 'D' for daily, 'W' for weekly,
            'M' for monthly).
        fcst_h : int
            Forecast horizon (number of periods to forecast ahead).
        loss_fn : Callable
            Loss function for optimization. Must be a PyTorch loss function.
            Default is MSE loss. Losses other than nn.MSELoss are not
            recommended, as they have not been systematically tested yet.
            nn.L1Loss is rejected (zero second derivative almost everywhere
            breaks Newton boosting).
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            More probes reduce variance but increase computation. Default is 5.
        """
        # Validate inputs
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
        if getattr(loss_fn, "reduction", "mean") == "none":
            raise ValueError(
                "loss_fn must use a scalar reduction ('mean' or 'sum'); "
                "reduction='none' returns per-element losses that the "
                "boosting objective cannot consume."
            )
        if not isinstance(loss_fn, nn.MSELoss):
            warnings.warn(
                f"Loss {type(loss_fn).__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )
        if not isinstance(freq, str):
            raise TypeError("freq must be a string representing the frequency of the time series.")

        self.freq = freq
        self.n_params = 2              # alpha_p (probability), alpha_d (demand size)
        self.fcst_h = fcst_h
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.dtype = torch.float32
        self.model = None
        self.features = None            # Stores feature names after training
        self.is_trained = False         # Flag to track if model has been trained
        self.dataset_references = {}    # Store references to LightGBM datasets
        self.eps = 1e-6                 # Small constant to prevent numerical issues in sigmoid
        self.fcst_states = None         # Store final TSB states for forecasting
        self.n_hessian_probes = n_hessian_probes
        self._iter_count = 0            # Iteration counter for seeding Hessian probes
        # Recursive h-step validation metric: the terminal (p, z) states from
        # the "train" eval call are stashed in _eval_boundary and consumed by
        # the "validation" eval call of the same boosting iteration
        # (valid_sets order is [train, validation]); see eval_fn.
        self._last_states = None
        self._eval_boundary = None

        # Shared Gauss-Newton Hessian estimator
        self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)

        # Conformal prediction interval state
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        # Activation function for parameter bounds
        self.sigmoid_fn = nn.Sigmoid()

    def _create_mask_from_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Create a mask for valid observations from the data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the time series data

        Returns
        -------
        torch.Tensor
            Mask indicating valid observations (1 = valid, 0 = padding).
        """
        if 'mask' in data.columns:
            mask = torch.tensor(
                data['mask'].values.reshape(self.n_series, -1),
                dtype=self.dtype
            )
        else:
            data_shape = data.shape[0]
            mask = torch.ones((data_shape, 1), dtype=self.dtype).reshape(self.n_series, -1)

        return mask

    def _init_states(
            self,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initial probability and demand-size states.

        Follows statsforecast's TSB: the probability state starts at the first
        (valid) occurrence indicator and the size state at the first (valid)
        nonzero demand. All-zero series get a size state of 0 and therefore
        forecast 0. The initialization depends only on the data, never on the
        boosted parameters.

        Parameters
        ----------
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding),
            shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Initial probability ``p_0`` and size ``z_0``, each ``(n_series,)``.
        """
        occurrence = ((target != 0) & (mask > 0)).to(self.dtype)

        # p_0: occurrence indicator at the first valid observation
        first_valid = torch.argmax((mask > 0).to(torch.long), dim=1)
        p0 = occurrence.gather(1, first_valid.unsqueeze(1)).squeeze(1)

        # z_0: first valid nonzero demand (0 if the series is all zeros)
        has_demand = occurrence.any(dim=1)
        first_demand = torch.argmax(occurrence.to(torch.long), dim=1)
        z0 = target.gather(1, first_demand.unsqueeze(1)).squeeze(1)
        z0 = torch.where(has_demand, z0, torch.zeros_like(z0))

        return p0, z0

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts LightGBM outputs to
        PyTorch tensors, computes the TSB forward pass, and then backpropagates
        to get gradients.

        Parameters
        ----------
        predt : np.ndarray
            Outputs from LightGBM, representing the TSB smoothing parameters.
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

        Notes
        -----
        The validation metric is the **recursive h-step forecast** loss: the
        classical flat TSB forecast ``p_T * z_T`` (from the training-window
        terminal states) scored against the holdout. The naive in-sample
        validation loss is degenerate -- the TSB fitted value at every
        validation row collapses to a parameter-independent fixed point, so it
        is ~0 for *any* parameters and early stopping selects noise. The
        training metric remains the in-sample one-step fit.
        """
        is_higher_better = False  # Lower loss is better, so we don't maximize
        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        target = torch.tensor(
            eval_data.get_label().reshape(self.n_series, -1),
            dtype=self.dtype
        )

        if dataset_name == "validation" and self._eval_boundary is not None:
            # Recursive h-step forecast metric. The terminal states were stashed
            # during this same iteration's "train" eval call, so they come from
            # the identical (post-update) model state.
            loss = self._recursive_eval_loss(eval_data, target)
            loss_val = loss.item()
            if not np.isfinite(loss_val):
                # A diverged state early in boosting would otherwise feed NaN to
                # early stopping; report a large finite value (worst) instead.
                loss_val = float(np.finfo(np.float32).max)
            return self.loss_name, loss_val, is_higher_better

        # Train metric (and validation fallback before the first boundary is
        # stashed): the teacher-forced in-sample one-step loss.
        _, loss = self.get_params_loss(predt, target, eval_data)
        if dataset_name == "train":
            # Stash terminal states for the validation rollout that follows in
            # this same iteration.
            self._eval_boundary = self._last_states

        return self.loss_name, loss.item(), is_higher_better

    def _recursive_eval_loss(
            self,
            eval_data: lgb.Dataset,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Recursive h-step forecast loss for the validation split.

        Mirrors deployment: the flat TSB forecast ``p_T * z_T`` (via the shared
        :meth:`_roll_forecast` helper), starting from the training-window
        terminal states stored in ``self._eval_boundary``, is scored against
        the holdout. The forecast is independent of the horizon parameters, as
        in the classical method. Padded holdout rows (mask == 0) are excluded.

        Parameters
        ----------
        eval_data : lgb.Dataset
            Validation dataset (provides the mask).
        target : torch.Tensor
            Holdout observations, shape ``(n_series, fcst_h)``.

        Returns
        -------
        torch.Tensor
            Scalar loss between the flat forecast and the holdout.
        """
        last_p, last_z = self._eval_boundary
        point = self._roll_forecast(last_p, last_z, target.shape[1])

        if "mask" in eval_data.data.columns:
            mask = torch.tensor(
                eval_data.data["mask"].values.reshape(self.n_series, -1),
                dtype=self.dtype,
            )
        else:
            mask = torch.ones_like(target)

        return self.loss_fn(point * mask, target * mask)

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            data: lgb.Dataset = None,
            requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into TSB parameters and calculate loss.

        This function:
        1. Reshapes the raw outputs into TSB parameters
        2. Applies sigmoid transformation to ensure parameter bounds
        3. Runs the TSB forward pass to compute fitted values
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
        Tuple[torch.Tensor, torch.Tensor]
            Parameters tensor and loss value.
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

        # Apply sigmoid transformation and reshape for TSB computation; clamp to avoid numerical issues
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

        # Forward pass to compute fitted values. Keep the terminal probability/
        # size states so the recursive validation metric can roll the flat
        # deployment forecast from the training-window boundary (see eval_fn).
        last_p, last_z, fit = self.forward(params, target, mask)
        self._last_states = (last_p, last_z)

        # Stack fitted values and compute loss with masking
        fit = torch.stack(fit, dim=1)
        loss = self.loss_fn(fit * mask, target * mask)

        # Store for Gauss-Newton Hessian estimation
        self._fit = fit
        self._mask = mask
        self._target = target

        return predt, loss

    def forward(
            self,
            params: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the TSB recursion.

        This implements the TSB updates:
        - Probability: p_t = p_{t-1} + α_p(d_t - p_{t-1}), updated every period
        - Size: z_t = z_{t-1} + α_d(y_t - z_{t-1}) if demand occurs, else unchanged
        - Fitted: ŷ_t = p_{t-1} * z_{t-1}

        The occurrence indicator ``d_t`` is data (constant w.r.t. the
        parameters), so the size-update gating does not break differentiability.

        Parameters
        ----------
        params : torch.Tensor
            Sigmoid-transformed TSB parameters, shape ``(n_series, T, n_params)``.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]
            Final probability state (n_series,), final size state (n_series,),
            and list of fitted values.
        """
        series_len = target.shape[1]

        # Unpack and pre-unbind parameters
        alpha_p, alpha_d = params.unbind(dim=2)
        alpha_p_t = alpha_p.unbind(dim=1)
        alpha_d_t = alpha_d.unbind(dim=1)

        # Occurrence indicators (data, constant w.r.t. the parameters)
        occurrence = ((target != 0) & (mask > 0)).to(self.dtype)

        # Initialize states from the data
        p_prev, z_prev = self._init_states(target, mask)
        fits = [target[:, 0]]

        # Pre-unbind the data tensors so the loop indexes Python tuples
        # instead of slicing tensors at every step.
        target_t = target.unbind(dim=1)
        mask_t = mask.unbind(dim=1)
        occurrence_t = occurrence.unbind(dim=1)

        # TSB updates with masking for padded values.
        for t in range(1, series_len):
            valid_mask = mask_t[t]
            invalid_mask = 1 - valid_mask
            y_t = target_t[t]
            d_t = occurrence_t[t]

            fit_t = valid_mask * (p_prev * z_prev) + invalid_mask * fits[-1]

            p_new = valid_mask * (
                    p_prev + alpha_p_t[t] * (d_t - p_prev)
            ) + invalid_mask * p_prev

            z_new = valid_mask * (
                    d_t * (z_prev + alpha_d_t[t] * (y_t - z_prev)) +
                    (1 - d_t) * z_prev
            ) + invalid_mask * z_prev

            fits.append(fit_t)
            p_prev = p_new
            z_prev = z_new

        return p_prev, z_prev, fits

    def calculate_gradients_and_hessians(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute gradients and Generalized Gauss-Newton Hessians for LightGBM.

        Uses exact first-order gradients and the Generalized Gauss-Newton (GGN)
        approximation for the Hessian diagonal, estimated via Hutchinson probing.
        As for ``HyperTreeETS``, the TSB recurrence makes exact second
        derivatives propagate through the full recursion, so the
        residual-curvature term is dropped, retaining only H_GN = J^T B J,
        which guarantees positive semi-definite Hessians.

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
        Train the Hyper-Tree-TSB model on time series data.

        This method:
        1. Preprocesses the time series data to create features and handle variable lengths
        2. Sets up LightGBM datasets with proper masking
        3. Trains the model using gradient boosting
        4. Stores final TSB states for future forecasting

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
        # monotonic dates so the TSB reshape to (n_series, T, n_params) aligns.
        validate_series_order(train_data, name="train_data")

        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals, min_train=2,
            )

        # Check if all series in train_data have the same length
        unique_lengths = train_data.groupby('series_id')['date'].nunique()
        if len(unique_lengths.unique()) > 1:
            raise ValueError("All series in train_data must have the same length. Found multiple lengths.")

        # General model parameters. The objective wrapper stops lgb.train's
        # params deepcopy from cloning this instance (see NoDeepcopyObjective).
        self.lgb_params = {
            "num_class": self.n_params,
            "objective": NoDeepcopyObjective(self.objective_fn),
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
        self._last_states = None
        self._eval_boundary = None
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
            # Initialize TimeSeriesPreprocessor for TSB-specific preprocessing
            preprocessor = TimeSeriesPreprocessor(
                freq=self.freq,
                lags=[],  # TSB doesn't need lag features like AR models
            )

            # Process dataset, including masking for variable-length series
            full_ts = preprocessor.create_lags(train_data)
            full_dict = preprocessor.extract(full_ts)

            # Store feature names and dataset dimensions
            self.features = full_dict["features"].columns.tolist()
            self.n_series = len(train_data['series_id'].unique())

            # Prepare datasets (adapted for TSB)
            (valid_sets,
             valid_names,
             callbacks,
             evals_result,
             _,  # No lags for TSB
             _,  # No lags for TSB
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

            # Store final TSB states for forecasting
            self._store_final_states(train_data)

            # Set trained flag to True
            self.is_trained = True

            if forecast_intervals is not None:
                def _model_factory():
                    return HyperTreeTSB(
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
                best_iteration=self.model.best_iteration if self.model.best_iteration > 0 else num_iterations,
                training_time=training_time
            )

            return result

        except Exception as e:
            self.is_trained = False
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def _store_final_states(self, train_data: pd.DataFrame):
        """
        Store final TSB states after training for use in forecasting.

        Runs a full forward pass on the training data to obtain the final
        probability and size states per series. Also stores the series
        ordering to ensure consistent state access.

        Parameters
        ----------
        train_data : pd.DataFrame
            Training data used to compute final TSB states.
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

        # Forward pass to get final states
        last_p, last_z, fit = self.forward(params, target, train_mask)

        # Store final states as dictionary with series_id as keys
        self.fcst_states = {}
        for i, series_id in enumerate(self.series_order):
            self.fcst_states[series_id] = {
                'last_p': last_p[i],
                'last_z': last_z[i],
            }

    def set_forecast_origin(self, history: pd.DataFrame) -> None:
        """Re-anchor TSB states to the end of *history* without retraining.

        Recomputes the terminal ``{probability, size}`` states by running the
        full TSB forward recurrence over *history* using the already-trained
        GBDT parameters.  Used by conformal calibration with ``refit=False``.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with the same columns as the training data (including
            any ``mask`` column), ordered by ``(series_id, date)`` with each
            series in a contiguous block and all series of equal length.
        """
        validate_series_order(history, name="history")
        self._store_final_states(history)

    def _roll_forecast(
            self,
            last_p: torch.Tensor,
            last_z: torch.Tensor,
            h: int,
    ) -> torch.Tensor:
        """Classical TSB forecast: flat ``p_T * z_T`` over ``h`` steps.

        Shared by :meth:`forecast` and the recursive validation metric
        (:meth:`_recursive_eval_loss`) so the deployed forecast and the
        early-stopping metric cannot diverge. Future demand occurrence is
        unobserved and the expected states propagate unchanged, so the forecast
        is constant over the horizon and independent of the horizon parameters.

        Parameters
        ----------
        last_p : torch.Tensor
            Terminal probability state, shape ``(n_series,)``.
        last_z : torch.Tensor
            Terminal size state, shape ``(n_series,)``.
        h : int
            Number of horizon steps.

        Returns
        -------
        torch.Tensor
            Point forecasts, shape ``(n_series, h)``.
        """
        return (last_p * last_z).reshape(-1, 1).repeat(1, h)

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        Following the classical TSB method, the point forecast is flat over
        the horizon: ŷ_{T+h} = p_T * z_T, where p_T and z_T are the final
        probability and size states after the training period. Future demand
        occurrence is unobserved, and the expected one-step-ahead forecast
        propagates unchanged, so horizon features do not alter the point
        forecasts (they do affect ``type="parameters"``).

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the TSB smoothing parameters
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
            - alpha_p, alpha_d: TSB parameter values (if type="parameters")
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

            model_name = "Hyper-Tree-TSB"

            if type == "forecast":
                # Extract stored final states in the correct order of the test data series
                test_series_ids = test_data['series_id'].unique()
                last_p = torch.stack([self.fcst_states[series_id]['last_p'] for series_id in test_series_ids])
                last_z = torch.stack([self.fcst_states[series_id]['last_z'] for series_id in test_series_ids])

                # Classical TSB: flat forecast p_T * z_T over the horizon (via
                # the shared recursion, also used by the validation metric).
                point = self._roll_forecast(last_p, last_z, self.fcst_h)

                # Create output dataframe
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": point.flatten().numpy(),
                    "model": model_name,
                })

                if level is not None:
                    columns = interval_columns(
                        point=point.numpy(),
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
                    "model": model_name,
                })
                for i, param_name in enumerate(["alpha_p", "alpha_d"]):
                    out_df[param_name] = fcst_params[:, :, i].flatten().numpy()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
