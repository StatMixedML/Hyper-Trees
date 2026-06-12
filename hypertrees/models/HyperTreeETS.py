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
    - Supports triple exponential smoothing with multiplicative ("triple") or
      additive ("additive") seasonality, as well as trend-only models

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
            seasonal_init: str = "classical",
    ):
        """
        Initialize the Hyper-Tree-ETS model.

        Arguments
        ----------
        ets_type : str
            Type of ETS model to use. Options:
            - "triple": Holt-Winters with *multiplicative* seasonality and a
              damped trend. Requires strictly positive series.
            - "additive": Holt-Winters with *additive* seasonality and a
              damped trend. Use when seasonal swings are roughly constant in
              absolute size, or when series contain zeros or negative values
              (where multiplicative seasonality breaks down).
            - "trend": linear trend-only (no seasonality).
        season_length : int
            Seasonal length of the time series (e.g., 12 for monthly data, 4 for quarterly).
        seasonality_feature : str
            Feature name for seasonality. This is used to create seasonal indices. Must be present in the dataset.
            For example, "month" for monthly data, "quarter" for quarterly data, etc. This is required when
            ets_type is "triple" or "additive". Values must be 1-based season positions in [1, season_length];
            shift 0-based features (e.g. pandas dayofweek in 0..6) by +1.
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
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            More probes reduce variance but increase computation. Default is 5.
        seasonal_init : str
            Initialization of the seasonal/level/trend states for the triple
            ETS forward pass (ignored for ets_type="trend"; ets_type="additive"
            always uses its classical additive estimator). Options:
            - "classical" (default): decomposition-based estimator following
              R's forecast::ets and statsforecast (centered 2 x m moving-average
              detrending, slot-aligned seasonal indices, OLS level/trend seed).
            - "legacy": the exact pre-0.2.0 initialization, kept verbatim for
              reproducing earlier results (including the paper benchmarks).
        """
        # Validate inputs
        if ets_type not in ["triple", "additive", "trend"]:
            raise ValueError("ets_type must be one of 'triple', 'additive', or 'trend'.")
        if seasonal_init not in ["classical", "legacy"]:
            raise ValueError("seasonal_init must be either 'classical' or 'legacy'.")
        if season_length <= 0:
            raise ValueError("season_length must be a positive integer.")
        if not isinstance(season_length, int):
            raise TypeError("season_length must be an integer.")
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
        if seasonality_feature is None and ets_type in ("triple", "additive"):
            raise ValueError(f"seasonality_feature must be provided for {ets_type} ETS type.")

        self.ets_type = ets_type
        self.season_length = season_length
        self.seasonality_feature = seasonality_feature
        self.seasonal_init = seasonal_init
        self.freq = freq
        self.n_params = 4 if ets_type in ("triple", "additive") else 2  # alpha, beta, gamma, phi OR alpha, beta
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
        self._init_cache = {}           # Per-dataset cache of _init_triple_states results
        # Recursive h-step validation metric: the terminal level/trend/seasonality
        # states from the "train" eval call are stashed in _eval_boundary and
        # consumed by the "validation" eval call of the same boosting iteration
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

        # Set the appropriate forward function based on ETS type
        if self.ets_type == "triple":
            self.forward = self._forward_triple
        elif self.ets_type == "additive":
            self.forward = self._forward_additive
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

    def _seasonal_positions(self, values) -> torch.Tensor:
        """Convert 1-based seasonal feature values to 0-based tensor indices.

        Validates values lie in ``[1, season_length]``; a 0-based feature
        (e.g. pandas ``dayofweek``) would silently wrap into the wrong slot.

        Parameters
        ----------
        values : array-like
            Raw values of the ``seasonality_feature`` column.

        Returns
        -------
        torch.Tensor
            0-based seasonal positions as a flat ``torch.long`` tensor.
        """
        idx = torch.tensor(np.asarray(values), dtype=torch.long) - 1
        if idx.numel() > 0:
            lo = int(idx.min())
            hi = int(idx.max())
            if lo < 0 or hi >= self.season_length:
                raise ValueError(
                    f"seasonality_feature '{self.seasonality_feature}' must contain "
                    f"1-based season positions in [1, {self.season_length}]; got values "
                    f"in [{lo + 1}, {hi + 1}]. Shift 0-based features (e.g. pandas "
                    f"dayofweek) by +1."
                )
        return idx

    def _init_triple_states(
            self,
            target: torch.Tensor,
            mask: torch.Tensor,
            seasonality_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial seasonal indices and level/trend states for triple ETS.

        ``seasonal_init="legacy"`` reproduces the pre-0.2.0 initialization
        verbatim (flat seasonal profile), required to reproduce earlier
        results including the paper benchmarks. ``"classical"`` (default)
        follows R's ``forecast::ets`` / statsforecast: centered 2 x m
        moving-average detrending, per-slot ratio averages normalized to mean
        one, and an OLS level/trend seed on the seasonally-adjusted head.
        Indices are assigned via ``seasonality_idxs`` so they land in the
        slots the recursion reads; short series and empty slots fall back to
        a simple per-slot average. All statistics are mask-aware.

        Parameters
        ----------
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding),
            shape ``(n_series, T)``.
        seasonality_idxs : torch.Tensor
            0-based seasonal slot per observation, shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Seasonal indices ``(n_series, season_length)``, initial level
            ``(n_series,)``, and initial trend ``(n_series,)``.
        """
        N, T = target.shape
        m = self.season_length

        if self.seasonal_init == "legacy":
            # Pre-0.2.0 initialization, kept verbatim (including its positional
            # slot assignment and flat resulting profile) so that results from
            # earlier releases and the paper remain reproducible.
            init_len = min(m, T)
            seasonal_avg = torch.zeros((N, init_len), dtype=self.dtype)
            for i in range(init_len):
                valid_obs = mask[:, i::init_len]
                seasonal_avg[:, i] = (target[:, i::init_len] * valid_obs.float()).sum(
                    1) / valid_obs.float().sum(1).clamp(min=1)

            seasonality = target[:, :init_len] / (seasonal_avg + self.eps)
            if init_len < m:
                # Pad with ones for missing season positions
                pad = torch.ones((N, m - init_len), dtype=self.dtype)
                seasonality = torch.cat([seasonality, pad], dim=1)

            season_adj = target[:, :init_len] / seasonality[:, :init_len]
            level0 = season_adj[:, 0]
            trend0 = season_adj[:, min(1, init_len - 1)] - season_adj[:, 0]
            return seasonality, level0, trend0

        slot_ids = torch.arange(N, dtype=torch.long).unsqueeze(1) * m + seasonality_idxs  # (N, T)
        ym = target * mask

        def slot_sum(ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            """Sum ``values`` into their (series, slot) cells -> (N, m)."""
            return torch.zeros(N * m, dtype=self.dtype).index_add_(
                0, ids.reshape(-1), values.reshape(-1)
            ).reshape(N, m)

        # --- Fallback estimator: per-slot average of y over the series mean --
        cnts = slot_sum(slot_ids, mask)
        slot_mean = slot_sum(slot_ids, ym) / cnts.clamp(min=1.0)
        grand_mean = ym.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        simple_idx = torch.where(
            cnts > 0,
            slot_mean / (grand_mean.unsqueeze(1) + self.eps),
            torch.ones_like(slot_mean),
        )
        seasonality = simple_idx

        # --- Detrended estimator: centered 2 x m MA, then per-slot ratios ----
        L = m + 1 if m % 2 == 0 else m  # always odd
        if T >= L:
            kernel = torch.full((1, 1, L), 1.0 / m, dtype=self.dtype)
            if m % 2 == 0:
                kernel[0, 0, 0] = 0.5 / m
                kernel[0, 0, -1] = 0.5 / m
            trend_ma = torch.nn.functional.conv1d(
                ym.unsqueeze(1), kernel
            ).squeeze(1)  # (N, T - L + 1), centered at offset L // 2
            window_valid = torch.nn.functional.conv1d(
                mask.unsqueeze(1), torch.ones((1, 1, L), dtype=self.dtype)
            ).squeeze(1) >= L - 0.5  # full window unmasked (implies a valid center)

            half = L // 2
            y_c = target[:, half:T - half]  # window centers, length T - L + 1
            valid = window_valid & (trend_ma > self.eps)
            ratios = torch.where(
                valid, y_c / trend_ma.clamp(min=self.eps), torch.zeros_like(y_c)
            )

            r_cnts = slot_sum(slot_ids[:, half:T - half], valid.to(self.dtype))
            detr_idx = slot_sum(slot_ids[:, half:T - half], ratios) / r_cnts.clamp(min=1.0)
            seasonality = torch.where(r_cnts > 0, detr_idx, simple_idx)

        # Normalize to mean one and guard against tiny indices
        # (statsforecast clips initial seasonal states at 1e-2 as well).
        seasonality = seasonality / seasonality.mean(dim=1, keepdim=True).clamp(min=self.eps)
        seasonality = seasonality.clamp(min=1e-2)

        # --- Level/trend: OLS on the seasonally-adjusted head (as in ets) ----
        s_t = seasonality.gather(1, seasonality_idxs)
        y_sa = target / s_t.clamp(min=self.eps)
        maxn = min(max(10, 2 * m), T)
        t_idx = torch.arange(1, maxn + 1, dtype=self.dtype).unsqueeze(0)
        w = mask[:, :maxn]
        sw = w.sum(dim=1).clamp(min=1.0)
        mean_t = (w * t_idx).sum(dim=1) / sw
        mean_y = (w * y_sa[:, :maxn]).sum(dim=1) / sw
        dev_t = t_idx - mean_t.unsqueeze(1)
        var_t = (w * dev_t ** 2).sum(dim=1)
        cov_ty = (w * dev_t * (y_sa[:, :maxn] - mean_y.unsqueeze(1))).sum(dim=1)
        trend0 = torch.where(
            var_t > self.eps,
            cov_ty / var_t.clamp(min=self.eps),
            torch.zeros_like(cov_ty),
        )
        # Evaluate the line at the first observation (t = 1 on the OLS axis)
        # so level0 matches this recursion's timing, which seeds the state at
        # the first observation rather than one step before it.
        level0 = mean_y + trend0 * (1.0 - mean_t)

        return seasonality, level0, trend0

    def _cached_init_triple_states(
            self,
            data: lgb.Dataset,
            target: torch.Tensor,
            mask: torch.Tensor,
            seasonality_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache wrapper around :meth:`_init_triple_states`.

        The initialization depends only on the data, which is constant across
        boosting iterations, so results are cached per lgb.Dataset. Entries
        pin the dataset object, so a key hit proves it is that exact,
        still-alive dataset. The seasonal indices are cloned on every return
        because the recursion updates them in place; the cache is cleared by
        ``train()``.

        Parameters
        ----------
        data : lgb.Dataset
            Dataset whose identity serves as the cache key.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.
        seasonality_idxs : torch.Tensor
            0-based seasonal slot per observation, shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Seasonal indices ``(n_series, season_length)``, initial level
            ``(n_series,)``, and initial trend ``(n_series,)``.
        """
        key = id(data)
        entry = self._init_cache.get(key)
        if entry is not None:
            return entry["seasonality"].clone(), entry["level0"], entry["trend0"]

        seasonality, level0, trend0 = self._init_triple_states(
            target, mask, seasonality_idxs
        )
        if len(self._init_cache) > 8:
            # Bound growth from one-shot datasets (e.g. _store_final_states,
            # conformal re-anchoring); the persistent train/eval entries are
            # simply recomputed once after a clear.
            self._init_cache.clear()
        # Storing the dataset pins its id: a key hit therefore implies this
        # exact dataset, and with it identical target/mask/positions.
        self._init_cache[key] = {
            "data": data,
            "seasonality": seasonality,
            "level0": level0,
            "trend0": trend0,
        }
        return seasonality.clone(), level0, trend0

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

        Notes
        -----
        The validation metric is the **recursive h-step forecast** loss, not the
        in-sample one-step fit. Rolling the deployment recursion from the
        training-window terminal states and scoring against the holdout measures
        the quantity the model is actually used for. The naive in-sample
        validation loss is degenerate for the seasonal variants when
        ``fcst_h <= season_length`` (every state update becomes a
        parameter-independent fixed point, so the loss is ~0 for *any*
        parameters and early stopping selects noise). The training metric
        remains the in-sample one-step fit.
        """
        is_higher_better = False  # Lower loss is better, so we don't maximize
        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        target = torch.tensor(
            eval_data.get_label().reshape(self.n_series, -1),
            dtype=self.dtype
        )

        if dataset_name == "validation" and self._eval_boundary is not None:
            # Recursive h-step forecast metric. The terminal states were stashed
            # during this same iteration's "train" eval call, so the boundary
            # states and the horizon parameters come from the identical model
            # state (no off-by-one tree).
            loss = self._recursive_eval_loss(predt, eval_data, target)
            loss_val = loss.item()
            if not np.isfinite(loss_val):
                # A diverged rollout early in boosting would otherwise feed NaN
                # to early stopping; report a large finite value (worst) instead.
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
            predt: np.ndarray,
            eval_data: lgb.Dataset,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Recursive h-step forecast loss for the validation split.

        Mirrors deployment: the predicted parameters over the validation window
        drive the same rolled recursion as :meth:`forecast` (via the shared
        :meth:`_roll_forecast` helper), starting from the training-window
        terminal states stored in ``self._eval_boundary``. Padded holdout rows
        (mask == 0) are excluded from the loss.

        Parameters
        ----------
        predt : np.ndarray
            Raw LightGBM outputs over the validation rows (class-major order).
        eval_data : lgb.Dataset
            Validation dataset (provides the seasonality feature and mask).
        target : torch.Tensor
            Holdout observations, shape ``(n_series, fcst_h)``.

        Returns
        -------
        torch.Tensor
            Scalar loss between the rolled forecasts and the holdout.
        """
        params = torch.clamp(
            self.sigmoid_fn(
                torch.tensor(
                    predt.reshape(-1, self.n_params, order="F"),
                    dtype=self.dtype,
                ).reshape(self.n_series, -1, self.n_params)
            ),
            min=self.eps,
            max=1 - self.eps,
        )

        level_h, trend_h, seasonality = self._eval_boundary
        if self.ets_type in ("triple", "additive"):
            seasonality_idxs = self._seasonal_positions(
                eval_data.data[self.seasonality_feature].values
            ).reshape(self.n_series, -1)
        else:
            seasonality_idxs = None

        fcsts = self._roll_forecast(
            level_h, trend_h, seasonality, params, seasonality_idxs
        )

        if "mask" in eval_data.data.columns:
            mask = torch.tensor(
                eval_data.data["mask"].values.reshape(self.n_series, -1),
                dtype=self.dtype,
            )
        else:
            mask = torch.ones_like(target)

        return self.loss_fn(fcsts * mask, target * mask)

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

        # Forward pass to compute fitted values. Keep the terminal level/
        # trend/seasonality states so the recursive validation metric can roll
        # the deployment recursion forward from the training-window boundary
        # (see eval_fn / _recursive_eval_loss).
        last_level, last_trend, seasonality, fit = self.forward(params, data, target, mask)
        self._last_states = (last_level, last_trend, seasonality)

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

        Parameters
        ----------
        params : torch.Tensor
            Sigmoid-transformed ETS parameters, shape ``(n_series, T, n_params)``.
        data : lgb.Dataset
            LightGBM dataset whose raw DataFrame provides the seasonality feature.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.

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

        # Get seasonal indices from features first: the state initialization
        # assigns initial indices to the same slots the recursion reads, so a
        # series that does not start at season position 1 is not rotated.
        seasonality_idxs = self._seasonal_positions(
            data.data[self.seasonality_feature].values
        ).reshape(self.n_series, -1)
        batch_idx = torch.arange(self.n_series, dtype=torch.long)

        # Initialize seasonal indices and level/trend states via the classical
        # decomposition-based estimator. The result depends only on the data
        # (never on the boosted parameters), so it is cached per lgb.Dataset
        # and reused across boosting iterations.
        seasonality, level_prev, trend_prev = self._cached_init_triple_states(
            data, target, mask, seasonality_idxs
        )
        fits = [target[:, 0]]

        # Pre-unbind the data tensors so the loop indexes Python tuples
        # instead of slicing tensors at every step.
        target_t = target.unbind(dim=1)
        mask_t = mask.unbind(dim=1)
        idxs_t = seasonality_idxs.unbind(dim=1)

        # Triple ETS updates with masking for padded values. Shared
        # subexpressions are hoisted: every node created here is re-traversed
        # by each backward pass (gradient plus the Hutchinson probes), so a
        # smaller graph speeds up the forward and every backward.
        for t in range(1, series_len):
            valid_mask = mask_t[t]
            invalid_mask = 1 - valid_mask
            y_t = target_t[t]
            s_prev = seasonality[batch_idx, idxs_t[t]]  # s_{t-m}
            phi_trend = phi_t[t] * trend_prev           # phi_t * b_{t-1}
            pred_base = level_prev + phi_trend          # l_{t-1} + phi_t * b_{t-1}

            fit_t = valid_mask * (pred_base * s_prev) + invalid_mask * fits[-1]

            level_new = valid_mask * (
                    alpha_t[t] * (y_t / s_prev) +
                    (1 - alpha_t[t]) * pred_base
            ) + invalid_mask * level_prev

            trend_new = valid_mask * (
                    beta_t[t] * (level_new - level_prev) +
                    (1 - beta_t[t]) * phi_trend
            ) + invalid_mask * trend_prev

            seasonality[batch_idx, idxs_t[t]] = valid_mask * (
                    gamma_t[t] * (y_t / pred_base) +
                    (1 - gamma_t[t]) * s_prev
            ) + invalid_mask * s_prev

            fits.append(fit_t)
            level_prev = level_new
            trend_prev = trend_new

        return level_prev, trend_prev, seasonality, fits

    def _init_additive_states(
            self,
            target: torch.Tensor,
            mask: torch.Tensor,
            seasonality_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initial seasonal indices and level/trend states for additive ETS.

        Additive counterpart of the classical ``_init_triple_states``
        estimator, following R's ``forecast::ets`` / statsmodels' heuristic
        initialization [1, 2]: the trend is removed with a centered 2 x m
        moving average, the detrended *differences* ``y - trend`` are averaged
        per seasonal slot and normalized to mean zero, and level/trend are
        seeded by an OLS fit on the seasonally adjusted head ``y - s``.
        Indices are assigned via ``seasonality_idxs`` so they land in the
        slots the recursion reads; short series and empty slots fall back to a
        simple per-slot deviation from the series mean. All statistics are
        mask-aware.

        References
        ----------
        [1] Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D.
            (2008). Forecasting with Exponential Smoothing: The State Space
            Approach. Springer. (Initialization heuristic, Section 2.6.1)
        [2] Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting:
            Principles and Practice (3rd ed.). OTexts.
            https://otexts.com/fpp3/

        Parameters
        ----------
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding),
            shape ``(n_series, T)``.
        seasonality_idxs : torch.Tensor
            0-based seasonal slot per observation, shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Seasonal indices ``(n_series, season_length)``, initial level
            ``(n_series,)``, and initial trend ``(n_series,)``.
        """
        N, T = target.shape
        m = self.season_length

        slot_ids = torch.arange(N, dtype=torch.long).unsqueeze(1) * m + seasonality_idxs  # (N, T)
        ym = target * mask

        def slot_sum(ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            """Sum ``values`` into their (series, slot) cells -> (N, m)."""
            return torch.zeros(N * m, dtype=self.dtype).index_add_(
                0, ids.reshape(-1), values.reshape(-1)
            ).reshape(N, m)

        # --- Fallback estimator: per-slot deviation from the series mean ----
        cnts = slot_sum(slot_ids, mask)
        slot_mean = slot_sum(slot_ids, ym) / cnts.clamp(min=1.0)
        grand_mean = ym.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        simple_idx = torch.where(
            cnts > 0,
            slot_mean - grand_mean.unsqueeze(1),
            torch.zeros_like(slot_mean),
        )
        seasonality = simple_idx

        # --- Detrended estimator: centered 2 x m MA, then per-slot diffs ----
        L = m + 1 if m % 2 == 0 else m  # always odd
        if T >= L:
            kernel = torch.full((1, 1, L), 1.0 / m, dtype=self.dtype)
            if m % 2 == 0:
                kernel[0, 0, 0] = 0.5 / m
                kernel[0, 0, -1] = 0.5 / m
            trend_ma = torch.nn.functional.conv1d(
                ym.unsqueeze(1), kernel
            ).squeeze(1)  # (N, T - L + 1), centered at offset L // 2
            window_valid = torch.nn.functional.conv1d(
                mask.unsqueeze(1), torch.ones((1, 1, L), dtype=self.dtype)
            ).squeeze(1) >= L - 0.5  # full window unmasked (implies a valid center)

            half = L // 2
            y_c = target[:, half:T - half]  # window centers, length T - L + 1
            diffs = torch.where(
                window_valid, y_c - trend_ma, torch.zeros_like(y_c)
            )

            d_cnts = slot_sum(slot_ids[:, half:T - half], window_valid.to(self.dtype))
            detr_idx = slot_sum(slot_ids[:, half:T - half], diffs) / d_cnts.clamp(min=1.0)
            seasonality = torch.where(d_cnts > 0, detr_idx, simple_idx)

        # Normalize to mean zero (additive seasonal indices sum to ~0)
        seasonality = seasonality - seasonality.mean(dim=1, keepdim=True)

        # --- Level/trend: OLS on the seasonally-adjusted head (as in ets) ----
        s_t = seasonality.gather(1, seasonality_idxs)
        y_sa = target - s_t
        maxn = min(max(10, 2 * m), T)
        t_idx = torch.arange(1, maxn + 1, dtype=self.dtype).unsqueeze(0)
        w = mask[:, :maxn]
        sw = w.sum(dim=1).clamp(min=1.0)
        mean_t = (w * t_idx).sum(dim=1) / sw
        mean_y = (w * y_sa[:, :maxn]).sum(dim=1) / sw
        dev_t = t_idx - mean_t.unsqueeze(1)
        var_t = (w * dev_t ** 2).sum(dim=1)
        cov_ty = (w * dev_t * (y_sa[:, :maxn] - mean_y.unsqueeze(1))).sum(dim=1)
        trend0 = torch.where(
            var_t > self.eps,
            cov_ty / var_t.clamp(min=self.eps),
            torch.zeros_like(cov_ty),
        )
        # Evaluate the line at the first observation (t = 1 on the OLS axis)
        # so level0 matches this recursion's timing, which seeds the state at
        # the first observation rather than one step before it.
        level0 = mean_y + trend0 * (1.0 - mean_t)

        return seasonality, level0, trend0

    def _cached_init_additive_states(
            self,
            data: lgb.Dataset,
            target: torch.Tensor,
            mask: torch.Tensor,
            seasonality_idxs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache wrapper around :meth:`_init_additive_states`.

        Identical caching semantics to :meth:`_cached_init_triple_states`:
        results are cached per lgb.Dataset (pinning the dataset object), the
        seasonal indices are cloned on every return because the recursion
        updates them in place, and the cache is cleared by ``train()``.

        Parameters
        ----------
        data : lgb.Dataset
            Dataset whose identity serves as the cache key.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.
        seasonality_idxs : torch.Tensor
            0-based seasonal slot per observation, shape ``(n_series, T)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Seasonal indices ``(n_series, season_length)``, initial level
            ``(n_series,)``, and initial trend ``(n_series,)``.
        """
        key = id(data)
        entry = self._init_cache.get(key)
        if entry is not None:
            return entry["seasonality"].clone(), entry["level0"], entry["trend0"]

        seasonality, level0, trend0 = self._init_additive_states(
            target, mask, seasonality_idxs
        )
        if len(self._init_cache) > 8:
            # Bound growth from one-shot datasets (e.g. _store_final_states,
            # conformal re-anchoring); the persistent train/eval entries are
            # simply recomputed once after a clear.
            self._init_cache.clear()
        # Storing the dataset pins its id: a key hit therefore implies this
        # exact dataset, and with it identical target/mask/positions.
        self._init_cache[key] = {
            "data": data,
            "seasonality": seasonality,
            "level0": level0,
            "trend0": trend0,
        }
        return seasonality.clone(), level0, trend0

    def _forward_additive(
            self,
            params: torch.Tensor,
            data: lgb.Dataset,
            target: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for additive triple exponential smoothing.

        This implements the ETS state space equations for the additive
        damped-trend Holt-Winters method (ETS(A,Ad,A)), in the component form
        of Hyndman & Athanasopoulos, Forecasting: Principles and Practice
        (https://otexts.com/fpp3/):
        - Level: l_t = α(y_t - s_{t-m}) + (1-α)(l_{t-1} + φb_{t-1})
        - Trend: b_t = β(l_t - l_{t-1}) + (1-β)φb_{t-1}
        - Seasonality: s_t = γ(y_t - l_{t-1} - φb_{t-1}) + (1-γ)s_{t-m}
        - Fitted: ŷ_t = l_{t-1} + φb_{t-1} + s_{t-m}

        Parameters
        ----------
        params : torch.Tensor
            Sigmoid-transformed ETS parameters, shape ``(n_series, T, n_params)``.
        data : lgb.Dataset
            LightGBM dataset whose raw DataFrame provides the seasonality feature.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.

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

        # Get seasonal indices from features first: the state initialization
        # assigns initial indices to the same slots the recursion reads, so a
        # series that does not start at season position 1 is not rotated.
        seasonality_idxs = self._seasonal_positions(
            data.data[self.seasonality_feature].values
        ).reshape(self.n_series, -1)
        batch_idx = torch.arange(self.n_series, dtype=torch.long)

        # Initialize seasonal indices and level/trend states via the classical
        # decomposition-based estimator (additive form). The result depends
        # only on the data, so it is cached per lgb.Dataset and reused across
        # boosting iterations.
        seasonality, level_prev, trend_prev = self._cached_init_additive_states(
            data, target, mask, seasonality_idxs
        )
        fits = [target[:, 0]]

        # Pre-unbind the data tensors so the loop indexes Python tuples
        # instead of slicing tensors at every step.
        target_t = target.unbind(dim=1)
        mask_t = mask.unbind(dim=1)
        idxs_t = seasonality_idxs.unbind(dim=1)

        # Additive ETS updates with masking for padded values (shared
        # subexpressions hoisted; see _forward_triple).
        for t in range(1, series_len):
            valid_mask = mask_t[t]
            invalid_mask = 1 - valid_mask
            y_t = target_t[t]
            s_prev = seasonality[batch_idx, idxs_t[t]]  # s_{t-m}
            phi_trend = phi_t[t] * trend_prev           # phi_t * b_{t-1}
            pred_base = level_prev + phi_trend          # l_{t-1} + phi_t * b_{t-1}

            fit_t = valid_mask * (pred_base + s_prev) + invalid_mask * fits[-1]

            level_new = valid_mask * (
                    alpha_t[t] * (y_t - s_prev) +
                    (1 - alpha_t[t]) * pred_base
            ) + invalid_mask * level_prev

            trend_new = valid_mask * (
                    beta_t[t] * (level_new - level_prev) +
                    (1 - beta_t[t]) * phi_trend
            ) + invalid_mask * trend_prev

            seasonality[batch_idx, idxs_t[t]] = valid_mask * (
                    gamma_t[t] * (y_t - pred_base) +
                    (1 - gamma_t[t]) * s_prev
            ) + invalid_mask * s_prev

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

        Parameters
        ----------
        params : torch.Tensor
            Sigmoid-transformed ETS parameters, shape ``(n_series, T, n_params)``.
        data : lgb.Dataset
            Unused; kept for a uniform forward signature with ``_forward_triple``.
        target : torch.Tensor
            Observations, shape ``(n_series, T)``.
        mask : torch.Tensor
            Validity mask (1 = real observation, 0 = padding), shape ``(n_series, T)``.

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

        # Initialize states for the trend model. With back-appended padding
        # the slope endpoint must be the last *valid* observation inside the
        # window, not a padded value, so the endpoint index is mask-aware.
        level_prev = target[:, 0]
        cap = min(2 * self.season_length - 1, series_len - 1)
        n_valid = mask[:, :cap + 1].sum(dim=1).to(torch.long)
        last_idx = (n_valid - 1).clamp(min=0)
        endpoint = target.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        trend_prev = (endpoint - level_prev) / last_idx.to(self.dtype).clamp(min=1.0)
        fits = [target[:, 0]]

        # Pre-unbind the data tensors so the loop indexes Python tuples
        # instead of slicing tensors at every step.
        target_t = target.unbind(dim=1)
        mask_t = mask.unbind(dim=1)

        # Trend-only updates with masking for padded values (shared
        # subexpressions hoisted; see _forward_triple).
        for t in range(1, series_len):
            valid_mask = mask_t[t]
            invalid_mask = 1 - valid_mask
            pred_base = level_prev + trend_prev  # l_{t-1} + b_{t-1}

            fit_t = valid_mask * pred_base + invalid_mask * fits[-1]

            level_new = valid_mask * (
                    alpha_t[t] * target_t[t] +
                    (1 - alpha_t[t]) * pred_base
            ) + invalid_mask * level_prev

            trend_new = valid_mask * (
                    beta_t[t] * (level_new - level_prev) +
                    (1 - beta_t[t]) * trend_prev
            ) + invalid_mask * trend_prev

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
        self._init_cache = {}
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
                        seasonal_init=self.seasonal_init,
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
        Store final ETS states after training for use in forecasting.

        Runs a full forward pass on the training data to obtain the final
        level, trend, and (for the seasonal variants, triple/additive)
        seasonality states per series.
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

        # Store seasonality for the seasonal ETS variants
        if self.ets_type in ("triple", "additive") and seasonality is not None:
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
        validate_series_order(history, name="history")
        self._store_final_states(history)

    def _roll_forecast(
            self,
            level_h: torch.Tensor,
            trend_h: torch.Tensor,
            seasonality: Optional[torch.Tensor],
            params: torch.Tensor,
            seasonality_idxs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Roll the ETS deployment recursion forward from terminal states.

        Shared by :meth:`forecast` and the recursive validation metric
        (:meth:`_recursive_eval_loss`) so the deployed forecast and the
        early-stopping metric can never diverge. Each step's own one-step fit
        serves as the pseudo-observation in the state updates (future
        innovations are zero in expectation), exactly mirroring the training
        forward passes. The caller's ``seasonality`` is cloned, not mutated.

        Parameters
        ----------
        level_h : torch.Tensor
            Terminal level state at the forecast origin, shape ``(n_series,)``.
        trend_h : torch.Tensor
            Terminal trend state at the forecast origin, shape ``(n_series,)``.
        seasonality : torch.Tensor or None
            Terminal seasonal states, shape ``(n_series, season_length)``;
            ``None`` for ``ets_type="trend"``.
        params : torch.Tensor
            Sigmoid-transformed ETS parameters per horizon step,
            shape ``(n_series, H, n_params)``.
        seasonality_idxs : torch.Tensor or None
            0-based seasonal slot per horizon step, shape ``(n_series, H)``;
            ``None`` for ``ets_type="trend"``.

        Returns
        -------
        torch.Tensor
            Point forecasts, shape ``(n_series, H)``.
        """
        H = params.shape[1]
        n_series = params.shape[0]
        batch_idx = torch.arange(n_series, dtype=torch.long)
        level_h = level_h.clone()
        trend_h = trend_h.clone()
        if seasonality is not None:
            seasonality = seasonality.clone()
        fcsts = []

        if self.ets_type == "triple":
            for h in range(H):
                alpha = params[:, h, 0]
                beta = params[:, h, 1]
                gamma = params[:, h, 2]
                phi = params[:, h, 3]
                s_idx = seasonality_idxs[:, h].long()
                s_h = seasonality[batch_idx, s_idx]

                # One-step-ahead fit (pseudo-observation),
                # structurally identical to _forward_triple.
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

        elif self.ets_type == "additive":
            for h in range(H):
                alpha = params[:, h, 0]
                beta = params[:, h, 1]
                gamma = params[:, h, 2]
                phi = params[:, h, 3]
                s_idx = seasonality_idxs[:, h].long()
                s_h = seasonality[batch_idx, s_idx]

                # One-step-ahead fit (pseudo-observation),
                # structurally identical to _forward_additive.
                pred_base = level_h + phi * trend_h
                pseudo_y = pred_base + s_h
                fcsts.append(pseudo_y.reshape(-1, 1))

                # State updates exactly as in _forward_additive.
                level_new = (
                    alpha * (pseudo_y - s_h)
                    + (1 - alpha) * pred_base
                )
                trend_new = (
                    beta * (level_new - level_h)
                    + (1 - beta) * phi * trend_h
                )
                seasonality[batch_idx, s_idx] = (
                    gamma * (pseudo_y - pred_base)
                    + (1 - gamma) * s_h
                )
                level_h = level_new
                trend_h = trend_new

        elif self.ets_type == "trend":
            for h in range(H):
                alpha = params[:, h, 0]
                beta = params[:, h, 1]

                # One-step-ahead fit (pseudo-observation),
                # structurally identical to _forward_trend.
                pseudo_y = level_h + trend_h
                fcsts.append(pseudo_y.reshape(-1, 1))

                # State updates exactly as in _forward_trend.
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

        return torch.cat(fcsts, dim=1)

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

        Note that the pseudo-observation rollout collapses algebraically: the
        predicted alpha/beta/gamma at the forecast horizon cancel out of the
        trajectory (they multiply innovations that are zero in expectation),
        so only the damping phi and the seasonal-slot rotation shape the
        h-step forecast -- and for ``ets_type="trend"`` no horizon parameter
        affects it at all. Horizon features therefore do not alter the point
        forecasts beyond phi; they do affect ``type="parameters"``.

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

                # Roll the ETS state forward via the shared recursion (also
                # used by the recursive validation metric in eval_fn, so the
                # two can never diverge).
                if self.ets_type in ("triple", "additive"):
                    # Extract seasonality in test series order
                    seasonality = torch.stack(
                        [self.fcst_states[series_id]['seasonality'] for series_id in test_series_ids]
                    )
                    seasonality_idxs = self._seasonal_positions(
                        test_data[self.seasonality_feature].values
                    ).reshape(self.n_series, self.fcst_h)
                else:
                    seasonality = None
                    seasonality_idxs = None

                fcsts_mat = self._roll_forecast(
                    last_level, last_trend, seasonality, fcst_params, seasonality_idxs
                )

                # Create output dataframe
                model_name = f"Hyper-Tree-ETS({self.ets_type})"
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": fcsts_mat.flatten().numpy(),
                    "model": model_name,
                })

                if level is not None:
                    point = fcsts_mat.numpy()  # (n_series, fcst_h)
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
                param_names = ["alpha", "beta", "gamma", "phi"] if self.ets_type in ("triple", "additive") else ["alpha", "beta"]
                for i, param_name in enumerate(param_names):
                    out_df[param_name] = fcst_params[:, :, i].flatten().numpy()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
