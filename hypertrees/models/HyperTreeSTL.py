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

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, GaussNewtonHessian, NoDeepcopyObjective
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)

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
            hessian_method: str = "exact",
            n_hessian_probes: int = 5,
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
        hessian_method : str
            Method for computing the Hessian diagonal. Options:
            - "exact" (default): per-parameter second-order autograd, with the
              diagonal floored at a small positive value (the trend-smoothing
              window of the "default" variant enters the fit nonlinearly
              through a sigmoid, so its exact curvature can go negative,
              which Newton boosting cannot consume).
            - "gn": Gauss-Newton approximation estimated via Hutchinson
              probing. Positive semi-definite by construction; the curvature
              of the trend smoothness penalty is dropped.
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            Only used when hessian_method="gn". More probes reduce variance but
            increase computation. Default is 5.
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
        if getattr(loss_fn, "reduction", "mean") == "none":
            raise ValueError(
                "loss_fn must use a scalar reduction ('mean' or 'sum'); "
                "reduction='none' returns per-element losses that the "
                "boosting objective cannot consume."
            )
        if not isinstance(freq, str):
            raise TypeError("freq must be a string representing the frequency of the time series.")
        if hessian_method not in ("exact", "gn"):
            raise ValueError("hessian_method must be either 'exact' or 'gn'.")
        if not isinstance(n_hessian_probes, int) or n_hessian_probes <= 0:
            raise ValueError("n_hessian_probes must be a positive integer.")
        if type not in ["default", "paper"]:
            raise ValueError("Type must be either 'default' or 'paper'.")

        if hessian_method == "gn" and not isinstance(loss_fn, nn.MSELoss):
            warnings.warn(
                f"Loss {loss_fn.__class__.__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )

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
        self._seasonal_offset = None  # Training-window seasonal centering (set in train)
        self._trend_tail = None       # Raw-trend tail of the training window (default variant)
        self._w_eff_train = None      # Trained effective smoothing window (default variant)
        self._train_time_end = None   # Last training time index (out-of-sample gate)

        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
        self._iter_count = 0
        self._fit = None
        self._target = None

        # Conformal forecast interval state (populated when train() is
        # called with forecast_intervals).
        self._is_calibrated = False
        self._cs_scores = None          # conformity scores (n_windows, n_series, fcst_h)
        self._cs_series_order = None    # series order along axis 1 of _cs_scores
        self._pi_config = None          # ForecastIntervals configuration

        # Bind Hessian computation strategy
        if hessian_method == "exact":
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians
        else:
            self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_gn

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
        self._iter_count += 1

        # Target values
        target = torch.tensor(
            data.get_label(),
            dtype=self.dtype
        ).reshape(-1, self.n_series)

        # Calculate gradients and hessians
        params, loss = self.get_params_loss(predt, target, self.time_idx_train, requires_grad=True)
        grad, hess = self.calculate_gradients_and_hessians(loss, params)

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

        if self.hessian_method == "gn":
            # The trend and seasonal losses share the same residual
            # (trend - (y - seasonality) == seasonality - (y - trend)), so
            # their average reduces to loss_fn(trend + seasonality, target);
            # the GGN is estimated for that data-fit term.
            self._fit = trend + seasonality
            self._target = target

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

        # Convert to numpy arrays and reshape as expected by LightGBM.
        # The exact diagonal Hessian is floored at a small positive value:
        # the trend-smoothing window ("default" variant) enters the fit
        # nonlinearly through a sigmoid, so its exact second derivative can
        # go negative, which Newton boosting cannot consume (the leaf update
        # would step uphill and LightGBM curtails splits). The linear
        # parameters have nonnegative exact curvature under MSE and are
        # unaffected by the floor.
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).clamp(min=1e-6).cpu().detach().numpy().ravel(order="F")

        # Clear existing gradients to prevent accumulation
        params.grad = None

        return grad, hess

    def _calculate_gradients_and_hessians_gn(
            self,
            loss: torch.Tensor,
            params: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian diagonal estimated via Hutchinson probing.

        The GGN is computed for the data-fit term ``loss_fn(trend +
        seasonality, target)`` stored by ``get_params_loss``; the curvature
        of the trend smoothness penalty is dropped, keeping the estimate
        positive semi-definite. Gradients remain exact (including the
        penalty).

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
        grad = autograd(loss, params, retain_graph=True)[0]
        rng = torch.Generator().manual_seed(self._iter_count)
        hess = self._gn_hessian.estimate(self._fit, self._target, params, rng)
        self._fit = None
        self._target = None
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")

        return grad, hess

    def _forward_paper(
            self,
            params: torch.Tensor,
            time_idx: torch.Tensor,
            seasonal_offset: Optional[torch.Tensor] = None,
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
        seasonal_offset : torch.Tensor, optional
            Per-series centering constant, shape ``(n_series,)``. When None
            (training), the seasonal component is re-centered over the given
            window; when provided (forecasting), this stored training offset
            is subtracted instead so the decomposition continues the trained
            one (see ``_compute_seasonal_offset``).

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

        # Center the seasonal component for identifiability. During training
        # (seasonal_offset=None) the mean over the given window is removed;
        # at forecast time the stored training offset is subtracted instead,
        # so the decomposition continues the trained one rather than
        # re-centering over the (typically partial-cycle) forecast window.
        if seasonal_offset is None:
            seasonality = seasonality - torch.mean(seasonality, dim=0, keepdim=True)
        else:
            seasonality = seasonality - seasonal_offset

        return trend, seasonality

    def _forward_default(
            self,
            params: torch.Tensor,
            time_idx: torch.Tensor,
            seasonal_offset: Optional[torch.Tensor] = None,
            trend_tail: Optional[torch.Tensor] = None,
            w_eff: Optional[torch.Tensor] = None,
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
        seasonal_offset : torch.Tensor, optional
            Per-series centering constant, shape ``(n_series,)``. When None
            (training), the seasonal component is re-centered per cycle over
            the given window; when provided (forecasting), this stored
            training offset is subtracted instead so the decomposition
            continues the trained one (see ``_anchor_forecast_state``).
        trend_tail : torch.Tensor, optional
            Raw-trend tail of the training window, shape ``(L, n_series)``.
            When provided (out-of-sample forecasting), the trend is smoothed
            jointly with this real history instead of over the window in
            isolation (see ``_smooth_trend_continuation``).
        w_eff : torch.Tensor, optional
            Trained effective smoothing window per series, shape
            ``(n_series,)``. When provided, overrides the window implied by
            the given rows so train and forecast use the same kernel.

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
        if w_eff is None:
            w_logit = params[:, :, 2]
            w_eff = (max_w_model - 3.0) * torch.sigmoid(w_logit.mean(dim=0)) + 3.0  # (N,)

        if trend_tail is not None:
            trend = self._smooth_trend_continuation(trend_raw, trend_tail, w_eff, max_w_model)
        else:
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

        # At forecast time, continue the training decomposition by
        # subtracting the stored training offset instead of re-centering
        # over the forecast window (see _compute_seasonal_offset).
        if seasonal_offset is not None:
            return trend, seasonality - seasonal_offset

        # Per-cycle centering (sum over a cycle ≈ 0)
        S_ext, C = self._extend_to_full_cycles(seasonality)
        S_mcN = S_ext.view(C, m, N).transpose(0, 1).contiguous()  # (m,C,N)
        S_mcN = S_mcN - S_mcN.mean(dim=0, keepdim=True)  # zero-mean per cycle
        seasonality = S_mcN.transpose(0, 1).reshape(C * m, N)[:T, :]  # (T,N)

        return trend, seasonality

    def _extend_to_full_cycles(self, seasonality: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Reflect-extend the seasonal component to a whole number of cycles.

        When the window length is not a multiple of the period, the final
        cycle is completed by reflection; when the series is shorter than the
        padding (less than half a seasonal cycle observed), the reflection is
        ping-ponged until a full cycle can be assembled. This is the extension
        the per-cycle training centering uses, so consumers that need "the
        constant training removed" must extend the same way.

        Parameters
        ----------
        seasonality : torch.Tensor
            Raw seasonal component, shape ``(T, n_series)``.

        Returns
        -------
        Tuple[torch.Tensor, int]
            The extended component, shape ``(C * period, n_series)``, and the
            number of cycles ``C``.
        """
        T = seasonality.shape[0]
        m = self.period
        C = (T + m - 1) // m
        pad_T = C * m - T
        if pad_T > 0:
            tail = torch.flip(seasonality, dims=[0])
            while tail.shape[0] < pad_T:
                tail = torch.cat([tail, torch.flip(tail, dims=[0])], dim=0)
            seasonality = torch.cat([seasonality, tail[:pad_T]], dim=0)

        return seasonality, C

    def _smooth_trend_continuation(
            self,
            trend_raw: torch.Tensor,
            trend_tail: torch.Tensor,
            w_eff: torch.Tensor,
            max_w_model: int,
    ) -> torch.Tensor:
        """Smooth the forecast-window trend jointly with the training tail.

        Smoothing the horizon in isolation reflect-pads both of its edges;
        with ``fcst_h`` close to the period, every horizon point then sits
        within half a kernel of an edge, and folding a linear ramp back onto
        itself flattens the slope the tree intended. Here the left side of
        the kernel reads the *real* end of the training window, and the right
        side is extended by odd (point) reflection around the last value,
        which continues a locally linear trend exactly instead of folding it.

        Parameters
        ----------
        trend_raw : torch.Tensor
            Raw (unsmoothed) trend over the forecast window, shape ``(T, N)``.
        trend_tail : torch.Tensor
            Raw-trend tail of the training window, shape ``(L, N)``.
        w_eff : torch.Tensor
            Effective smoothing window per series, shape ``(N,)``.
        max_w_model : int
            Maximum kernel support (odd).

        Returns
        -------
        torch.Tensor
            Smoothed trend over the forecast window, shape ``(T, N)``.
        """
        T, N = trend_raw.shape
        full = torch.cat([trend_tail.to(trend_raw.dtype), trend_raw], dim=0)  # (L+T,N)
        M = full.shape[0]
        K = min(max_w_model, 2 * M - 1)
        if K < 3:
            return trend_raw

        half = K // 2
        w = torch.clamp(w_eff, max=float(K))
        offsets = torch.arange(-half, half + 1, dtype=trend_raw.dtype).abs().view(1, -1)  # (1,K)
        k = torch.sigmoid(w.view(-1, 1) / 2.0 - offsets)  # (N,K)
        k = (k / k.sum(dim=1, keepdim=True)).unsqueeze(1)  # (N,1,K)

        xin = full.T.contiguous().unsqueeze(0)  # (1,N,L+T)
        # Left: covered by the real tail; reflect-pad only any deficit
        # (training windows shorter than half a kernel).
        deficit = half - trend_tail.shape[0]
        if deficit > 0:
            xin = torch.nn.functional.pad(xin, (deficit, 0), mode="reflect")
        # Right: odd reflection around the endpoint continues the local
        # linear trend, where even reflection would fold it into a tent.
        rpad = (2.0 * xin[..., -1:] - xin[..., -half - 1:-1]).flip(-1)
        xin = torch.cat([xin, rpad], dim=-1)

        smoothed = torch.nn.functional.conv1d(xin, k, groups=N).squeeze(0).T

        return smoothed[-T:]

    def _anchor_forecast_state(self, full_ts: pd.DataFrame) -> None:
        """Anchor the forecast continuation to the training window.

        Two pieces of state make ``forecast`` continue the trained
        decomposition instead of re-deriving it over the forecast window in
        isolation:

        * The seasonal centering offset: the forward passes enforce the
          seasonal identifiability constraint by re-centering over whatever
          window they are given. Re-centering over a (typically
          partial-cycle) forecast window would subtract a different constant
          than the training fit removed, leaking a phase-dependent level
          offset between trend and seasonality across the train/test
          boundary. This stores the constant the training fit removed: the
          mean raw seasonal value over the training window ("paper" variant),
          or the mean over the last training cycle, reflect-extended exactly
          as the per-cycle training centering extends it ("default" variant).
        * For the "default" variant, the raw-trend tail of the training
          window and the trained effective smoothing window, so the horizon
          trend is smoothed jointly with real history using the training
          kernel (see ``_smooth_trend_continuation``).

        Parameters
        ----------
        full_ts : pd.DataFrame
            Preprocessed training data (features and ``time`` column).
        """
        params = torch.tensor(
            self.model.predict(
                full_ts[self.features]
            ).reshape(-1, self.n_series, self.n_params, order="F"),
            dtype=self.dtype,
        )
        time_idx = torch.tensor(
            full_ts["time"].to_numpy().reshape(-1, self.n_series), dtype=self.dtype
        )
        # A zero offset returns the *uncentered* seasonal component.
        zero = torch.zeros(self.n_series, dtype=self.dtype)
        _, seasonality_raw = self._forward(params, time_idx, seasonal_offset=zero)

        if self.forward_type == "paper":
            self._seasonal_offset = seasonality_raw.mean(dim=0)
        else:
            # Training centers cycles anchored at row 0; take the constant it
            # removed from the *last* cycle (including the reflected
            # completion when T is not a multiple of the period), not the
            # mean of a trailing window that straddles cycle boundaries.
            S_ext, C = self._extend_to_full_cycles(seasonality_raw)
            self._seasonal_offset = S_ext[(C - 1) * self.period:].mean(dim=0)

            max_w_model = min(2 * self.period + 1, 101)
            trend_raw = params[:, :, 0] + params[:, :, 1] * time_idx
            self._trend_tail = trend_raw[-(max_w_model // 2):].detach()
            self._w_eff_train = (
                (max_w_model - 3.0) * torch.sigmoid(params[:, :, 2].mean(dim=0)) + 3.0
            ).detach()

        self._train_time_end = float(time_idx[-1].max())

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
        forecast_intervals : ForecastIntervals, optional
            If provided, calibrate conformal forecast intervals via rolling-window
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

        # Reset state for re-training
        self.model = None
        self.dataset_references = {}
        self.is_trained = False
        self.features = None
        self._seasonal_offset = None
        self._trend_tail = None
        self._w_eff_train = None
        self._train_time_end = None
        self._iter_count = 0
        self._fit = None
        self._target = None
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

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

        # Fail fast if the series is too short for the requested conformal calibration.
        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals, min_train=self.period + 1
            )

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

            # Anchor the forecast continuation to the training window: the
            # seasonal centering constant and, for the "default" variant, the
            # raw-trend tail and effective smoothing window (see
            # _anchor_forecast_state).
            self._anchor_forecast_state(full_ts)

            # Set trained flag to True
            self.is_trained = True

            # Calibrate conformal forecast intervals via rolling-window CV.
            # Fresh model instances are trained per window (no forecast_intervals
            # passed, so there is no recursion) using the same hyper-parameters.
            if forecast_intervals is not None:
                def _model_factory():
                    return HyperTreeSTL(
                        period=self.period,
                        num_seasonal_components=self.num_seasonal_components,
                        freq=self.freq,
                        fcst_h=self.fcst_h,
                        loss_fn=self.loss_fn,
                        hessian_method=self.hessian_method,
                        n_hessian_probes=self.n_hessian_probes,
                        type=self.forward_type,
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

    def set_forecast_origin(self, history: pd.DataFrame) -> None:
        """Re-anchor the decomposition continuation to the end of *history*.

        Recomputes the seasonal centering offset and (default variant) the
        trend tail and smoothing window over *history* without retraining.
        Used by conformal calibration with ``refit=False``.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with ``series_id``, ``date``, ``time``, ``value`` and
            the training feature columns, ordered by date.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("set_forecast_origin requires a trained model.")
        validate_series_order(history, name="history")
        self._anchor_forecast_state(history)

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None,
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
        level : list of int, optional
            Confidence levels (in ``(0, 100)``, e.g. ``[80, 90]``) for conformal
            forecast intervals. Only valid with ``type="forecast"`` and requires
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
            - trend, seasonality: Component values (if type="components")
            - trend_intercept, trend_slope, trend_window_logit (default only),
              seasonal_sine{i}, seasonal_cosine{i}: Parameter values (if type="parameters")
            - <model>-lo-<level> / <model>-hi-<level>: forecast interval bounds
              (if type="forecast" and level is provided)
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

        # Validate conformal interval request
        if level is not None:
            if type != "forecast":
                raise ValueError("level is only supported with type='forecast'.")
            if not self._is_calibrated:
                raise RuntimeError(
                    "Forecast intervals were requested via level, but the model "
                    "was not calibrated. Pass forecast_intervals=ForecastIntervals(...) "
                    "to train() before forecasting with level."
                )
            if not isinstance(level, (list, tuple)) or len(level) == 0:
                raise ValueError("level must be a non-empty list of integers.")
            for lv in level:
                if not isinstance(lv, (int, np.integer)) or not 0 < lv < 100:
                    raise ValueError(f"level values must be integers in (0, 100); got {lv}.")

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

            # Forward pass to calculate trend and seasonal components; the
            # stored training offset continues the trained decomposition
            # instead of re-centering over the forecast window. On a genuinely
            # out-of-sample window, the "default" variant additionally smooths
            # the trend jointly with the stored training tail; windows that
            # overlap the training data (e.g. in-sample decompositions via
            # type="components") keep the plain windowed smoothing.
            out_of_sample = (
                self.forward_type == "default"
                and self._trend_tail is not None
                and self._train_time_end is not None
                and float(time_idx[0].min()) > self._train_time_end
            )
            if out_of_sample:
                trend, seasonality = self._forward(
                    params_fcst, time_idx, self._seasonal_offset,
                    trend_tail=self._trend_tail, w_eff=self._w_eff_train,
                )
            else:
                trend, seasonality = self._forward(params_fcst, time_idx, self._seasonal_offset)

            # Combine components to get forecasted values
            fcsts_stl = trend + seasonality

            # Create output dataframe based on requested type
            model_name = f"Hyper-Tree-STL({self.period})"
            if type == "forecast":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": fcsts_stl.detach().numpy().flatten(),
                    "model": model_name,
                })

                # Append conformal forecast intervals if requested.
                if level is not None:
                    point = fcsts_stl.detach().numpy().reshape(-1, n_series_test).T  # (n_series, fcst_h)
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
            elif type == "components":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "trend": trend.detach().numpy().flatten(),
                    "seasonality": seasonality.detach().numpy().flatten(),
                    "model": model_name,
                })
            elif type == "parameters":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": model_name,
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
