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

from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, validate_series_order, extract_forecast_lags, GaussNewtonHessian, NoDeepcopyObjective
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)
from .HyperTreeAR import HyperTreeAR


class HyperTreeARMA:
    """
    Class that implements a Hyper-Tree-ARMA(p, q) model for time series forecasting.

    The Hyper-Tree-ARMA(p, q) model extends the Hyper-Tree-AR(p) model with a
    moving-average block, so that

        y_t = sum_{j=1..p} phi_j(x_t) * y_{t-j} + sum_{i=1..q} theta_i(x_t) * eps_{t-i} + eps_t

    where the AR coefficients phi_j and the MA coefficients theta_i are
    time-varying and estimated by gradient boosted trees. The MA block is an
    error-correction mechanism: it regresses on the model's own past one-step
    forecast errors, adjusting the forecast when recent periods were over- or
    under-predicted.

    Because the innovations eps_t are latent, an exact ARMA fit would require
    reconstructing them recursively from the parameters (eps_t depends on all
    earlier eps), which makes the fit nonlinear in its parameters and forces
    an O(T) sequential autograd graph per boosting iteration (the
    Hyper-Tree-ETS situation). This implementation avoids the recursion with
    the classical two-stage Hannan-Rissanen approach:

    1. **Stage 1**: a long autoregression (a ``HyperTreeAR`` of order
       ``stage1_p``, by default the Gomez-Maravall proposal
       ``max(floor(log(T)**2), 2 * max(p, q))`` used by statsmodels'
       ``hannan_rissanen`` and RATS' ``@HannanRissanen``) is fitted to the
       training data and its in-sample one-step residuals
       ``eps_hat_t = y_t - y_hat_t`` are extracted.
    2. **Stage 2**: the lagged residuals are treated as *observed* regressors,
       so the ARMA fit becomes linear in its parameters -- structurally
       identical to the AR design with a widened lag matrix
       ``[y_{t-1..t-p}, eps_hat_{t-1..t-q}]`` -- and trains with the same
       closed-form analytic gradients and exact diagonal Hessians at AR speed.

    The classical third Hannan-Rissanen stage (a one-step Gauss-Newton bias
    correction of the stage-2 estimates) is intentionally omitted: its
    derivative series are themselves recursive filters in the estimated
    coefficients, which would reintroduce the sequential graph this
    estimator exists to avoid. Unlike the classical procedure, both stages
    here are feature-driven GBDTs, so the residual extractor and the ARMA
    coefficients are time-varying.

    The stage-1 residuals are used both as training regressors and as the
    forecast seed, so the coefficients are applied at forecast time to the
    same quantities they were trained on. Beyond the forecast origin, future
    innovations are unobserved with expectation zero, so the MA terms
    contribute to the first ``q`` horizon steps (multiplying the known last
    residuals) and then vanish, leaving the pure AR recursion.

    Key features:
    - Combines tree-based models (LightGBM) with ARMA time series modeling
    - Allows AR and MA coefficients to vary based on features
    - Recursion-free estimation via Hannan-Rissanen residual proxies:
      analytic gradients/Hessians and AR-level training speed
    - MA block corrects the first q forecast steps using the latest
      observed forecast errors

    Use this model when:
    - The series has short-memory error-correction structure that a pure
      AR(p) of moderate order does not capture
    - You have relevant features that might influence the autoregressive
      or error-correction structure

    Note that training fits two GBDTs (the stage-1 AR and the stage-2 ARMA),
    roughly doubling the training cost relative to ``HyperTreeAR``. A pure
    ``HyperTreeAR`` with a longer lag order approximates the same conditional
    mean (every invertible ARMA has an AR(infinity) representation) and is the
    natural baseline to compare against.

    References
    ----------
    [1] Hannan, E. J., & Rissanen, J. (1982). Recursive Estimation of Mixed
        Autoregressive-Moving Average Order. Biometrika, 69(1), 81-94.
    [2] Gomez, V., & Maravall, A. (2001). Automatic Modeling Methods for
        Univariate Series. In Pena, Tiao & Tsay (eds.), A Course in Time
        Series Analysis. Wiley. (Default order of the stage-1 long AR.)

    Example usage:
    ```python
    # Imports
    from hypertrees.models import HyperTreeARMA
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 2
    lag_q = 1
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeARMA(p=lag_p, q=lag_q, freq=frequency, fcst_h=fcst_h)

    # Data
    # The data needs to have the following columns: 'date', 'series_id', 'value'. All other columns are automatically treated as features.
    # You don't have to add lag-values or residuals yourself, this happens automatically during training.
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

    # Generate forecasts and inspect the time-varying ARMA coefficients
    forecasts = model.forecast(test_data=test)
    coefficients = model.forecast(test_data=test, type="parameters")

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
            q: int = 1,
            freq: str = "M",
            fcst_h: int = 1,
            loss_fn: Callable = nn.MSELoss(),
            hessian_method: str = "analytic",
            n_hessian_probes: int = 5,
            stage1_p: Optional[int] = None,
    ):
        """
        Initialize the Hyper-Tree-ARMA(p, q) model.

        Arguments
        ----------
        p : int
            Number of AR lags. Must be a positive integer.
        q : int
            Number of MA terms (lagged residual regressors). Must be a
            positive integer; for q = 0 use ``HyperTreeAR`` directly.
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
            - "exact": Exact diagonal Hessian via per-parameter second-order
              autograd (one backward pass per coefficient and iteration).
            - "analytic" (default): Closed-form gradients and exact diagonal
              Hessians, exploiting that -- with the stage-1 residuals frozen
              as observed regressors -- the ARMA fit is linear in its
              parameters (dL/dtheta_j = l'(y_hat) * z_j and
              d2L/dtheta_j2 = l''(y_hat) * z_j**2, the second-order fit term
              vanishing exactly). Produces the same values as "exact" for any
              loss that is a mean/sum of per-observation terms -- which covers
              all standard PyTorch regression losses -- at a fraction of the
              cost. nn.MSELoss uses a fully closed-form fast path with no
              autograd at all.
            - "gn": Gauss-Newton approximation estimated via Hutchinson
              probing. Guarantees positive semi-definite Hessians. Because
              the fit is linear in its parameters, this estimates the same
              diagonal as "analytic", with Hutchinson sampling variance.
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            Only used when hessian_method="gn". More probes reduce variance but
            increase computation. Default is 5.
        stage1_p : int, optional
            Lag order of the stage-1 autoregression used to extract the
            residual proxies (the Hannan-Rissanen "long AR"). If None
            (default), it is resolved at training time via the
            Gomez-Maravall (2001) proposal used by statsmodels and RATS:
            ``max(floor(log(T)**2), 2 * max(p, q))``, with ``T`` the
            shortest series length. Larger values give cleaner residual
            proxies at the cost of dropping more training rows: stage-2
            training uses rows from ``max(p, stage1_p + q) + 1`` onward per
            series. Pass a smaller value explicitly for short series.
        """
        # Validate inputs
        if not isinstance(p, int) or p <= 0:
            raise ValueError("Parameter 'p' must be a positive integer.")
        if not isinstance(q, int) or q <= 0:
            raise ValueError(
                "Parameter 'q' must be a positive integer. For q = 0 (no MA "
                "terms) use HyperTreeAR directly."
            )
        if fcst_h <= 0:
            raise ValueError("Forecast horizon 'fcst_h' must be a positive integer.")
        if not isinstance(freq, str):
            raise TypeError("freq must be a string.")
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
        if hessian_method not in ("exact", "analytic", "gn"):
            raise ValueError("hessian_method must be one of 'exact', 'analytic', or 'gn'.")
        if not isinstance(n_hessian_probes, int) or n_hessian_probes <= 0:
            raise ValueError("n_hessian_probes must be a positive integer.")
        if stage1_p is not None and (not isinstance(stage1_p, int) or stage1_p <= 0):
            raise ValueError("stage1_p must be a positive integer.")

        if hessian_method == "gn" and not isinstance(loss_fn, nn.MSELoss):
            warnings.warn(
                f"Loss {loss_fn.__class__.__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )

        self.p = p
        self.q = q
        self.n_params = p + q
        self._stage1_p_arg = stage1_p
        self.stage1_p = stage1_p  # resolved at training time when None
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
        self._stage1 = None  # Trained stage-1 HyperTreeAR (residual extractor)
        self.fcst_lags = None  # {series_id: last p values, newest first}
        self.fcst_eps = None   # {series_id: last q stage-1 residuals, newest first}
        self._iter_count = 0
        self._fit = None
        self._target = None
        self._design = None

        # Conformal prediction interval state (populated when train() is called
        # with forecast_intervals).
        self._is_calibrated = False
        self._cs_scores = None          # conformity scores (n_windows, n_series, fcst_h)
        self._cs_series_order = None    # series order along axis 1 of _cs_scores
        self._pi_config = None          # ForecastIntervals configuration

        # Bind Hessian computation strategy
        if hessian_method == "exact":
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_exact
        elif hessian_method == "analytic":
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_analytic
        else:
            self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_gn

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts the raw LightGBM outputs to
        ARMA coefficients, computes the loss, and then derives gradients and
        Hessians via the bound ``hessian_method`` strategy.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the p AR and q MA
            coefficients per training row.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        self._iter_count += 1

        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=self.dtype)
        params, loss = self.get_params_loss(predt, target, self.design_train, requires_grad=True)
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
        # Use appropriate design rows based on dataset name
        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        if dataset_name == "train":
            design = self.design_train
        elif dataset_name == "validation":
            design = self.design_eval
        else:
            # Default to training design if unknown
            design = self.design_train
            warnings.warn("Unknown dataset in metric_fn. Using training design.")

        # Calculate loss
        is_higher_better = False # Lower loss is better, so we don't maximize
        target = torch.tensor(eval_data.get_label().reshape(-1, 1), dtype=self.dtype)
        _, loss = self.get_params_loss(predt, target, design)

        return self.loss_name, loss.item(), is_higher_better

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            design: torch.Tensor = None,
            requires_grad: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into ARMA parameters and calculate loss.

        This function:
        1. Reshapes the raw outputs into the coefficient matrix
        2. Multiplies the coefficients with the joint design rows
           ``[y-lags, residual-lags]``
        3. Computes the fit by summing the weighted design entries
        4. Calculates the loss between fitted and actual values

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM.
        target : torch.Tensor
            Target values (actual time series values).
        design : torch.Tensor
            Joint design rows ``[y_{t-1..t-p}, eps_hat_{t-1..t-q}]``,
            shape ``(n_samples, p + q)``.
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
                predt.reshape(-1, self.n_params, order="F"),
                dtype=self.dtype
            ),
            requires_grad=requires_grad
        )

        # Forward pass: Compute the fit by multiplying coefficients with the
        # design rows and summing
        fcst = torch.sum(params * design, dim=1, dtype=torch.float32).unsqueeze(1)

        # Calculate loss between fitted and actual values
        loss = self.loss_fn(fcst, target)

        if self.hessian_method in ("gn", "analytic"):
            self._fit = fcst
            self._target = target
            self._design = design

        return params, loss

    def _calculate_gradients_and_hessians_exact(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Exact diagonal Hessian via per-parameter second-order autograd.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (ARMA coefficients as an ``nn.Parameter``,
            shape ``(n_samples, p + q)``).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        loss.backward(create_graph=True)
        grad = params.grad
        hess = [
            autograd(grad[:, i].sum(), params, retain_graph=True)[0][:, i:(i + 1)]
            for i in range(self.n_params)
        ]

        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).cpu().detach().numpy().ravel(order="F")
        params.grad = None

        return grad, hess

    def _calculate_gradients_and_hessians_analytic(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form gradients and exact diagonal Hessians via model linearity.

        With the stage-1 residuals frozen as observed regressors, the ARMA fit
        is linear in its parameters, so ``grad = l'(y_hat) * z`` and
        ``hess = l''(y_hat) * z**2``, matching the "exact" method for any
        per-observation loss. MSELoss uses closed-form derivatives; other
        losses use one small double-backward through ``loss(fit, target)``.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model (unused; derivatives come from the
            fit/target/design stored by ``get_params_loss``).
        params : torch.Tensor
            Model parameters (unused, kept for a uniform dispatch signature).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        fit = self._fit.detach()
        target = self._target
        design = self._design

        if isinstance(self.loss_fn, nn.MSELoss) and self.loss_fn.reduction in ("mean", "sum"):
            # MSE fast path: l' = scale * (y_hat - y), l'' = scale
            scale = 2.0 / fit.numel() if self.loss_fn.reduction == "mean" else 2.0
            g = scale * (fit - target)
            h = torch.full_like(fit, scale)
        else:
            # Generic path: per-element first and second loss derivatives via
            # a double-backward through the (tiny) loss(fit, target) graph.
            # Requires the loss to have a well-defined double-backward (true
            # for HuberLoss/SmoothL1Loss and other standard smooth losses).
            fit_leaf = fit.requires_grad_(True)
            loss_local = self.loss_fn(fit_leaf, target)
            g = autograd(loss_local, fit_leaf, create_graph=True)[0]
            h = autograd(g.sum(), fit_leaf)[0].detach()
            g = g.detach()

        # Broadcast (N, 1) loss derivatives over the (N, p + q) design matrix.
        grad = (g * design).cpu().numpy().ravel(order="F")
        hess = (h * design ** 2).cpu().numpy().ravel(order="F")

        self._fit = None
        self._target = None
        self._design = None

        return grad, hess

    def _calculate_gradients_and_hessians_gn(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian diagonal estimated via Hutchinson probing.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (ARMA coefficients as an ``nn.Parameter``,
            shape ``(n_samples, p + q)``).

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
        self._design = None
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")

        return grad, hess

    def _stage1_residual_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Attach the stage-1 in-sample residuals to a sorted copy of *data*.

        Runs the trained stage-1 AR over *data* to compute its one-step
        in-sample residuals ``eps_hat_t = y_t - y_hat_t``. The first
        ``stage1_p`` rows of each series have no stage-1 fit and carry NaN.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with ``series_id``, ``date``, ``value`` and the
            training feature columns, ordered by ``(series_id, date)``.

        Returns
        -------
        pd.DataFrame
            Copy of *data*, sorted by ``(series_id, date)``, with an added
            ``resid`` column (NaN for the first ``stage1_p`` rows per series).
        """
        preprocessor = TimeSeriesPreprocessor(
            freq=self.freq,
            lags=[i for i in range(1, self.stage1_p + 1)],
        )
        lagged = preprocessor.create_lags(data)
        lagged_dict = preprocessor.extract(lagged)

        # Predict the stage-1 AR coefficients on the lagged rows; enforce the
        # stage-1 model's training feature order for the Booster.
        params = np.asarray(
            self._stage1.model.predict(lagged_dict["features"][self._stage1.features])
        )
        # Booster.predict returns (n_rows, stage1_p) for multi-class output
        if params.ndim == 1:
            params = params.reshape(-1, self.stage1_p)
        fit = (params * lagged_dict["lags_target"]).sum(axis=1)
        resid = lagged_dict["target"].ravel() - fit

        # Align back: `lagged` equals the sorted frame minus the first
        # stage1_p rows of each series, in the same row order.
        work = data.sort_values(["series_id", "date"]).reset_index(drop=True).copy()
        occ = work.groupby("series_id", sort=False).cumcount()
        work["resid"] = np.nan
        work.loc[occ >= self.stage1_p, "resid"] = resid

        return work

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
        Train the Hyper-Tree-ARMA model on time series data.

        This method:
        1. Trains the stage-1 long autoregression (a ``HyperTreeAR`` of order
           ``stage1_p``) with the same LightGBM hyper-parameters and extracts
           its in-sample one-step residuals (Hannan-Rissanen)
        2. Builds the joint design ``[y-lags, residual-lags]`` and sets up
           LightGBM datasets
        3. Trains the stage-2 ARMA model using gradient boosting

        The training data must contain columns:
        - 'series_id': Identifier for each time series
        - 'date': Timestamp for each observation
        - 'value': Target value to forecast
        - Additional feature columns used for forecasting

        Each series must have at least ``max(p, stage1_p + q) + 1`` rows so
        that one stage-2 training row remains. Note that the stage-1 model is
        fitted on the full training data, so with ``validation=True`` the
        validation metric shares stage-1 information through the residual
        regressors.

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters like 'learning_rate', 'num_leaves', etc.
            Used for both the stage-1 and the stage-2 GBDT.
        num_iterations : int
            Number of boosting rounds for training (both stages)
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
            Object containing evaluation results and training information
            for the stage-2 model.
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
        # monotonic dates so the training reshape and forecast seeds align.
        validate_series_order(train_data, name="train_data")

        # Resolve the stage-1 long-AR order. The default follows the
        # Gomez-Maravall (2001) proposal used by statsmodels'
        # hannan_rissanen and RATS' @HannanRissanen: the long AR grows with
        # the sample so the residual proxies stay consistent.
        lengths = train_data.groupby("series_id", sort=False).size()
        if self._stage1_p_arg is not None:
            self.stage1_p = self._stage1_p_arg
        else:
            t_min = int(lengths.min())
            self.stage1_p = max(
                int(np.floor(np.log(t_min) ** 2)), 2 * max(self.p, self.q)
            )

        # Each series must keep at least one stage-2 training row.
        needed = max(self.p, self.stage1_p + self.q) + 1
        bad = lengths[lengths < needed]
        if len(bad) > 0:
            raise ValueError(
                f"Series too short for stage1_p={self.stage1_p} and q={self.q}: "
                f"each series needs at least max(p, stage1_p + q) + 1 = {needed} "
                f"rows, but these series are shorter: {bad.to_dict()}. Pass a "
                f"smaller stage1_p to HyperTreeARMA for short series."
            )

        # Fail fast if any series is too short for the requested conformal
        # calibration. The stage-2 ARMA needs max(p, stage1_p + q) + 1 rows to
        # retain one training sample.
        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals,
                min_train=max(self.p, self.stage1_p + self.q) + 1,
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

        # Reset state for re-training
        self._iter_count = 0
        self._fit = None
        self._target = None
        self._design = None
        self.model = None
        self._stage1 = None
        self.fcst_lags = None
        self.fcst_eps = None
        self.dataset_references = {}
        self.is_trained = False
        self.features = None
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        try:
            # Stage 1 (Hannan-Rissanen): fit the long autoregression and
            # extract its in-sample one-step residuals as MA-term proxies.
            self._stage1 = HyperTreeAR(
                p=self.stage1_p,
                freq=self.freq,
                fcst_h=self.fcst_h,
                loss_fn=self.loss_fn,
                hessian_method="analytic",
            )
            self._stage1.train(
                lgb_params=lgb_params,
                num_iterations=num_iterations,
                train_data=train_data,
                validation=False,
                seed=seed,
                verbose=verbose,
                deterministic=deterministic,
            )
            work = self._stage1_residual_frame(train_data)

            # Stage 2: build the joint design. The y-lags come from the
            # standard preprocessor; the residual lags are appended as
            # lag{p+1}..lag{p+q} so the shared extract()/prepare_datasets
            # machinery picks up the joint [y-lags | eps-lags] design as one
            # (n_samples, p + q) tensor while keeping the residual columns
            # out of the GBDT feature set.
            preprocessor = TimeSeriesPreprocessor(
                freq=self.freq,
                lags=[i for i in range(1, self.p + 1)],
            )
            full_ts = preprocessor.create_lags(work.drop(columns=["resid"]))

            resid_grouped = work.groupby("series_id", sort=False)["resid"]
            elag_names = []
            elags = {}
            for i in range(1, self.q + 1):
                name = f"lag{self.p + i}"
                elags[name] = resid_grouped.shift(i)
                elag_names.append(name)
            occ = work.groupby("series_id", sort=False).cumcount()
            elag_df = pd.DataFrame(elags)[(occ >= self.p).to_numpy()].reset_index(drop=True)
            full_ts = pd.concat([full_ts, elag_df], axis=1)
            # Drop rows without q valid residual lags (the head of each
            # series up to stage1_p + q observations).
            full_ts = full_ts.dropna(subset=elag_names).reset_index(drop=True)

            full_dict = preprocessor.extract(full_ts)

            # Store feature names for later use
            self.features = full_dict["features"].columns.tolist()

            # Prepare datasets
            (valid_sets,
             valid_names,
             callbacks,
             evals_result,
             design_train,
             design_eval,
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

            # Store design rows for training and evaluation
            self.design_train = design_train
            self.design_eval = design_eval

            # Store the value and residual seeds to be used in the forecast method
            self.set_forecast_origin(train_data)

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
                    return HyperTreeARMA(
                        p=self.p,
                        q=self.q,
                        freq=self.freq,
                        fcst_h=self.fcst_h,
                        loss_fn=self.loss_fn,
                        hessian_method=self.hessian_method,
                        n_hessian_probes=self.n_hessian_probes,
                        stage1_p=self.stage1_p,
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
        """Re-anchor the ARMA value and residual seeds to the end of *history*.

        Recomputes the last ``p`` observed values and the last ``q`` stage-1
        residuals per series without retraining either GBDT. Used by conformal
        calibration with ``refit=False``.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with ``series_id``, ``date``, ``value`` and the training
            feature columns, ordered by ``(series_id, date)`` with each series
            in a contiguous block. Each series must have at least
            ``max(p, stage1_p + q)`` observations so that the residual seed
            exists.
        """
        if self._stage1 is None or self._stage1.model is None:
            raise RuntimeError("set_forecast_origin requires a trained model.")
        validate_series_order(history, name="history")

        needed = max(self.p, self.stage1_p + self.q)
        lengths = history.groupby("series_id", sort=False).size()
        bad = lengths[lengths < needed]
        if len(bad) > 0:
            raise ValueError(
                f"history must contain at least max(p, stage1_p + q) = {needed} "
                f"observations per series. Series too short: {bad.to_dict()}."
            )

        # Value seed: last p observations per series, newest first.
        self.fcst_lags = extract_forecast_lags(history, self.p)

        # Residual seed: last q stage-1 residuals per series, newest first.
        # The stage-1 residuals are the same quantities the MA coefficients
        # multiplied during training, keeping train and forecast consistent.
        work = self._stage1_residual_frame(history)
        tail = work.groupby("series_id", sort=False).tail(self.q)
        self.fcst_eps = {
            sid: grp["resid"].to_numpy()[::-1]
            for sid, grp in tail.groupby("series_id", sort=False)
        }

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast ARMA coefficients for each test point
        2. Recursively generates forecasts using the forecasted coefficients

        The forecasting process implements an ARMA model where:
        y_t = φ₁(x)y_{t-1} + ... + φₚ(x)y_{t-p} + θ₁(x)ε_{t-1} + ... + θ_q(x)ε_{t-q}

        Past residuals at the forecast origin are known (stage-1 in-sample
        errors); future innovations are unobserved with expectation zero, so
        the MA terms correct the first q horizon steps and then vanish,
        leaving the pure AR recursion.

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the ARMA coefficients used for forecasting
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
            - AR(j) / MA(i): coefficient values (if type="parameters")
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
        # monotonic dates so the forecast reshape aligns forecasts with seeds.
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

        model_name = f"Hyper-Tree-ARMA({self.p},{self.q})"

        try:

            if type == "forecast":
                # Get ARMA coefficient forecasts from the LightGBM model
                # Shape: (n_series, fcst_h, n_params)
                n_series_test = len(test_series_ids)
                params_fcst = self.model.predict(test_data[self.features]).reshape(n_series_test, self.fcst_h, self.n_params)

                # Reconstruct the seed states in the same order as test data
                lags = np.array([self.fcst_lags[series_id] for series_id in test_series_ids])
                eps = np.array([self.fcst_eps[series_id] for series_id in test_series_ids])

                # Generate multi-step forecasts
                forecasts = []
                for h in range(self.fcst_h):
                    # Compute next value using the ARMA equation:
                    # y_t = φ₁y_{t-1} + ... + φₚy_{t-p} + θ₁ε_{t-1} + ... + θ_qε_{t-q}
                    next_val = (
                        np.sum(params_fcst[:, h, :self.p] * lags, axis=1)
                        + np.sum(params_fcst[:, h, self.p:] * eps, axis=1)
                    ).reshape(-1, 1)
                    forecasts.append(next_val)

                    # Update the value lags with the new forecast; future
                    # innovations are unobserved with expectation zero, so the
                    # residual state is shifted with zeros (the MA terms die
                    # out after q steps).
                    lags = np.concatenate([next_val, lags[:, :-1]], axis=1)
                    eps = np.concatenate([np.zeros((n_series_test, 1)), eps[:, :-1]], axis=1)

                # Create output dataframe based on requested type
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
                # Booster.predict returns (n_test, p + q) for multi-class output
                if params_fcst.ndim == 1:
                    params_fcst = params_fcst.reshape(-1, self.n_params)
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": model_name,
                })
                # Add the AR and MA coefficients to the dataframe
                for j in range(self.p):
                    out_df[f"AR({j + 1})"] = params_fcst[:, j].flatten()
                for i in range(self.q):
                    out_df[f"MA({i + 1})"] = params_fcst[:, self.p + i].flatten()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
