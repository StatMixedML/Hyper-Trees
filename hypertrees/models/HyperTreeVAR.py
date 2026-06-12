import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Callable, Optional, Tuple

from ..utils import TrainingResult, GaussNewtonHessian
from ..conformal import ForecastIntervals
from ._var_base import _HyperTreeVARBase


class HyperTreeVAR(_HyperTreeVARBase):
    """
    Class that implements a Hyper-Tree-VAR(p) model for multivariate time series forecasting.

    The Hyper-Tree-VAR(p) model extends the Hyper-Tree-AR(p) model to a vector
    autoregression over an aligned panel of k series, learning the full
    time-varying coefficient matrices A_1(x), ..., A_p(x) with gradient boosted
    trees so that y_{i,t} = sum_j A_j[i, :](x_{i,t}) . y_{t-j}. Cross-series
    dependence is captured by the off-diagonal coefficients of the lag matrices
    (see ``_var_base.py`` for the formulation, data requirements, coefficient
    ordering, and the restricted ``type="factor"`` design).

    Key features:
    - Combines tree-based models (LightGBM) with vector autoregressive modeling
    - Learns the time-varying lag matrices A_1, ..., A_p as functions of
      features (full k x k, or own + factor lags with type="factor")
    - Captures cross-series (Granger-causal) lead/lag dependencies via the
      off-diagonal coefficients
    - Provides VAR coefficients that can vary over time

    Use this model when:
    - Your series influence each other and forecasts should exploit
      cross-series lead/lag structure
    - The panel is small (k * p up to a few dozen coefficients) and
      coefficient-level interpretability (including SHAP values per
      coefficient) is desired
    - For larger panels, use HyperTreeNetVAR, whose boosting cost is
      independent of the number of coefficients

    Example usage:
    ```python
    # Imports
    from hypertrees.models.HyperTreeVAR import HyperTreeVAR
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 4
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeVAR(p=lag_p, freq=frequency, fcst_h=fcst_h)

    # Data
    # The data needs to be an aligned panel (equal lengths and identical dates
    # across series) with the columns: 'date', 'series_id', 'value'. All other
    # columns are automatically treated as features. You don't have to add
    # lag-values yourself, this happens automatically during training.
    rng = np.random.RandomState(1)
    dates = pd.date_range("2010-01-01", periods=120 + fcst_h, freq="MS")
    df = pd.concat(
        [
            pd.DataFrame({
                "series_id": f"s{i}",
                "date": dates,
                "value": base + np.cumsum(rng.randn(len(dates))),
                "month": dates.month,
                "quarter": dates.quarter,
                "series_num": i,  # identifies the series, so equations can differ
            })
            for i, base in enumerate([100.0, 150.0])
        ],
        ignore_index=True,
    )
    test = df.groupby("series_id", sort=False).tail(fcst_h)
    train = df.drop(test.index)

    # Train model
    model.train(
        lgb_params={'learning_rate': 0.1},
        num_iterations=100,
        train_data=train
    )

    # Generate forecasts and inspect the time-varying VAR coefficients
    forecasts = model.forecast(test_data=test)
    coefficients = model.forecast(test_data=test, type="parameters")

    # Plot results
    for sid, group in df.groupby("series_id", sort=False):
        plt.plot(group["date"], group["value"], label=f"Actual {sid}",
                 color='#2E86AB', linewidth=2, alpha=0.8)
    for sid, group in forecasts.groupby("series_id", sort=False):
        plt.plot(group["date"], group["fcst"], label=f"Forecast {sid}",
                 color='#F18F01', linestyle='--', linewidth=2, alpha=0.8)

    plt.title('Aligned Panel - VAR Forecasts', fontsize=14)
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ```
    """

    _model_label = "Hyper-Tree-VAR"
    _valid_forecast_types = ("forecast", "parameters")

    def __init__(
            self,
            p: int = 2,
            freq: str = "M",
            fcst_h: int = 1,
            loss_fn: Callable = nn.MSELoss(),
            scaling: Optional[str] = "mean",
            type: str = "full",
            hessian_method: str = "analytic",
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-Tree-VAR(p) model.

        Arguments
        ----------
        p : int
            VAR lag order. Must be a positive integer.
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
        scaling : str, optional
            Per-series scaling applied internally before training; forecasts
            (and prediction intervals) are transformed back to the original
            scale automatically. Options: "mean" (default; divide by the
            mean absolute training value), "standard" (z-score), or None.
            Strongly recommended for heterogeneous panels: VAR coefficients
            multiply *other* series' values, so unscaled panels force the
            model to learn scale conversions while the loss is dominated by
            the largest series. Coefficients returned by
            ``forecast(type="parameters")`` live in the scaled space.
        type : str
            Structure of the VAR design vector. Options:
            - "full" (default): unrestricted VAR; every equation regresses on
              the lags of all k series (``k * p`` coefficients per equation,
              one boosted tree each).
            - "factor": restricted GVAR-style design; every equation
              regresses on its own lags plus the lags of the equal-weighted
              cross-sectional average of the scaled panel (``2 * p``
              coefficients per equation, independent of k). Recommended for
              larger panels, where the unrestricted design overparameterizes.
              Parameter columns are named ``A{j}(own)`` / ``A{j}(factor)``
              and the model name becomes ``Hyper-Tree-FactorVAR(p)``.
        hessian_method : str
            Method for computing the Hessian diagonal. Options:
            - "exact": Exact diagonal Hessian via per-coefficient second-order
              autograd (one backward pass per coefficient, i.e. k * p per
              iteration -- costly for larger panels).
            - "analytic" (default): Closed-form gradients and exact diagonal
              Hessians, exploiting that the VAR fit is linear in its
              parameters (dL/dA_q = l'(y_hat) * z_q and
              d2L/dA_q2 = l''(y_hat) * z_q**2, the second-order fit term
              vanishing exactly). Produces the same values as "exact" for any
              loss that is a mean/sum of per-observation terms -- which covers
              all standard PyTorch regression losses -- at a fraction of the
              cost: at most one small double-backward through
              loss(fit, target) instead of one backward per coefficient.
              nn.MSELoss uses a fully closed-form fast path with no autograd
              at all.
            - "gn": Gauss-Newton approximation estimated via Hutchinson
              probing. Guarantees positive semi-definite Hessians. Because
              the VAR fit is linear in its parameters, this estimates the
              same diagonal as "analytic", with Hutchinson sampling variance.
        n_hessian_probes : int
            Number of Hutchinson probes for Gauss-Newton Hessian diagonal estimation.
            Only used when hessian_method="gn". More probes reduce variance but
            increase computation. Default is 5.
        """
        super().__init__(
            p=p,
            freq=freq,
            fcst_h=fcst_h,
            loss_fn=loss_fn,
            scaling=scaling,
            type=type,
        )
        if hessian_method not in ("exact", "analytic", "gn"):
            raise ValueError("hessian_method must be one of 'exact', 'analytic', or 'gn'.")
        if not isinstance(n_hessian_probes, int) or n_hessian_probes <= 0:
            raise ValueError("n_hessian_probes must be a positive integer.")
        if hessian_method == "gn" and not isinstance(loss_fn, nn.MSELoss):
            warnings.warn(
                f"Loss {loss_fn.__class__.__name__} is not nn.MSELoss. The Gauss-Newton "
                "Hessian requires a twice-differentiable loss; non-smooth losses "
                "(e.g., L1Loss, quantile loss, HuberLoss/SmoothL1Loss outside the quadratic "
                "region) have zero or undefined second derivatives at kinks, "
                "causing degenerate Hessians."
            )

        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
        self._fit = None
        self._target = None
        self._lags = None

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
        based on the PyTorch loss function. It converts the raw LightGBM outputs
        to VAR coefficients, computes the loss, and then derives gradients and
        Hessians via the bound ``hessian_method`` strategy.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the VAR coefficients per
            training row.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        self._iter_count += 1

        target = torch.tensor(
            data.get_label().reshape(self.k, -1), dtype=self.dtype
        )
        params, loss = self.get_params_loss(predt, target, self._Z_train, requires_grad=True)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Training diverged at boosting iteration {self._iter_count}: the loss "
                "is no longer finite. With one boosted tree per VAR coefficient, "
                "strongly correlated series make the per-coefficient Newton steps "
                "overshoot: reduce learning_rate (a rule of thumb is eta / k) and "
                "keep per-series scaling enabled."
            )
        grad, hess = self.calculate_gradients_and_hessians(loss, params)

        return grad, hess

    def get_params_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            Z: torch.Tensor,
            requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into VAR coefficients and calculate loss.

        This function:
        1. Reshapes the raw outputs into the coefficient matrix
        2. Computes the fitted values via the VAR forward pass
        3. Calculates the loss between fitted and actual values

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM (flattened Fortran-order).
        target : torch.Tensor
            Target values (actual time series values), shape ``(k, T_r)``.
        Z : torch.Tensor
            VAR design matrix, shape ``(T_r, k * p)`` for the full design or
            ``(k * T_r, 2 * p)`` for the factor design.
        requires_grad : bool
            Whether to compute gradients (True during training).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Parameters tensor and loss value.
        """
        params = nn.Parameter(
            torch.tensor(
                predt.reshape(-1, self.n_params, order="F"),
                dtype=self.dtype
            ),
            requires_grad=requires_grad
        )

        fit = self._compute_fit(params, Z)
        loss = self.loss_fn(fit, target)

        if self.hessian_method in ("gn", "analytic"):
            self._fit = fit
            self._target = target
            self._lags = Z

        return params, loss

    def _calculate_gradients_and_hessians_exact(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Exact diagonal Hessian via per-coefficient second-order autograd.

        One backward pass per VAR coefficient (k * p per iteration); identical
        values to "analytic" for any per-observation loss, at much higher cost.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (VAR coefficients as an ``nn.Parameter``,
            shape ``(k * T_r, n_params)``).

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

        Since the VAR fit is linear in its parameters, ``grad = l'(y_hat) * z``
        and ``hess = l''(y_hat) * z**2``, matching the "exact" method for any
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
        Z = self._lags

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

        # Broadcast the per-observation loss derivatives over the design
        # matrix (shared per time step for the full design, per row for the
        # factor design), then Fortran-ravel as expected by LightGBM.
        if self.type == "factor":
            grad = g.reshape(-1, 1) * Z
            hess = h.reshape(-1, 1) * Z ** 2
        else:
            grad = (g.unsqueeze(2) * Z.unsqueeze(0)).reshape(-1, self.n_params)
            hess = (h.unsqueeze(2) * Z.unsqueeze(0) ** 2).reshape(-1, self.n_params)
        grad = grad.cpu().numpy().ravel(order="F")
        hess = hess.cpu().numpy().ravel(order="F")

        self._fit = None
        self._target = None
        self._lags = None

        return grad, hess

    def _calculate_gradients_and_hessians_gn(self, loss: torch.Tensor, params: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian diagonal estimated via Hutchinson probing.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        params : torch.Tensor
            Model parameters (VAR coefficients as an ``nn.Parameter``,
            shape ``(k * T_r, n_params)``).

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
        self._lags = None
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")

        return grad, hess

    def _fit_from_predt(self, predt: np.ndarray, Z: torch.Tensor) -> torch.Tensor:
        """Compute the fitted values for raw LightGBM coefficient outputs.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM (flattened Fortran-order).
        Z : torch.Tensor
            Design matrix for the dataset being evaluated, shape
            ``(T_r, k * p)`` for the full design or ``(k * T_r, 2 * p)`` for
            the factor design.

        Returns
        -------
        torch.Tensor
            Fitted values, shape ``(k, T_r)``.
        """
        params = torch.tensor(
            predt.reshape(-1, self.n_params, order="F"), dtype=self.dtype
        )

        return self._compute_fit(params, Z)

    def _num_class(self) -> int:
        """LightGBM output dimension: one tree per VAR coefficient."""

        return self.n_params

    def _reset_training_state(self) -> None:
        """Reset per-training state, including the Hessian-strategy buffers."""
        super()._reset_training_state()
        self._fit = None
        self._target = None
        self._lags = None

    def _post_datasets_setup(self, seed: int) -> None:
        """Warn when the one-vs-all strategy becomes the runtime bottleneck."""
        if self.type == "full" and self.n_params > 50:
            warnings.warn(
                f"HyperTreeVAR grows num_class = k * p = {self.n_params} trees per "
                f"boosting iteration, which scales linearly in runtime. For panels "
                f"of this size consider HyperTreeNetVAR (GBDT cost independent of "
                f"the number of coefficients) or type='factor' (2 * p "
                f"coefficients per equation, independent of k)."
            )

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
        Train the Hyper-Tree-VAR model on an aligned panel of time series.

        This method:
        1. Pivots the panel and builds the VAR design matrix
        2. Sets up LightGBM datasets (one row per series and time step)
        3. Trains the model using gradient boosting

        The training data must contain columns:
        - 'series_id': Identifier for each time series
        - 'date': Timestamp for each observation
        - 'value': Target value to forecast
        - Additional feature columns used for forecasting

        All series must have the same length and identical dates (aligned
        panel); see ``_var_base.py``.

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters like 'learning_rate', 'num_leaves', etc.
        num_iterations : int
            Number of boosting rounds for training. Note that each round grows
            one tree per coefficient (``k * p`` for type="full", ``2 * p`` for
            type="factor").
        train_data : pd.DataFrame
            Training data containing series_id, date, value and feature columns
        validation : bool
            If True, a validation set will be created for evaluation. It splits
            the last fcst_h time steps of each series for validation.
        early_stopping_round : int, optional
            If provided, training will stop if the validation loss does not
            improve for this many rounds.
        seed : int
            Random seed for reproducibility
        verbose : int
            Verbosity level for LightGBM training
        deterministic : bool
            If True, sets LightGBM's ``deterministic`` and ``force_row_wise``
            parameters to ensure reproducible results. May slow down training.
            See https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic
        forecast_intervals : ForecastIntervals, optional
            If provided, calibrate conformal prediction intervals via
            rolling-window cross-validation after the main model is trained.
            The collected conformity scores are then used by
            ``forecast(..., level=[...])`` to produce ``<model>-lo-<level>`` /
            ``<model>-hi-<level>`` columns. See
            :class:`hypertrees.conformal.ForecastIntervals`.

        Returns
        -------
        TrainingResult
            Object containing evaluation results and training information.
        """
        def _model_factory():
            return HyperTreeVAR(
                p=self.p,
                freq=self.freq,
                fcst_h=self.fcst_h,
                loss_fn=self.loss_fn,
                scaling=self.scaling,
                type=self.type,
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

        return self._train_core(
            lgb_params=lgb_params,
            num_iterations=num_iterations,
            train_data=train_data,
            validation=validation,
            early_stopping_round=early_stopping_round,
            seed=seed,
            verbose=verbose,
            deterministic=deterministic,
            forecast_intervals=forecast_intervals,
            model_factory=_model_factory,
            cal_train_kwargs=cal_train_kwargs,
        )

    def _forecast_params(self, features_df: pd.DataFrame) -> np.ndarray:
        """Forecast the ``(n_rows, n_params)`` coefficient matrix from the GBDT.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature frame (training feature columns only).

        Returns
        -------
        np.ndarray
            VAR coefficients per row, shape ``(n_rows, n_params)``.
        """
        params_fcst = np.asarray(self.model.predict(features_df))
        # Booster.predict returns (n_rows, n_params) for multi-class output
        if params_fcst.ndim == 1:
            params_fcst = params_fcst.reshape(-1, self.n_params)

        return params_fcst
