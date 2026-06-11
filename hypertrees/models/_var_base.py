"""Shared base for the Hyper-Tree VAR models (``HyperTreeVAR`` / ``HyperTreeNetVAR``).

The Hyper-Tree VAR models extend the univariate AR variants to a
multivariate VAR(p) target model. For a panel of ``k`` aligned series:

    y_{i,t} = sum_{j=1..p} sum_{m=1..k} A_j[i, m](x_{i,t}) * y_{m,t-j}

i.e. every series' forecast is a feature-dependent linear combination of the
lagged values of *all* series. Cross-series dependence in the conditional
mean is carried entirely by the off-diagonal elements of the lag matrices
A_j (Granger-causal lead/lag effects). A multivariate innovation
distribution (e.g. the multivariate Gaussian used by GluonTS' DeepVAR) only
models the *contemporaneous covariance of the residuals*; it does not enter
the conditional mean and is therefore not needed for point forecasts.
Because all k equations share the same regressors, minimizing a
per-observation loss equation-by-equation is equivalent to the joint
(GLS/SUR) estimate in the classical linear case (Zellner, 1962), so nothing
is lost by training with the standard element-wise losses used across the
Hyper-Tree models. Conformal prediction intervals are supported exactly as
for the other models (per-series, per-horizon-step marginal intervals).

The two concrete models live in their own modules, mirroring the repo's
one-class-per-file convention:

- ``HyperTreeVAR`` (``HyperTreeVAR.py``): one boosted tree per VAR
  coefficient, closed-form analytic gradients/Hessians. For small panels;
  fully interpretable.
- ``HyperTreeNetVAR`` (``HyperTreeNetVAR.py``): GBDT encoder + MLP decoder
  (Hyper-TreeNet architecture); boosting cost independent of the number of
  coefficients, for larger panels.

Both accept a ``type`` argument: ``"full"`` (default; unrestricted,
``k * p`` coefficients per equation) or ``"factor"``, a restricted VAR in
the spirit of the Global VAR (GVAR) literature [1, 2], where every equation
regresses on its **own lags** plus the lags of the equal-weighted
cross-sectional average of the scaled panel (the GVAR "star variable"),
i.e. ``2 * p`` coefficients per equation independent of k -- the principled
remedy for the overparameterization of unrestricted VARs on larger panels.

Data requirements
-----------------
- Standard Hyper-Tree layout: columns ``series_id``, ``date``, ``value``,
  sorted by ``(series_id, date)`` with each series in a contiguous block.
- The panel must be *aligned*: every series must have the same length and
  identical dates, because each equation's design vector stacks the lags of
  all series at the same time points.
- Because the GBDT learns a single global mapping from features to
  coefficient vectors, the equations can only differ if some feature
  identifies the series. As with the other global Hyper-Tree models,
  include a series-identity feature (e.g. an integer-coded series id)
  in the feature set -- ideally with pandas ``category`` dtype, so
  LightGBM applies true categorical splits (an integer-coded identity
  treated as numeric can only separate series by intervals of an
  arbitrary coding).
- Series scales matter more than for the univariate AR models: VAR
  coefficients multiply *other* series' values, so on an unscaled
  heterogeneous panel the model must learn scale conversions between every
  pair of series while the loss is dominated by the largest series.
  Per-series scaling is therefore built in (``scaling="mean"`` by default)
  and forecasts are transformed back to the original scale automatically.

Coefficient ordering
--------------------
For the full design, the flat coefficient vector of length ``k * p``
produced per row is ordered lag-major: position ``(j - 1) * k + m`` is the
coefficient on lag ``j`` of the ``m``-th series (series in training
first-appearance order), matching the statsmodels VAR design-vector
convention ``z_t = [y'_{t-1}, y'_{t-2}, ..., y'_{t-p}]'``. For the factor
design, the own-lag block (j = 1..p) comes first, then the factor-lag block.

References
----------
[1] Pesaran, M. H., Schuermann, T., & Weiner, S. M. (2004). Modeling
    Regional Interdependencies Using a Global Error-Correcting
    Macroeconometric Model. Journal of Business & Economic Statistics,
    22(2), 129-162. (Global VAR; the "star variable" construction)
[2] Chudik, A., & Pesaran, M. H. (2016). Theory and Practice of GVAR
    Modelling. Journal of Economic Surveys, 30(1), 165-197.
[3] Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
    Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive
    (FAVAR) Approach. The Quarterly Journal of Economics, 120(1), 387-422.
"""

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb

from ..utils import CustomLogger
lgb.register_logger(CustomLogger())

from ..utils import TrainingResult, validate_series_order, NoDeepcopyObjective
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)

# Columns that are never features.
_RESERVED_COLUMNS = ("series_id", "date", "value")


def _pivot_panel(data: pd.DataFrame, name: str) -> Tuple[np.ndarray, List, np.ndarray]:
    """Pivot an aligned long panel into a ``(T, k)`` value matrix.

    Validates that all series have the same length and identical dates
    (required so that the VAR design vector ``z_t`` exists for every row).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel with ``series_id``, ``date``, ``value`` columns,
        sorted by ``(series_id, date)`` with contiguous series blocks.
    name : str
        Label used in error messages.

    Returns
    -------
    Tuple[np.ndarray, List, np.ndarray]
        Value matrix ``Y`` of shape ``(T, k)`` with columns in
        first-appearance series order, the series order, and the shared
        date vector of length ``T``.
    """
    lengths = data.groupby("series_id", sort=False).size()
    if lengths.nunique() != 1:
        raise ValueError(
            f"{name}: a VAR model requires an aligned panel where all series have "
            f"the same length. Found lengths: {lengths.to_dict()}."
        )
    T = int(lengths.iloc[0])
    series_order = list(dict.fromkeys(data["series_id"].tolist()))
    k = len(series_order)

    dates = pd.to_datetime(data["date"]).to_numpy().reshape(k, T)
    if k > 1 and not (dates == dates[0]).all():
        raise ValueError(
            f"{name}: all series must share identical dates (aligned panel). "
            f"The VAR design vector stacks the lags of all series at the same "
            f"time points, which requires aligned observations."
        )

    Y = data["value"].to_numpy(dtype=np.float64).reshape(k, T).T  # (T, k)

    return Y, series_order, dates[0]


def _validate_aligned_dates(data: pd.DataFrame, name: str) -> None:
    """Validate equal lengths and identical dates across series.

    Same alignment check as :func:`_pivot_panel` but without requiring a
    ``value`` column, so it can be applied to forecast inputs.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel with ``series_id`` and ``date`` columns, sorted by
        ``(series_id, date)`` with contiguous series blocks.
    name : str
        Label used in error messages.

    Raises
    ------
    ValueError
        If series lengths differ or dates are not identical across series.
    """
    lengths = data.groupby("series_id", sort=False).size()
    if lengths.nunique() != 1:
        raise ValueError(
            f"{name}: all series must have the same number of rows. "
            f"Found lengths: {lengths.to_dict()}."
        )
    T = int(lengths.iloc[0])
    k = lengths.size
    dates = pd.to_datetime(data["date"]).to_numpy().reshape(k, T)
    if k > 1 and not (dates == dates[0]).all():
        raise ValueError(f"{name}: all series must share identical dates (aligned panel).")


def _build_var_lags(Y: np.ndarray, p: int) -> np.ndarray:
    """Build the VAR design matrix ``Z`` from the value matrix ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Value matrix of shape ``(T, k)``.
    p : int
        VAR lag order.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(T - p, k * p)`` where row ``r``
        corresponds to time ``t = r + p`` and holds
        ``[y'_{t-1}, y'_{t-2}, ..., y'_{t-p}]`` (lag-major ordering).
    """
    T = Y.shape[0]

    return np.concatenate([Y[p - j: T - j] for j in range(1, p + 1)], axis=1)


class _HyperTreeVARBase:
    """Shared plumbing for the Hyper-Tree VAR variants.

    Handles panel validation/pivoting, design-matrix construction, dataset
    preparation, the training loop, the evaluation function, the multi-step
    forecast recursion, conformal interval wiring, and forecast-origin
    re-anchoring. Subclasses provide the LightGBM objective
    (``objective_fn``), the fitted values for a raw prediction vector
    (``_fit_from_predt``), the feature -> coefficient-matrix mapping
    (``_forecast_params``), and the LightGBM output dimension
    (``_num_class``).
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
    ):
        """
        Initialize the shared VAR state and validate the common arguments.

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
            nn.L1Loss is rejected (zero second derivative almost everywhere
            breaks Newton boosting).
        scaling : str, optional
            Per-series scaling applied internally before training; forecasts
            (and prediction intervals) are transformed back to the original
            scale automatically. Options:
            - "mean" (default): divide each series by its mean absolute
              training value. Location-free, so it introduces no implicit
              intercept, and per-equation least squares is equivariant under
              it (forecasts match manual pre-scaling exactly).
            - "standard": z-score per series (subtract the training mean,
              divide by the training standard deviation).
            - None: use the series as provided.
            Coefficients returned by ``forecast(type="parameters")`` live in
            the scaled space.
        type : str
            Structure of the VAR design vector. Options:
            - "full" (default): unrestricted VAR; every equation regresses on
              the lags of *all* k series (``k * p`` coefficients per equation).
            - "factor": restricted VAR in the spirit of the Global VAR (GVAR)
              literature; every equation regresses on its **own lags** plus
              the lags of the equal-weighted cross-sectional average of the
              scaled panel (the GVAR "star variable"), i.e. ``2 * p``
              coefficients per equation, independent of k. The principled
              remedy for the overparameterization of unrestricted VARs on
              larger panels.
        """
        if p <= 0:
            raise ValueError("Parameter 'p' must be a positive integer.")
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
        if scaling not in (None, "mean", "standard"):
            raise ValueError("scaling must be one of None, 'mean', or 'standard'.")
        if type not in ("full", "factor"):
            raise ValueError("type must be either 'full' or 'factor'.")

        self.p = p
        self.freq = freq
        self.fcst_h = fcst_h
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.scaling = scaling
        self.type = type
        self.dtype = torch.float32
        self.device = "cpu"

        self.model = None
        self.features = None            # Stores feature names after training
        self.is_trained = False         # Flag to track if model has been trained
        self.dataset_references = {}    # Store references to LightGBM datasets
        self.k = None                   # Number of series (set during training)
        self.n_params = None            # k*p (full) or 2*p (factor), set during training
        self.series_order_ = None       # Training series order (axis/coefficient order)
        self._Z_train = None            # design tensor: (T_train, k*p) full, (N, 2*p) factor
        self._Z_eval = None             # design tensor for the validation split
        self._fcst_state = None         # lag state at the forecast origin
        self._scale_loc = None          # (k,) per-series location (training order)
        self._scale_scale = None        # (k,) per-series scale (training order)
        self._iter_count = 0

        # Conformal prediction interval state (populated when train() is
        # called with forecast_intervals).
        self._is_calibrated = False
        self._cs_scores = None          # conformity scores (n_windows, n_series, fcst_h)
        self._cs_series_order = None    # series order along axis 1 of _cs_scores
        self._pi_config = None          # ForecastIntervals configuration

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Custom objective function for LightGBM training (subclass hook)."""
        raise NotImplementedError

    def _fit_from_predt(self, predt: np.ndarray, Z: torch.Tensor) -> torch.Tensor:
        """Compute the (gradient-free) fitted values for raw LightGBM outputs.

        Used by :meth:`eval_fn` to monitor the loss on the train/validation
        datasets without building an autograd graph.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM (flattened Fortran-order).
        Z : torch.Tensor
            Design matrix for the dataset being evaluated, shape ``(T_r, k * p)``.

        Returns
        -------
        torch.Tensor
            Fitted values, shape ``(k, T_r)``.
        """
        raise NotImplementedError

    def _forecast_params(self, features_df: pd.DataFrame) -> np.ndarray:
        """Map a feature frame to the ``(n_rows, n_params)`` coefficient matrix."""
        raise NotImplementedError

    def _num_class(self) -> int:
        """LightGBM output dimension; called after the datasets are built."""
        raise NotImplementedError

    def _post_datasets_setup(self, seed: int) -> None:
        """Model-specific setup that requires the panel dimensions.

        Called by :meth:`_train_core` after ``_build_panel_datasets`` has set
        ``self.k`` / ``self.n_params`` and before LightGBM training starts.
        The default is a no-op.

        Parameters
        ----------
        seed : int
            Random seed forwarded from ``train()``.
        """

    def _forecast_tree_embeddings(self, test_data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Build the ``type="tree_embeddings"`` output (HyperTreeNetVAR only)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training plumbing
    # ------------------------------------------------------------------
    def _validate_train_args(
            self, lgb_params, num_iterations, train_data, validation,
            early_stopping_round, seed, verbose, deterministic, forecast_intervals,
    ) -> None:
        """Common train() argument validation (mirrors the other models)."""
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

        required_columns = ["series_id", "date", "value"]
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")
        validate_series_order(train_data, name="train_data")

    def _reset_training_state(self) -> None:
        """Reset per-training state so instances can be retrained safely."""
        self.model = None
        self.dataset_references = {}
        self.is_trained = False
        self.features = None
        self._iter_count = 0
        self._Z_train = None
        self._Z_eval = None
        self._fcst_state = None
        self._scale_loc = None
        self._scale_scale = None
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

    def _fit_scaling(self, Y: np.ndarray) -> np.ndarray:
        """Fit per-series scaling statistics on the training panel and transform it.

        Stores ``self._scale_loc`` / ``self._scale_scale`` (training series
        order) and returns the scaled panel. With ``scaling=None`` the
        statistics are identity, so downstream code can always apply
        ``(Y - loc) / scale`` unconditionally.

        Parameters
        ----------
        Y : np.ndarray
            Raw value matrix, shape ``(T, k)``, columns in training series order.

        Returns
        -------
        np.ndarray
            Scaled value matrix, shape ``(T, k)``.
        """
        k = Y.shape[1]
        if self.scaling == "mean":
            loc = np.zeros(k)
            scale = np.abs(Y).mean(axis=0)
        elif self.scaling == "standard":
            loc = Y.mean(axis=0)
            scale = Y.std(axis=0)
        else:
            loc = np.zeros(k)
            scale = np.ones(k)

        self._scale_loc = loc
        # Guard against constant / all-zero series.
        self._scale_scale = np.where(scale > 1e-8, scale, 1.0)

        return (Y - self._scale_loc) / self._scale_scale

    def _build_panel_datasets(
            self,
            train_data: pd.DataFrame,
            validation: bool,
            early_stopping_round: Optional[int],
            free_raw_data: bool = True,
    ):
        """Pivot the panel, build the VAR design matrix, and create lgb datasets.

        Sets ``self.k``, ``self.n_params``, ``self.series_order_``,
        ``self.features``, ``self._Z_train`` / ``self._Z_eval`` and
        ``self.dataset_references``. When ``validation`` is True, the last
        ``fcst_h`` time steps of every series form the validation set.

        Parameters
        ----------
        train_data : pd.DataFrame
            Aligned panel of training data.
        validation : bool
            Whether to split off a validation set.
        early_stopping_round : int, optional
            Number of rounds for early stopping (validation only).
        free_raw_data : bool
            Whether to free raw data in the LightGBM datasets.

        Returns
        -------
        Tuple[List[lgb.Dataset], List[str], Optional[List], Optional[dict]]
            ``(valid_sets, valid_names, callbacks, evals_result)`` in the
            layout expected by ``lgb.train``.
        """
        Y, series_order, _ = _pivot_panel(train_data, name="train_data")
        T, k = Y.shape
        if k == 1:
            warnings.warn(
                "Only one series found. With k=1 the multivariate structure adds "
                "nothing (and the factor equals the series itself); consider "
                "HyperTreeAR instead."
            )
        if T <= self.p:
            raise ValueError(
                f"Series length ({T}) must exceed the lag order p={self.p}; "
                f"no training rows remain after lagging."
            )

        self.series_order_ = series_order
        self.k = k
        self.n_params = 2 * self.p if self.type == "factor" else k * self.p

        # Scale the panel; targets, design matrix, and the forecast recursion
        # all live in the scaled space, forecasts are transformed back.
        Y = self._fit_scaling(Y)

        if self.type == "factor":
            # Per-series design blocks [own lags j=1..p, factor lags j=1..p],
            # where the factor is the equal-weighted star variable of the
            # scaled panel; row r of a block corresponds to time t = r + p.
            factor = Y.mean(axis=1)
            factor_lags = np.column_stack(
                [factor[self.p - j: T - j] for j in range(1, self.p + 1)]
            )
            designs = [
                np.concatenate(
                    [
                        np.column_stack([Y[self.p - j: T - j, i] for j in range(1, self.p + 1)]),
                        factor_lags,
                    ],
                    axis=1,
                )
                for i in range(k)
            ]  # k blocks of shape (T - p, 2p)

            def design_slice(lo, hi):
                # Per-row design rows in dataset (series-major) order.
                return np.concatenate([d[lo:hi] for d in designs], axis=0)
        else:
            Z = _build_var_lags(Y, self.p)  # (T - p, k*p), row r <-> time t = r + p

            def design_slice(lo, hi):
                # One shared design row per time step.
                return Z[lo:hi]

        # Feature frame: drop the first p time steps of each series so feature
        # rows align 1:1 with the design-matrix rows.
        occ = train_data.groupby("series_id", sort=False).cumcount()
        feats = train_data[occ >= self.p].reset_index(drop=True)
        self.features = [c for c in feats.columns if c not in _RESERVED_COLUMNS]

        n_avail = T - self.p
        if validation:
            if n_avail <= self.fcst_h:
                raise ValueError(
                    f"Validation requires more than fcst_h={self.fcst_h} usable time "
                    f"steps per series, but only {n_avail} remain after lagging."
                )
            t_train = n_avail - self.fcst_h
        else:
            t_train = n_avail

        # Targets in dataset row order (series-major, time within series).
        label_train = Y[self.p: self.p + t_train].T.ravel()
        self._Z_train = torch.tensor(
            design_slice(0, t_train), dtype=self.dtype, device=self.device
        )

        row_t = feats.groupby("series_id", sort=False).cumcount()
        X_train = feats[row_t < t_train]
        dtrain = lgb.Dataset(
            data=X_train[self.features],
            label=label_train,
            free_raw_data=free_raw_data,
        )

        self.dataset_references = {id(dtrain): "train"}

        if validation:
            label_eval = Y[self.p + t_train:].T.ravel()
            self._Z_eval = torch.tensor(
                design_slice(t_train, n_avail), dtype=self.dtype, device=self.device
            )
            X_eval = feats[row_t >= t_train]
            deval = lgb.Dataset(
                data=X_eval[self.features],
                label=label_eval,
                free_raw_data=free_raw_data,
            )
            self.dataset_references[id(deval)] = "validation"

            evals_result = {}
            callbacks = [lgb.record_evaluation(evals_result)]
            if early_stopping_round is not None:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_round, verbose=False)
                )

            return [dtrain, deval], ["train", "validation"], callbacks, evals_result

        return [dtrain], ["train"], None, None

    def _compute_fit(self, params: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """VAR forward pass: per-equation inner product with the design vector.

        Deliberately computed with element-wise ops (broadcast multiply +
        sum) rather than ``torch.einsum``: einsum lowers to matmul kernels
        that honor ``torch.set_float32_matmul_precision`` -- which the
        experiments pipeline sets to ``"medium"`` -- and the reduced-precision
        backward/double-backward through this op systematically destabilizes
        HyperTreeNetVAR training (verified ~3x worse forecast errors across
        seeds on the ausretail benchmark). Element-wise ops always run in
        full float32, at the cost of materializing the ``(k, T_r, k * p)``
        product tensor.

        Parameters
        ----------
        params : torch.Tensor
            Coefficient matrix in dataset row order, shape ``(k * T_r, n_params)``.
        Z : torch.Tensor
            Design matrix: one shared row per time step ``(T_r, k * p)`` for
            the full design, or one row per observation ``(k * T_r, 2 * p)``
            for the factor design.

        Returns
        -------
        torch.Tensor
            Fitted values, shape ``(k, T_r)``.
        """
        if self.type == "factor":
            return (params * Z).sum(dim=1).reshape(self.k, -1)

        P = params.reshape(self.k, -1, self.n_params)  # (k, T_r, k*p)

        return (P * Z.unsqueeze(0)).sum(dim=2)

    def _train_core(
            self,
            lgb_params: dict,
            num_iterations: int,
            train_data: pd.DataFrame,
            validation: bool,
            early_stopping_round: Optional[int],
            seed: int,
            verbose: int,
            deterministic: bool,
            forecast_intervals: Optional[ForecastIntervals],
            model_factory: Callable[[], object],
            cal_train_kwargs: Dict,
    ) -> TrainingResult:
        """Shared training loop for both VAR variants.

        Validates the inputs, builds the panel datasets, runs the
        model-specific post-dataset setup (:meth:`_post_datasets_setup`),
        trains LightGBM with the subclass objective, and optionally
        calibrates conformal prediction intervals.

        Parameters
        ----------
        lgb_params, num_iterations, train_data, validation,
        early_stopping_round, seed, verbose, deterministic, forecast_intervals
            Forwarded verbatim from the public ``train()`` methods.
        model_factory : Callable[[], object]
            Zero-argument callable returning a fresh, untrained model with
            this instance's constructor configuration. Used by the conformal
            calibration to train per-window models.
        cal_train_kwargs : dict
            Keyword arguments forwarded to each calibration model's
            ``train()`` call.

        Returns
        -------
        TrainingResult
            Object containing evaluation results and training information.
        """
        self._validate_train_args(
            lgb_params, num_iterations, train_data, validation,
            early_stopping_round, seed, verbose, deterministic, forecast_intervals,
        )

        # Fail fast if any series is too short for the requested conformal
        # calibration. A VAR(p) needs at least p + 1 rows to retain one
        # training sample.
        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals, min_train=self.p + 1
            )

        if deterministic:
            run_lgb_params = {**lgb_params, "deterministic": True, "force_row_wise": True}
        else:
            run_lgb_params = dict(lgb_params)

        self._reset_training_state()

        try:
            valid_sets, valid_names, callbacks, evals_result = self._build_panel_datasets(
                train_data, validation, early_stopping_round
            )
            self._post_datasets_setup(seed)

            # General model parameters. The objective wrapper stops lgb.train's
            # params deepcopy from cloning this instance (see NoDeepcopyObjective).
            self.lgb_params = {
                "num_class": self._num_class(),
                "objective": NoDeepcopyObjective(self.objective_fn),
                "metric": "None",
                "random_seed": seed,
                "verbose": verbose,
            }
            self.lgb_params.update(run_lgb_params)

            # Anchor the forecast lag state at the end of the training panel.
            self.set_forecast_origin(train_data)

            start_time = time.time()
            self.model = lgb.train(
                self.lgb_params,
                valid_sets[0],
                num_boost_round=num_iterations,
                feval=self.eval_fn if validation else None,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            training_time = time.time() - start_time
            self.is_trained = True

            # Calibrate conformal prediction intervals via rolling-window CV.
            # Fresh model instances are trained per window (no forecast_intervals
            # passed, so there is no recursion) using the same hyper-parameters.
            if forecast_intervals is not None:
                self._cs_scores, self._cs_series_order = rolling_origin_residuals(
                    model_factory=model_factory,
                    train_data=train_data,
                    fcst_h=self.fcst_h,
                    forecast_intervals=forecast_intervals,
                    train_kwargs=cal_train_kwargs,
                )
                self._pi_config = forecast_intervals
                self._is_calibrated = True

            return TrainingResult(
                train_metrics=evals_result["train"] if validation else {"loss": []},
                validation_metrics=evals_result["validation"] if validation else None,
                best_iteration=self.model.best_iteration - 1
                if hasattr(self.model, "best_iteration") else num_iterations,
                training_time=training_time,
            )

        except Exception as e:
            self.is_trained = False
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def eval_fn(self, predt: np.ndarray, eval_data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        Custom evaluation function for evaluating forecast accuracy on an evaluation dataset.

        This function computes the loss value to be monitored during evaluation,
        selecting the design matrix that matches the dataset being evaluated.

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
        # Use the appropriate design matrix based on dataset name
        dataset_name = self.dataset_references.get(id(eval_data), "unknown")
        if dataset_name == "validation":
            Z = self._Z_eval
        else:
            # Default to the training design matrix if unknown
            if dataset_name == "unknown":
                warnings.warn("Unknown dataset in metric_fn. Using training design matrix.")
            Z = self._Z_train

        is_higher_better = False  # Lower loss is better, so we don't maximize
        target = torch.tensor(
            eval_data.get_label().reshape(self.k, -1), dtype=self.dtype, device=self.device
        )
        fit = self._fit_from_predt(predt, Z)
        loss = self.loss_fn(fit, target)

        return self.loss_name, loss.item(), is_higher_better

    # ------------------------------------------------------------------
    # Forecast plumbing
    # ------------------------------------------------------------------
    def set_forecast_origin(self, history: pd.DataFrame) -> None:
        """Re-anchor the VAR lag state to the end of *history* without retraining.

        Parameters
        ----------
        history : pd.DataFrame
            Aligned panel with ``series_id``, ``date``, ``value`` columns,
            ordered by ``(series_id, date)`` with contiguous series blocks.
            Must contain exactly the training series with at least ``p``
            observations each.
        """
        if self.series_order_ is None:
            raise RuntimeError("set_forecast_origin requires a trained model.")
        validate_series_order(history, name="history")
        Y, hist_order, _ = _pivot_panel(history, name="history")
        if set(hist_order) != set(self.series_order_):
            raise ValueError(
                f"history must contain exactly the training series. "
                f"Missing: {set(self.series_order_) - set(hist_order)}. "
                f"Extra: {set(hist_order) - set(self.series_order_)}."
            )
        if Y.shape[0] < self.p:
            raise ValueError(
                f"history must contain at least p={self.p} observations per series."
            )
        # Reorder columns to the training series order.
        idx = [hist_order.index(sid) for sid in self.series_order_]
        Y = Y[:, idx]
        # Scale with the training statistics so the lag state lives in the
        # same (scaled) space as the learned coefficients.
        Y = (Y - self._scale_loc) / self._scale_scale
        if self.type == "factor":
            # Own-lag state (k, p) and factor-lag state (p,), newest first.
            factor = Y.mean(axis=1)
            own_state = np.stack([Y[-j] for j in range(1, self.p + 1)], axis=1)
            factor_state = np.array([factor[-j] for j in range(1, self.p + 1)])
            self._fcst_state = (own_state, factor_state)
        else:
            # Lag state z = [y'_{T-1}, y'_{T-2}, ..., y'_{T-p}] (lag-major).
            self._fcst_state = np.concatenate([Y[-j] for j in range(1, self.p + 1)])

    def _validate_forecast_args(self, test_data, type, level) -> None:
        """Common forecast() validation.

        Parameters
        ----------
        test_data : pd.DataFrame
            Forecast input passed to ``forecast()``.
        type : str
            Requested output type.
        level : list of int, optional
            Requested conformal interval levels.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained. Call train() before forecasting.")
        for col in ["series_id", "date"]:
            if col not in test_data.columns:
                raise ValueError(f"Required column '{col}' not found in test_data")
        validate_series_order(test_data, name="test_data")

        # Validate series IDs match training data
        test_series_ids = list(dict.fromkeys(test_data["series_id"].tolist()))
        missing = set(test_series_ids) - set(self.series_order_)
        extra = set(self.series_order_) - set(test_series_ids)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"Missing series in training: {missing}")
            if extra:
                parts.append(f"Extra series not in test_data: {extra}")
            raise ValueError(
                ". ".join(parts) + ". A VAR forecast advances all series "
                "jointly, so test_data must contain exactly the training series."
            )

        if type not in self._valid_forecast_types:
            raise ValueError(f"Parameter 'type' must be one of {self._valid_forecast_types}.")

        if type == "forecast":
            rows_per_series = test_data.groupby("series_id", sort=False).size()
            bad = rows_per_series[rows_per_series != self.fcst_h]
            if not bad.empty:
                raise ValueError(
                    f"Each series must have exactly fcst_h={self.fcst_h} rows in test_data. "
                    f"Series with wrong counts: {bad.to_dict()}"
                )
            _validate_aligned_dates(test_data, name="test_data")

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

        missing_features = [f for f in self.features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in test_data: {missing_features}")

    def _model_name(self) -> str:
        """Model name identifier, reflecting the design variant.

        Returns
        -------
        str
            ``"<label>(p)"`` with ``VAR`` replaced by ``FactorVAR`` for the
            restricted design (e.g. ``"Hyper-Tree-FactorVAR(4)"``).
        """
        label = self._model_label
        if self.type == "factor":
            label = label.replace("VAR", "FactorVAR")

        return f"{label}({self.p})"

    def _forecast_recursion(self, P: np.ndarray) -> np.ndarray:
        """Roll the VAR recursion over the horizon (training order, scaled space).

        For the full design, the joint lag state is advanced with each step's
        forecasts as the new lag-1 block. For the factor design, the state is
        split into per-series own lags and the shared factor lags, and the
        factor is updated from the cross-sectional mean of each step's
        forecasts. The surrounding ``forecast()`` plumbing (validation, order
        mapping, finiteness check, de-normalization, interval columns) is
        shared.

        Parameters
        ----------
        P : np.ndarray
            Coefficient tensor in training series order,
            shape ``(k, fcst_h, n_params)``.

        Returns
        -------
        np.ndarray
            Point forecasts in the scaled space and training series order,
            shape ``(k, fcst_h)``. Overflow during the recursion is tolerated
            here; the caller checks finiteness and raises an actionable error.
        """
        forecasts = []
        with np.errstate(over="ignore", invalid="ignore"):
            if self.type == "factor":
                own_state, factor_state = self._fcst_state
                own_state = own_state.copy()
                factor_state = factor_state.copy()
                for h in range(self.fcst_h):
                    # y_t = Σ_j a_j y_{t-j} + Σ_j c_j f_{t-j}
                    next_val = (
                        (P[:, h, :self.p] * own_state).sum(axis=1)
                        + P[:, h, self.p:] @ factor_state
                    ).reshape(-1, 1)
                    forecasts.append(next_val)

                    # Shift the lag states; the new factor value is the
                    # cross-sectional mean of the forecasts.
                    own_state = np.concatenate([next_val, own_state[:, :-1]], axis=1)
                    factor_state = np.concatenate(
                        [[next_val.mean()], factor_state[:-1]]
                    )
            else:
                z = self._fcst_state.copy()
                for h in range(self.fcst_h):
                    # Compute next values using the VAR equation:
                    # y_t = A₁y_{t-1} + A₂y_{t-2} + ... + Aₚy_{t-p}
                    next_val = (P[:, h, :] @ z).reshape(-1, 1)
                    forecasts.append(next_val)

                    # Shift the lag state: next_val becomes the new lag-1 block.
                    z = np.concatenate([next_val.ravel(), z[:-self.k]])

        return np.hstack(forecasts)

    def _parameter_columns(self, out_df: pd.DataFrame, params_fcst: np.ndarray) -> None:
        """Append the named coefficient columns for ``type="parameters"``.

        Full design: column ``(j-1)*k + m`` is the coefficient on lag j of
        series m, named ``A{j}({series_id})``. Factor design: the own-lag
        block comes first, then the factor-lag block, named ``A{j}(own)``
        and ``A{j}(factor)``.

        Parameters
        ----------
        out_df : pd.DataFrame
            Output frame holding the ``series_id``/``date``/``model`` columns.
        params_fcst : np.ndarray
            Coefficients per row, shape ``(n_rows, n_params)``.
        """
        if self.type == "factor":
            for j in range(1, self.p + 1):
                out_df[f"A{j}(own)"] = params_fcst[:, j - 1].flatten()
            for j in range(1, self.p + 1):
                out_df[f"A{j}(factor)"] = params_fcst[:, self.p + j - 1].flatten()
        else:
            for j in range(1, self.p + 1):
                for m, sid in enumerate(self.series_order_):
                    out_df[f"A{j}({sid})"] = params_fcst[:, (j - 1) * self.k + m].flatten()

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast VAR coefficients for each test point
        2. Recursively generates forecasts by rolling the VAR recursion forward
           jointly for all series

        The forecasting process implements a vector autoregression where at each
        step the coefficient matrices forecasted from the test features are
        applied to the shared lag state, and the resulting vector of forecasts
        becomes the new lag-1 block:

        y_t = A₁(x)y_{t-1} + A₂(x)y_{t-2} + ... + Aₚ(x)y_{t-p}

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain exactly
            the training series with ``fcst_h`` aligned rows each (for
            ``type="forecast"``) and the same feature columns used during
            training.
        type : str
            Type of forecast to generate. Options:
            - "forecast": Generate forecasted values
            - "parameters": Return the VAR coefficients used for forecasting
              (one column per coefficient, named ``A{lag}({series_id})``)
            - "tree_embeddings": Return the tree embeddings (HyperTreeNetVAR only)
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
            - A{j}({series_id}): VAR coefficient values (if type="parameters")
            - <model>-lo-<level> / <model>-hi-<level>: prediction interval bounds
              (if type="forecast" and level is provided)
        """
        self._validate_forecast_args(test_data, type, level)
        model_name = self._model_name()

        try:
            if type == "forecast":
                params_fcst = self._forecast_params(test_data[self.features])
                # (k, fcst_h, k*p): axis 0 follows test row order, which the
                # series-set check above guarantees is a permutation of the
                # training series; the coefficient axis stays in training order.
                P = params_fcst.reshape(self.k, self.fcst_h, self.n_params)

                # The lag state, scaling statistics, and coefficient columns
                # are positional in *training* series order, so run the
                # recursion in that order and map back to the test row order
                # at the end.
                position = {sid: m for m, sid in enumerate(self.series_order_)}
                test_series_ids = list(dict.fromkeys(test_data["series_id"].tolist()))
                order_idx = np.array([position[sid] for sid in test_series_ids])
                P_train = np.empty_like(P)
                P_train[order_idx] = P

                # Generate multi-step forecasts via the model-specific
                # recursion (training order, scaled space). Finiteness is
                # checked once afterwards, replacing numpy's per-step
                # overflow warnings with an explicit error.
                point = self._forecast_recursion(P_train)  # (n_series, fcst_h)
                if not np.isfinite(point).all():
                    raise RuntimeError(
                        "The forecast recursion produced non-finite values, which "
                        "indicates diverged coefficients. With strongly correlated "
                        "series, the per-coefficient Newton steps can overshoot: "
                        "reduce the learning rate (a rule of thumb for the direct "
                        "VAR is eta / k) and keep per-series scaling enabled."
                    )

                # De-normalize to the original scale and reorder the series
                # axis back to the test row order.
                point = point * self._scale_scale[:, None] + self._scale_loc[:, None]
                point = point[order_idx]

                # Create output dataframe based on requested type
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": point.flatten(),
                    "model": model_name,
                })

                # Append conformal prediction intervals if requested.
                if level is not None:
                    columns = interval_columns(
                        point=point,
                        scores=self._cs_scores,
                        levels=level,
                        method=self._pi_config.method,
                        model_name=model_name,
                        cal_order=self._cs_series_order,
                        target_order=test_series_ids,
                    )
                    for col_name, values in columns.items():
                        out_df[col_name] = values

            elif type == "parameters":
                params_fcst = self._forecast_params(test_data[self.features])
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": model_name,
                })
                # Add the model-specific named coefficient columns.
                self._parameter_columns(out_df, params_fcst)

            else:  # "tree_embeddings" (HyperTreeNetVAR only)
                out_df = self._forecast_tree_embeddings(test_data, model_name)

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
