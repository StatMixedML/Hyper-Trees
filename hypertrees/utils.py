import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
import logging
from dataclasses import dataclass

@dataclass
class TrainingResult:
  """Dataclass to store the results of model training."""
  train_metrics: Dict[str, List[float]]
  validation_metrics: Optional[Dict[str, List[float]]] = None
  best_iteration: Optional[int] = None
  training_time: Optional[float] = None


def extract_forecast_lags(history: pd.DataFrame, p: int) -> dict:
    """Extract the last *p* values per series as the AR forecast seed.

    Parameters
    ----------
    history : pd.DataFrame
        Must contain ``series_id`` and ``value`` columns, ordered by
        ``(series_id, date)`` with each series in a contiguous block.
    p : int
        Number of AR lags.

    Returns
    -------
    dict
        ``{series_id: np.ndarray}`` with each array of length *p* in
        newest-first order, matching the convention used by ``forecast()``.
    """
    tail = history.groupby("series_id", sort=False).tail(p)
    return {
        sid: g["value"].to_numpy()[::-1]
        for sid, g in tail.groupby("series_id", sort=False)
    }


def validate_series_order(data: pd.DataFrame, name: str = "data") -> None:
    """
    Validate that a time series DataFrame is ordered so each series' rows are
    contiguous and chronologically sorted.

    Checks that:
    1. Every ``series_id`` occupies a single contiguous block of rows.
    2. Within each series, ``date`` is strictly increasing (gaps are allowed, but duplicate dates are not).

    These two conditions match the assumption made by every Hyper-Tree model's
    training and forecasting code, where
    ``model.predict(data[features]).reshape(n_series, fcst_h, ...)`` maps the
    first axis to series in first-appearance order.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to validate. Must contain ``series_id`` and ``date`` columns.
    name : str
        Label used in error messages (e.g. ``"train_data"`` or ``"test_data"``).

    Raises
    ------
    ValueError
        If the data is not properly ordered.
    """
    if "series_id" not in data.columns or "date" not in data.columns:
        raise ValueError(
            f"{name}: validate_series_order requires both 'series_id' and 'date' columns."
        )

    sid = data["series_id"]

    # Check 1: each series_id must occupy a single contiguous block of rows.
    is_new_run = sid != sid.shift()
    run_id = is_new_run.cumsum()
    runs_per_series = run_id.groupby(sid).nunique()
    bad = runs_per_series[runs_per_series > 1]
    if len(bad) > 0:
        raise ValueError(
            f"{name}: series_ids {list(bad.index)} appear in multiple non-contiguous "
            f"blocks. Each series must occupy a contiguous block of rows. Fix with "
            f"`{name} = {name}.sort_values(['series_id', 'date']).reset_index(drop=True)`."
        )

    # Check 2: within each series, `date` must be strictly increasing
    # (monotonic AND free of duplicates -- is_monotonic_increasing alone is
    # non-strict and would let duplicate dates through).
    for series_id, group in data.groupby("series_id", sort=False):
        dates = pd.to_datetime(group["date"])
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError(
                f"{name}: 'date' is not strictly increasing within series "
                f"'{series_id}' (out-of-order or duplicate dates). Sort with "
                f"`{name} = {name}.sort_values(['series_id', 'date']).reset_index(drop=True)` "
                f"and remove or aggregate duplicate dates."
            )

class NoDeepcopyObjective:
    """Keep a custom LightGBM objective bound to the live model instance.

    ``lgb.train`` deep-copies its ``params`` dict before extracting a callable
    objective, so a bound method would be cloned together with its whole model
    instance and boosting would train that hidden copy. This wrapper survives
    ``deepcopy``/``copy`` by reference, keeping the objective pointed at the
    original, live model.

    Parameters
    ----------
    fn : Callable
        The objective callable (typically ``self.objective_fn``).
    """

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger('lightgbm_custom')
        self.logger.setLevel(logging.ERROR)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        # Suppress warnings by not doing anything
        pass

    def error(self, message):
        self.logger.error(message)

class TimeSeriesPreprocessor:
    """
    A class for preprocessing time series data for Hyper-Tree models.

    This class handles frequency conversion, lag feature creation and data preparation
    for time series forecasting tasks. It encapsulates the preprocessing steps required
    for training Hyper-Tree models.
    """
    def __init__(self,
                 freq: str,
                 lags: List[int]
                 ) -> None:
        """
        Initialize the TimeSeriesPreprocessor.

        Arguments
        ----------
        freq : str
            Time series frequency (e.g., 'D', 'M', 'Q', 'Y').
        lags : List[int]
            List of lag values to use for feature creation.
            Should be integers as follows: [1, 2, 3] for lags of 1, 2, 3 time steps.

        Returns
        -------
        None
        """
        self.freq = self._convert_frequency(freq)
        self.lags = lags

    def _convert_frequency(self, freq: str) -> str:
        """
        Convert standard frequency format to Nixtla frequency format.

        Arguments
        ----------
        freq : str
            Standard frequency format (e.g., 'D', 'M', 'Q', 'Y').

        Returns
        -------
        str
            Converted frequency format compatible with Nixtla.
            For example, 'M' becomes 'MS', 'Q' becomes 'QS', etc.
        """
        # Mapping of standard frequencies to Nixtla/pandas frequencies
        if freq == "M":
            return "MS"  # Month start
        elif freq == "Q":
            return "QS"  # Quarter start
        elif freq == "Y":
            return "YS-JAN"  # Year start (January)
        else:
            return freq  # Keep as is for other frequencies

    def preprocess(self,
                   df: pd.DataFrame
                   ) -> pd.DataFrame:
        """
        Preprocess the time series data by creating a dataframe with lagged values.

        Arguments
        ----------
        df : pd.DataFrame
            Input DataFrame containing time series data with columns:
                - 'series_id': Unique identifier for each time series.
                - 'date': Date of the observation.
                - 'value': Target variable to forecast.


        Returns
        -------
        pd.DataFrame
            DataFrame with lagged values.

        Raises:
            ValueError: If required columns are missing from the input DataFrame.
        """
        required_columns = ["series_id", "date", "value"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Identify feature columns
        self.features = [col for col in df.columns if col not in required_columns]

        # Ensure contiguous ordering per series for correct shift() behavior
        result_df = df[["series_id", "date", "value"] + self.features].sort_values(
            ["series_id", "date"]
        ).reset_index(drop=True).copy()

        # Build lag columns in the order specified by self.lags (matches MLForecast convention)
        grouped = result_df.groupby("series_id", sort=False)["value"]
        lag_cols = []
        for lag in self.lags:
            col = f"lag{lag}"
            result_df[col] = grouped.shift(lag)
            lag_cols.append(col)

        # Drop rows where any lag is NaN (first max(lags) rows per series)
        result_df = result_df.dropna(subset=lag_cols).reset_index(drop=True)

        return result_df

    def extract(self,
                preprocessed_df: pd.DataFrame
                ) -> Dict[str, Any]:
        """
        Extract lags, target and features from the preprocessed DataFrame.

        Arguments
        ----------
        preprocessed_df : pd.DataFrame
            Preprocessed DataFrame containing lagged values and target variable.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - 'date': np.ndarray of observation dates.
                - 'lags_target': np.ndarray of lagged target values.
                - 'target': np.ndarray of target values.
                - 'features': pd.DataFrame of additional features.
        """

        # Extract
        result = {
            "date": preprocessed_df["date"].values,
            "lags_target": preprocessed_df.filter(regex=r"^lag\d+$").values,
            "target": preprocessed_df["value"].to_numpy().reshape(-1, 1),
            "features": preprocessed_df[self.features]
        }

        return result

    def create_lags(self,
                    df: pd.DataFrame,
                    ) -> pd.DataFrame:
        """
        Preprocess input data by converting frequencies and creating lag features.

        Applies frequency conversion, creates lagged columns (lag1, lag2, ...),
        and drops rows with NaN lags. The resulting DataFrame is suitable for
        passing to ``extract()`` or ``prepare_datasets()``.

        Arguments
        ----------
        df : pd.DataFrame
            Input DataFrame containing time series data with columns:
                - 'series_id': Unique identifier for each time series.
                - 'date': Date of the observation.
                - 'value': Target variable to forecast.
                All other columns are treated as exogenous features.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame with original columns plus lag columns.
        """
        preprocessed_df = self.preprocess(df)

        return preprocessed_df


class DatasetReferences(dict):
    """``id(dataset) -> name`` mapping that also pins the datasets.

    The mapping is keyed by object identity, which is only stable while the
    datasets stay alive: once a dataset is garbage-collected, CPython can
    reuse its memory address, and a later object would silently inherit its
    entry (observed as ``eval_fn`` misclassifying a foreign dataset as
    "train" instead of warning). Holding strong references means a key hit
    proves it is that exact, still-alive dataset -- the same pinning argument
    as HyperTreeETS's init cache. Behaves as a plain dict otherwise.

    Parameters
    ----------
    pairs : list of (lgb.Dataset, str)
        Datasets and their names ("train" / "validation").
    """

    def __init__(self, pairs):
        super().__init__({id(ds): name for ds, name in pairs})
        self._pinned = [ds for ds, _ in pairs]


def prepare_datasets(
    full_ts: pd.DataFrame,
    preprocessor: Any,  # TimeSeriesPreprocessor
    fcst_h: int,
    dtype: torch.dtype,
    validation: bool = True,
    early_stopping_round: Optional[int] = None,
    free_raw_data: bool = True
) -> Tuple[List[lgb.Dataset], List[str], Optional[List], Optional[Dict],
          torch.Tensor, Optional[torch.Tensor], Dict[int, str]]:
    """
    Prepare training and validation datasets for LightGBM.

    This function splits data into training and validation sets if requested,
    and performs necessary validation checks.

    Parameters
    ----------
    full_ts : pd.DataFrame
        Preprocessed time series data with features and lags
    preprocessor : TimeSeriesPreprocessor
        Preprocessor object used to extract features
    fcst_h : int
        Forecast horizon
    dtype : torch.dtype
        Data type for PyTorch tensors
    validation : bool
        Whether to create a validation set
    early_stopping_round : int, optional
        Number of rounds for early stopping
    free_raw_data : bool, default=True
        Whether to free raw data in LightGBM datasets

    Returns
    -------
    valid_sets : List[lgb.Dataset]
        List of LightGBM datasets for training and validation
    valid_names : List[str]
        Names of the datasets (e.g., "train", "validation")
    callbacks : List[Callable]
        Callbacks for LightGBM training (e.g., early stopping). Returns None if validation is False.
    evals_result : Dict[str, Any]
        Evaluation results from LightGBM training. Returns None if validation is False.
    lags_train : torch.Tensor
        Training lags tensor
    lags_eval : torch.Tensor, optional
        Evaluation lags tensor (None if validation is False)
    dataset_references : Dict[int, str]
        Dictionary mapping dataset IDs to their names
    """
    full_ts = full_ts.copy()
    full_ts.reset_index(drop=True, inplace=True)
    dataset_references = {}

    if validation:
        # Add validation callback to store evaluation results
        evals_result = {}
        callbacks = [lgb.record_evaluation(evals_result)]

        # Validate that fcst_h isn't larger than the minimum series length
        min_series_length = full_ts.groupby("series_id").size().min()
        if fcst_h >= min_series_length:
            raise ValueError(
                f"Forecast horizon (fcst_h={fcst_h}) must be smaller than the minimum series length ({min_series_length})")

        # Evaluation data
        eval_ts = full_ts.groupby("series_id").tail(fcst_h).copy()
        # Verify that we have the correct number of evaluation rows
        expected_eval_rows = len(full_ts["series_id"].unique()) * fcst_h
        if len(eval_ts) != expected_eval_rows:
            raise ValueError(
                f"Expected {expected_eval_rows} evaluation rows but got {len(eval_ts)}. Check for missing data in some series.")

        eval_dict = preprocessor.extract(eval_ts)
        features_eval = eval_dict["features"]
        target_eval = eval_dict["target"]
        lags_eval = torch.tensor(eval_dict["lags_target"], dtype=dtype)
        deval = lgb.Dataset(data=features_eval, label=target_eval.reshape(-1, ), free_raw_data=free_raw_data)

        # Training data
        train_ts = full_ts[~full_ts.index.isin(eval_ts.index)].copy()
        # Verify that training and evaluation sets together make up the full dataset
        if len(train_ts) + len(eval_ts) != len(full_ts):
            raise ValueError(
                f"Data split inconsistency. Train ({len(train_ts)}) + Eval ({len(eval_ts)}) != Full ({len(full_ts)})")

        train_dict = preprocessor.extract(train_ts)
        features_train = train_dict["features"]
        target_train = train_dict["target"]
        lags_train = torch.tensor(train_dict["lags_target"], dtype=dtype)
        dtrain = lgb.Dataset(data=features_train, label=target_train.reshape(-1, ), free_raw_data=free_raw_data)

        # Set up validation sets and names
        valid_sets = [dtrain, deval]
        valid_names = ["train", "validation"]

        # Store dataset references with names (pinned so the ids stay valid)
        dataset_references = DatasetReferences(
            [(dtrain, valid_names[0]), (deval, valid_names[1])]
        )

        if early_stopping_round is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_round,
                    verbose=False
                )
            )

        return valid_sets, valid_names, callbacks, evals_result, lags_train, lags_eval, dataset_references

    else:
        # Use all data for training if no evaluation is used
        full_dict = preprocessor.extract(full_ts)
        lags_full = full_dict["lags_target"]
        target_full = full_dict["target"]
        features_full = full_dict["features"]

        lags_train = torch.tensor(lags_full, dtype=dtype)
        dtrain = lgb.Dataset(data=features_full, label=target_full.reshape(-1, ), free_raw_data=free_raw_data)
        valid_sets = [dtrain]
        valid_names = ["train"]
        dataset_references = DatasetReferences([(dtrain, valid_names[0])])

        return valid_sets, valid_names, None, None, lags_train, None, dataset_references


class GaussNewtonHessian:
    """
    Gauss-Newton Hessian diagonal estimation via Hutchinson probing.

    Estimates diag(J^T B J) where J is the Jacobian of fitted values w.r.t.
    parameters and B is the diagonal loss curvature matrix, using Hutchinson's
    stochastic trace estimator with K random Gaussian probe vectors.

    For MSELoss, exploits the known constant curvature B = (2/N)*I.

    References
    ----------
    [1] Martens, J. (2020). New Insights and Perspectives on the Natural
        Gradient Method. JMLR, 21(146), 1-76.
    [2] Hutchinson, M. F. (1990). A Stochastic Estimator of the Trace of the
        Influence Matrix for Laplacian Smoothing Splines. Comm. Stat. Sim.
        Comp., 19(2), 433-450.

    Parameters
    ----------
    loss_fn : nn.Module
        PyTorch loss function. MSELoss triggers the optimized constant-curvature path.
    n_probes : int
        Number of Hutchinson probe vectors per estimate.
    dtype : torch.dtype
        Tensor data type for probe vectors.
    """

    def __init__(self, loss_fn: nn.Module, n_probes: int = 5, dtype: torch.dtype = torch.float32):
        self.loss_fn = loss_fn
        self.n_probes = n_probes
        self.dtype = dtype
        if isinstance(loss_fn, nn.MSELoss):
            self.estimate = self._mse
        else:
            self.estimate = self._general

    def _compute_loss_curvature(self, fit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element d^2L/d(y_hat)^2 via autograd.

        Parameters
        ----------
        fit : torch.Tensor
            Fitted values.
        target : torch.Tensor
            Target values, same shape as ``fit``.

        Returns
        -------
        torch.Tensor
            Detached per-element loss curvature, same shape as ``fit``.
        """
        fit_leaf = fit.detach().requires_grad_(True)
        loss_local = self.loss_fn(fit_leaf, target.detach())
        grad1 = autograd(loss_local, fit_leaf, create_graph=True)[0]
        loss_hess_diag = autograd(grad1.sum(), fit_leaf)[0]
        return loss_hess_diag.detach()

    def _mse(
        self,
        fit: torch.Tensor,
        target: torch.Tensor,
        params: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Hutchinson diagonal Hessian for MSELoss (constant curvature B = 2/N).

        Parameters
        ----------
        fit : torch.Tensor
            Fitted values (graph-connected to ``params``).
        target : torch.Tensor
            Target values (unused for MSE; curvature is constant).
        params : torch.Tensor
            Parameters w.r.t. which the Hessian diagonal is estimated.
        rng : torch.Generator
            Seeded generator for the Gaussian probe vectors.

        Returns
        -------
        torch.Tensor
            Estimated Hessian diagonal, same shape as ``params``.
        """
        N = fit.numel()
        hess_sum = torch.zeros_like(params)
        for _ in range(self.n_probes):
            z = torch.randn(fit.shape, generator=rng, dtype=self.dtype).to(fit.device)
            Jt_z = autograd((z * fit).sum(), params, retain_graph=True)[0]
            hess_sum += Jt_z ** 2
        return (2.0 / N) * hess_sum / self.n_probes

    def _general(
        self,
        fit: torch.Tensor,
        target: torch.Tensor,
        params: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Hutchinson diagonal Hessian for arbitrary twice-differentiable losses.

        Parameters
        ----------
        fit : torch.Tensor
            Fitted values (graph-connected to ``params``).
        target : torch.Tensor
            Target values, same shape as ``fit``.
        params : torch.Tensor
            Parameters w.r.t. which the Hessian diagonal is estimated.
        rng : torch.Generator
            Seeded generator for the Gaussian probe vectors.

        Returns
        -------
        torch.Tensor
            Estimated Hessian diagonal, same shape as ``params``.
        """
        loss_curv = self._compute_loss_curvature(fit, target)
        sqrt_curv = loss_curv.clamp(min=0.0).sqrt()
        hess_sum = torch.zeros_like(params)
        for _ in range(self.n_probes):
            z = torch.randn(fit.shape, generator=rng, dtype=self.dtype).to(fit.device)
            Jt_z = autograd((z * sqrt_curv * fit).sum(), params, retain_graph=True)[0]
            hess_sum += Jt_z ** 2
        return hess_sum / self.n_probes

