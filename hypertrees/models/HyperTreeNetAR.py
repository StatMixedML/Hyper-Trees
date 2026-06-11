import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import lightgbm as lgb
from typing import Tuple, Callable, Optional, List
import time
from ..utils import TimeSeriesPreprocessor, prepare_datasets, TrainingResult, CustomLogger, validate_series_order, extract_forecast_lags, GaussNewtonHessian, NoDeepcopyObjective
from ..conformal import (
    ForecastIntervals,
    validate_calibration_length,
    rolling_origin_residuals,
    interval_columns,
)
from .mlp import MLP
import warnings
lgb.register_logger(CustomLogger())

warnings.filterwarnings(
    "ignore",
    message="Using backward\\(\\) with create_graph=True will create a reference cycle.*"
)

class HyperTreeNetAR:
    """
    Class that implements a Hyper-TreeNet-AR(p) model for time series forecasting.

    It combines LightGBM with a neural network, where the LightGBM first creates embeddings from the input data which
    are then mapped as parameters to the target time series model. The Hyper-TreeNet-AR(p) model extends traditional
    autoregressive models by allowing the AR coefficients to be time-varying and estimated by a
    combination of neural network and gradient boosted trees. This creates a non-linear, adaptive autoregressive model
    that can capture complex temporal dependencies.

    Key features:
    - Combines LightGBM and a neural network for time series forecasting
    - Allows AR coefficients to vary based on features
    - Provides AR coefficients that can vary over time

    Use this model when:
    - You have relevant features that might influence the autoregressive structure
    - You want more flexibility than traditional AR models
    - You have a large number of AR(p) parameters to estimate, since GBDTs do not scale well with the number of parameters
    - You want to leverage the power of LightGBM for feature selection and representation learning
    - You want to use a neural network to learn the mapping from features to AR coefficients

    Example usage:
    ```python
    from hypertrees.models import HyperTreeNetAR
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 12
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeNetAR(
        p=lag_p,
        freq=frequency,
        fcst_h=fcst_h,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

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
        lgb_params={'learning_rate': 1e-1},
        network_params={
        'learning_rate': 1e-3,             # learning rate for the neural network optimizer
        'embedding_dimension': 1,          # embedding dimension for tree-embeddings
        'hidden_dim': 128,                 # hidden dimension for the MLP network
        'dropout': 0.1,                    # dropout rate for the MLP network
        'use_random_projection': True,     # whether to use random projections for the embeddings
        'rp_embed_dim': 12,                # dimension of the random projections (if used)
        },
        num_iterations=100,
        train_data=train,
        seed=123,
        verbose=-1
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
            device: str = "cpu",
            hessian_method: str = "exact",
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-TreeNet-AR(p) model.

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
            Default is MSE loss. Losses other than nn.MSELoss are not
            recommended, as they have not been systematically tested yet.
            nn.L1Loss is rejected (zero second derivative almost everywhere
            breaks Newton boosting).
        device : str
            Device to run the model on. Default is 'cpu'.
            This allows for GPU acceleration of network training if available.
        hessian_method : str
            Method for computing the Hessian diagonal. Options:
            - "exact": Exact diagonal Hessian via per-embedding-dimension
              second-order autograd (cheap, since the embedding is
              low-dimensional).
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
        if isinstance(loss_fn, nn.L1Loss):
            raise ValueError(
                "nn.L1Loss is not supported: its second derivative is zero almost "
                "everywhere, so LightGBM's Newton boosting receives all-zero Hessians "
                "and cannot grow trees. Use nn.HuberLoss or nn.SmoothL1Loss for an "
                "MAE-like loss with usable curvature."
            )
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
        self.device = device
        self.model = None
        self.features = None  # Stores feature names after training
        self.is_trained = False  # Flag to track if model has been trained
        self.dataset_references = {}  # Store references to LightGBM datasets
        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
        self._iter_count = 0
        self._fit = None
        self._target = None

        # Conformal prediction interval state
        self._is_calibrated = False
        self._cs_scores = None
        self._cs_series_order = None
        self._pi_config = None

        if hessian_method == "gn":
            self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts the raw LightGBM outputs to
        PyTorch tensors, computes the loss, and then backpropagates to get gradients.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the GBDT embeddings.
        data : lgb.Dataset
            LightGBM dataset containing the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians for LightGBM optimization.
        """
        self._iter_count += 1

        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=self.dtype, device=self.device)
        embeds, loss = self.get_embeds_loss(predt, target, self.lags_train, requires_grad=True)
        grad, hess = self.calculate_gradients_and_hessians(loss, embeds)

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
        is_higher_better = False  # Lower loss is better, so we don't maximize
        target = torch.tensor(eval_data.get_label().reshape(-1, 1), dtype=self.dtype, device=self.device)

        # For evaluation, we need to compute loss without any backward pass or gradient computation
        gbdt_embed = torch.tensor(
            predt.reshape(-1, self.embedding_dim, order="F"),
            dtype=self.dtype
        ).to(self.device)

        # self.network is the live module the objective updated in this same
        # boosting iteration (guaranteed by NoDeepcopyObjective).

        # Compute loss without gradients
        self.network.eval()
        with torch.no_grad():
            ar_params = self.network(gbdt_embed)
            fcst = torch.sum(ar_params * lags, dim=1).unsqueeze(1)
            loss = self.loss_fn(fcst, target)

        return self.loss_name, loss.item(), is_higher_better

    def get_embeds_loss_separate(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            lags: torch.Tensor = None,
            requires_grad: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into embeddings and calculate loss for separate gradients (Option 2).

        This function:
        1. Reshapes the raw outputs into tree embeddings
        2. Maps embeddings to AR parameters via the MLP
        3. Computes the forecast by multiplying AR parameters with lags and summing
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
            GBDT embeddings and loss value.
        """
        # Reshape outputs into embedding matrix (samples × embedding_dim)
        # The 'F' order means Fortran-style ordering (column-major)
        gbdt_embed = torch.tensor(
            predt.reshape(-1, self.embedding_dim, order="F"),
            requires_grad = requires_grad,
            dtype = self.dtype,
            device=self.device
        )

        # Train network (forward pass)
        self.network.train()
        ar_params_net = self.network(gbdt_embed)
        fcst_net = torch.sum(ar_params_net * lags, dim=1).unsqueeze(1)
        network_loss = self.loss_fn(fcst_net, target)
        self.optimizer.zero_grad()
        network_loss.backward()
        self.optimizer.step()

        # Calculate loss for GBDT
        self.network.eval()
        ar_params_gbdt = self.network(gbdt_embed)
        fcst_gbdt = torch.sum(ar_params_gbdt * lags, dim=1).unsqueeze(1)
        gbm_loss = self.loss_fn(fcst_gbdt, target)

        if self.hessian_method == "gn":
            self._fit = fcst_gbdt
            self._target = target

        return gbdt_embed, gbm_loss

    def get_embeds_loss_shared(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            lags: torch.Tensor = None,
            requires_grad: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into embeddings and calculate loss for shared gradients (Option 1).

        This function:
        1. Reshapes the raw outputs into tree embeddings
        2. Maps embeddings to AR parameters via the MLP
        3. Computes the forecast by multiplying AR parameters with lags and summing
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
            GBDT embeddings and shared loss.
        """
        # Clear existing gradients
        self.optimizer.zero_grad()
        self.network.train()

        # Reshape outputs into embedding matrix (samples × embedding_dim)
        # The 'F' order means Fortran-style ordering (column-major)
        predt = torch.tensor(
            predt.reshape(-1, self.embedding_dim, order="F"),
            dtype=self.dtype
        ).to(self.device)

        # Convert to PyTorch network parameter
        gbdt_embed = nn.Parameter(predt, requires_grad=requires_grad)

        # Forward pass of the network
        ar_params = self.network(gbdt_embed)
        fcst = torch.sum(ar_params * lags, dim=1, dtype=torch.float32).unsqueeze(1)
        shared_loss = self.loss_fn(fcst, target)

        if self.hessian_method == "gn":
            self._fit = fcst
            self._target = target

        # Back propagation for the network
        shared_loss.backward(create_graph=True)

        return gbdt_embed, shared_loss

    def _calculate_gradients_and_hessians_separate(self, loss: torch.Tensor, embeds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and hessians for LightGBM optimization using separate gradients (Option 2).

        This function computes first and second-order derivatives needed for
        gradient boosting optimization in LightGBM.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        embeds : torch.Tensor
            GBDT embeddings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        # Compute gradients
        grad = autograd(loss, inputs=embeds, create_graph=True)[0]

        # Compute hessians. We compute the diagonal of the Hessian matrix for each parameter separately
        hess = [
            autograd(grad[:, i].sum(), embeds, retain_graph=True)[0][:, i:(i + 1)]
            for i in range(self.embedding_dim)
        ]

        # Convert to numpy arrays and reshape as expected by LightGBM
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).cpu().detach().numpy().ravel(order="F")

        return grad, hess


    def _calculate_gradients_and_hessians_separate_gn(self, loss: torch.Tensor, embeds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian for separate gradient mode via Hutchinson probing.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        embeds : torch.Tensor
            GBDT embeddings, shape ``(n_samples, embedding_dim)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        grad = autograd(loss, inputs=embeds, retain_graph=True)[0]
        rng = torch.Generator().manual_seed(self._iter_count)
        hess = self._gn_hessian.estimate(self._fit, self._target, embeds, rng)
        self._fit = None
        self._target = None
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")

        return grad, hess

    def _calculate_gradients_and_hessians_shared(self, loss: torch.Tensor, embeds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and hessians for LightGBM optimization using shared gradients (Option 1).

        This function computes first and second-order derivatives needed for
        gradient boosting optimization in LightGBM.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model.
        embeds : torch.Tensor
            GBDT embeddings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        # Compute gradients
        grad = embeds.grad
        hess = [
            autograd(grad[:, i].sum(), embeds, retain_graph=True)[0][:, i:(i + 1)]
            for i in range(self.embedding_dim)
        ]

        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = torch.cat(hess, dim=1).cpu().detach().numpy().ravel(order="F")

        # Update network parameters
        self.optimizer.step()

        # Clear existing gradients to prevent accumulation
        embeds.grad = None
        self.optimizer.zero_grad()

        return grad, hess

    def _calculate_gradients_and_hessians_shared_gn(self, loss: torch.Tensor, embeds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Gauss-Newton Hessian for shared gradient mode via Hutchinson probing.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value from the model (already backpropagated by
            ``get_embeds_loss_shared``, which populates ``embeds.grad``).
        embeds : torch.Tensor
            GBDT embeddings, shape ``(n_samples, embedding_dim)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians as numpy arrays in the format expected by LightGBM.
        """
        grad = embeds.grad
        rng = torch.Generator().manual_seed(self._iter_count)
        hess = self._gn_hessian.estimate(self._fit, self._target, embeds, rng)
        self._fit = None
        self._target = None
        grad = grad.cpu().detach().numpy().ravel(order="F")
        hess = hess.cpu().detach().numpy().ravel(order="F")
        self.optimizer.step()
        embeds.grad = None
        self.optimizer.zero_grad()

        return grad, hess

    def train(
            self,
            lgb_params: dict = None,
            network_params: dict = None,
            gradient_mode: str = "separate",
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
        Train the Hyper-TreeNet-AR model on time series data.

        This method:
        1. Preprocesses the time series data to create lag features
        2. Sets up LightGBM datasets
        3. Trains the models

        The training data must contain columns:
        - 'series_id': Identifier for each time series
        - 'date': Timestamp for each observation
        - 'value': Target value to forecast
        - Additional feature columns used for forecasting

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters
        network_params : dict
            Network parameters. Available parameters are:
                - "learning_rate": Learning rate for the neural network optimizer
                - "hidden_dim": Dimension of the hidden layer in the MLP
                - "embedding_dimension": Dimension of the tree embeddings from LightGBM
                - "use_random_projection": Whether to use random projection for embeddings
                - "rp_embed_dim": Dimension of the random projection embeddings (if used)
                - "dropout": Dropout rate for regularization
        gradient_mode : str
            Gradient mode for MLP and GBM interaction. Options are:
                - "shared": Use shared gradients for both MLP and GBM (Option 1 in the paper)
                - "separate": Train MLP separately from GBM gradient computation (Option 2 in the paper)
            Default is "separate".
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
        if network_params is None:
            raise ValueError("network_params must be provided.")
        if gradient_mode not in ["shared", "separate"]:
            raise ValueError("gradient_mode must be either 'shared' or 'separate'.")
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must be a pandas DataFrame.")
        if not isinstance(lgb_params, dict):
            raise TypeError("lgb_params must be a dictionary.")
        if not isinstance(network_params, dict):
            raise TypeError("network_params must be a dictionary.")
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

        # Store the training mode
        self.gradient_mode = gradient_mode

        # Check required columns
        required_columns = ['series_id', 'date', 'value']
        for col in required_columns:
            if col not in train_data.columns:
                raise ValueError(f"Required column '{col}' not found in training data.")

        # Validate row ordering: each series must be a contiguous block with
        # monotonic dates so the training reshape and fcst_lags extraction align.
        validate_series_order(train_data, name="train_data")

        if forecast_intervals is not None:
            validate_calibration_length(
                train_data, self.fcst_h, forecast_intervals, min_train=self.p + 1
            )

        # Set the network and optimizer
        gbdt_params = lgb_params.copy()
        self.embedding_dim = network_params["embedding_dimension"]

        # Seed torch before constructing the MLP so initialization (and dropout
        # draws during training) are reproducible even when the random
        # projection layer -- whose constructor reseeds torch -- is disabled.
        torch.manual_seed(seed)

        self.network = MLP(
            tree_embed_dim=self.embedding_dim,
            output_dim=self.p,
            hidden_dim=network_params["hidden_dim"],
            use_random_projection=network_params["use_random_projection"],
            rp_embed_dim=network_params["rp_embed_dim"] if network_params["use_random_projection"] else None,
            dropout_rate=network_params["dropout"],
            seed=seed
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=network_params["learning_rate"])

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

        # Select objective function based on training mode and Hessian method
        if self.gradient_mode == "separate":
            self.get_embeds_loss = self.get_embeds_loss_separate
            if self.hessian_method == "exact":
                self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_separate
            else:
                self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_separate_gn
        elif self.gradient_mode == "shared":
            self.get_embeds_loss = self.get_embeds_loss_shared
            if self.hessian_method == "exact":
                self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_shared
            else:
                self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_shared_gn

        # GBDT parameters. The objective wrapper stops lgb.train's params
        # deepcopy from cloning this instance (see NoDeepcopyObjective).
        self.lgb_params = {
            "num_class": self.embedding_dim,
            "objective": NoDeepcopyObjective(self.objective_fn),
            "metric": "None",
            "random_seed": seed,
            "verbose": verbose
        }

        # Update with user-provided LightGBM parameters
        self.lgb_params.update(gbdt_params)

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
            self.lags_train = lags_train.to(self.device) if lags_train is not None else None
            self.lags_eval = lags_eval.to(self.device) if lags_eval is not None else None

            # Store lagged train values to be used in the forecast method
            self.set_forecast_origin(train_data)

            # Train model
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

            if forecast_intervals is not None:
                def _model_factory():
                    return HyperTreeNetAR(
                        p=self.p,
                        freq=self.freq,
                        fcst_h=self.fcst_h,
                        loss_fn=self.loss_fn,
                        device=self.device,
                        hessian_method=self.hessian_method,
                        n_hessian_probes=self.n_hessian_probes,
                    )

                cal_train_kwargs = dict(
                    lgb_params=lgb_params,
                    network_params=network_params,
                    gradient_mode=gradient_mode,
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
            raise RuntimeError(f"Training failed: {str(e)}") from e

    def set_forecast_origin(self, history: pd.DataFrame) -> None:
        """Re-anchor the AR lag seed to the end of *history* without retraining.

        Parameters
        ----------
        history : pd.DataFrame
            DataFrame with ``series_id``, ``date``, ``value`` columns, ordered
            by ``(series_id, date)`` with each series in a contiguous block.
        """
        validate_series_order(history, name="history")
        self.fcst_lags = extract_forecast_lags(history, self.p)

    def forecast(
            self,
            test_data: pd.DataFrame,
            type: str = "forecast",
            level: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        This method:
        1. Uses the trained model to forecast AR coefficients for each test point
        2. Recursively generates forecasted values using the forecasted AR coefficients

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
            - "tree_embeddings": Return the tree embeddings
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
            - AR(i) for i=1..p: AR coefficient values (if type="parameters")
            - tree_embedding_{i} for i=1..embedding_dim: GBDT tree-embedding dimensions (if type="tree_embeddings")
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

        # Validate rows per series matches forecast horizon (forecast only;
        # parameters/embeddings can be requested for arbitrary-length input).
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
        if type not in ["forecast", "parameters", "tree_embeddings"]:
            raise ValueError("Parameter 'type' must be either 'forecast', 'parameters' or 'tree_embeddings'.")

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
            # Get tree embeddings
            # Predict on the DataFrame (not .values) so pandas ``category``
            # dtype features keep their categorical encoding at forecast time.
            gbdt_embeds = torch.tensor(
                self.model.predict(test_data[self.features]),
                dtype=self.dtype,
                device=self.device
            ).reshape(-1, self.embedding_dim)

            # self.network holds this instance's trained weights (boosting
            # updated it in place; see NoDeepcopyObjective).
            self.network.eval()

            if type == "forecast":
                # Forecast parameters: (n_series, fcst_h, n_params)
                n_series_test = len(test_series_ids)
                with torch.no_grad():
                    params_fcst = (self.network(gbdt_embeds)
                                   .cpu()
                                   .detach()
                                   .numpy()
                                   .reshape(n_series_test, self.fcst_h, self.p))

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
                model_name = f"Hyper-TreeNet-AR({self.p})"
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "fcst": np.hstack(forecasts).flatten(),
                    "model": model_name,
                })

                if level is not None:
                    point = np.hstack(forecasts)  # (n_series, fcst_h)
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
                with torch.no_grad():
                    params_fcst = (self.network(gbdt_embeds)
                                   .cpu()
                                   .detach()
                                   .numpy()
                                   )

                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": f"Hyper-TreeNet-AR({self.p})",
                })
                # Add AR parameters to the dataframe
                for i in range(self.p):
                    out_df[f"AR({i + 1})"] = params_fcst[:, i].flatten()

            elif type == "tree_embeddings":
                out_df = pd.DataFrame({
                    "series_id": test_data["series_id"].to_numpy().flatten(),
                    "date": test_data["date"].to_numpy().flatten(),
                    "model": f"Hyper-TreeNet-AR({self.p})",
                })
                # Add tree embeddings to the dataframe
                for i in range(self.embedding_dim):
                    out_df[f"tree_embedding_{i + 1}"] = gbdt_embeds[:, i].cpu().numpy().flatten()

            return out_df

        except Exception as e:
            raise RuntimeError(f"Forecasting not successful: {str(e)}") from e
