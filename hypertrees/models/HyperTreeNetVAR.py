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
from .mlp import MLP

warnings.filterwarnings(
    "ignore",
    message="Using backward\\(\\) with create_graph=True will create a reference cycle.*"
)


class HyperTreeNetVAR(_HyperTreeVARBase):
    """
    Class that implements a Hyper-TreeNet-VAR(p) model for multivariate time series forecasting.

    It combines LightGBM with a neural network, where the LightGBM first creates
    embeddings from the input data which are then mapped as parameters to the
    target vector autoregression. The model learns the full time-varying
    coefficient matrices A_1(x), ..., A_p(x) over an aligned panel of k series
    so that y_{i,t} = sum_j A_j[i, :](x_{i,t}) . y_{t-j}, with cross-series
    dependence captured by the off-diagonal coefficients (see ``_var_base.py``
    for the formulation, data requirements, coefficient ordering, and the
    restricted ``type="factor"`` design).

    Training uses separated gradient flows (Option 2 in the paper, which the
    ablations found indistinguishable from the shared-flow Option 1): per
    boosting iteration the MLP takes one optimizer step on the current GBDT
    embeddings, then gradients and Hessians for the GBDT are computed through
    the updated network in inference mode. As in HyperTreeNetAR, the MLP
    decoder lives on the instance (``self.network``) and is updated in place
    during boosting; there is deliberately no shared/class-level network state.
    Unlike ``HyperTreeNetAR``, this model exposes no ``gradient_mode`` option;
    only the separated flow (Option 2) is implemented.

    Key features:
    - Combines LightGBM and a neural network for multivariate forecasting
    - Learns the time-varying lag matrices A_1, ..., A_p as functions of
      features (full k x k, or own + factor lags with type="factor")
    - Captures cross-series (Granger-causal) lead/lag dependencies via the
      off-diagonal coefficients
    - Boosting cost is independent of the number of VAR coefficients

    Use this model when:
    - Your series influence each other and forecasts should exploit
      cross-series lead/lag structure
    - The panel implies many coefficients (k * p beyond a few dozen), since
      GBDTs do not scale well with the number of parameters
    - You want to leverage LightGBM for feature encoding and a neural network
      for the mapping from embeddings to VAR coefficients

    Example usage:
    ```python
    # Imports
    from hypertrees.models.HyperTreeNetVAR import HyperTreeNetVAR
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    # Initialize model
    lag_p = 4
    frequency = 'M'
    fcst_h = 12
    model = HyperTreeNetVAR(
        p=lag_p,
        freq=frequency,
        fcst_h=fcst_h,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

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
        lgb_params={'learning_rate': 1e-1},
        network_params={
            'learning_rate': 1e-3,             # learning rate for the neural network optimizer
            'embedding_dimension': 1,          # embedding dimension for tree-embeddings
            'hidden_dim': 128,                 # hidden dimension for the MLP network
            'dropout': 0.1,                    # dropout rate for the MLP network
            'use_random_projection': True,     # whether to use random projections for the embeddings
            'rp_embed_dim': 8,                 # dimension of the random projections (if used)
        },
        num_iterations=100,
        train_data=train,
        seed=123,
        verbose=-1
    )

    # Generate forecasts
    forecasts = model.forecast(test_data=test)

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

    _model_label = "Hyper-TreeNet-VAR"
    _valid_forecast_types = ("forecast", "parameters", "tree_embeddings")

    def __init__(
            self,
            p: int = 2,
            freq: str = "M",
            fcst_h: int = 1,
            loss_fn: Callable = nn.MSELoss(),
            scaling: Optional[str] = "mean",
            type: str = "full",
            device: str = "cpu",
            hessian_method: str = "exact",
            n_hessian_probes: int = 5,
    ):
        """
        Initialize the Hyper-TreeNet-VAR(p) model.

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
              decoded by the MLP).
            - "factor": restricted GVAR-style design; every equation
              regresses on its own lags plus the lags of the equal-weighted
              cross-sectional average of the scaled panel (``2 * p``
              coefficients per equation, independent of k). Recommended for
              larger panels, where the unrestricted design overparameterizes.
              Parameter columns are named ``A{j}(own)`` / ``A{j}(factor)``
              and the model name becomes ``Hyper-TreeNet-FactorVAR(p)``.
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
        super().__init__(
            p=p,
            freq=freq,
            fcst_h=fcst_h,
            loss_fn=loss_fn,
            scaling=scaling,
            type=type,
        )
        if hessian_method not in ("exact", "gn"):
            raise ValueError("hessian_method must be either 'exact' or 'gn'.")
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

        self.device = device
        self.hessian_method = hessian_method
        self.n_hessian_probes = n_hessian_probes
        self.network = None
        self.optimizer = None
        self.embedding_dim = None
        self._network_params = None
        self._fit = None
        self._target = None
        if hessian_method == "gn":
            self._gn_hessian = GaussNewtonHessian(loss_fn, n_hessian_probes, self.dtype)

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training.

        This function defines the gradients and hessians for the LightGBM model
        based on the PyTorch loss function. It converts the raw LightGBM outputs
        to embeddings, updates the MLP, computes the loss through the updated
        network, and then backpropagates to get gradients.

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

        target = torch.tensor(
            data.get_label().reshape(self.k, -1), dtype=self.dtype, device=self.device
        )
        embeds, loss = self.get_embeds_loss_separate(predt, target, self._Z_train)
        grad, hess = self.calculate_gradients_and_hessians(loss, embeds)

        return grad, hess

    def get_embeds_loss_separate(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            Z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into embeddings and calculate loss for separate gradients (Option 2).

        This function:
        1. Reshapes the raw outputs into tree embeddings
        2. Maps embeddings to VAR coefficients via the MLP and takes one optimizer step
        3. Recomputes the coefficients through the updated network in inference mode
        4. Calculates the GBDT loss between fitted and actual values

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM.
        target : torch.Tensor
            Target values (actual time series values), shape ``(k, T_r)``.
        Z : torch.Tensor
            VAR design matrix, shape ``(T_r, k * p)`` for the full design or
            ``(k * T_r, 2 * p)`` for the factor design.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            GBDT embeddings and loss value.
        """
        # Reshape outputs into embedding matrix (samples × embedding_dim)
        # The 'F' order means Fortran-style ordering (column-major)
        gbdt_embed = torch.tensor(
            predt.reshape(-1, self.embedding_dim, order="F"),
            requires_grad=True,
            dtype=self.dtype,
            device=self.device
        )

        # Train network (forward pass)
        self.network.train()
        var_params_net = self.network(gbdt_embed)
        fit_net = self._compute_fit(var_params_net, Z)
        network_loss = self.loss_fn(fit_net, target)
        self.optimizer.zero_grad()
        network_loss.backward()
        self.optimizer.step()

        # Calculate loss for GBDT
        self.network.eval()
        var_params_gbdt = self.network(gbdt_embed)
        fit_gbdt = self._compute_fit(var_params_gbdt, Z)
        gbm_loss = self.loss_fn(fit_gbdt, target)

        if self.hessian_method == "gn":
            self._fit = fit_gbdt
            self._target = target

        return gbdt_embed, gbm_loss

    def _calculate_gradients_and_hessians_separate(
            self, loss: torch.Tensor, embeds: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _calculate_gradients_and_hessians_separate_gn(
            self, loss: torch.Tensor, embeds: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _fit_from_predt(self, predt: np.ndarray, Z: torch.Tensor) -> torch.Tensor:
        """Compute the fitted values for raw LightGBM embedding outputs.

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
        gbdt_embed = torch.tensor(
            predt.reshape(-1, self.embedding_dim, order="F"),
            dtype=self.dtype
        ).to(self.device)

        # self.network is the live module the objective updated in this same
        # boosting iteration (guaranteed by NoDeepcopyObjective).
        self.network.eval()
        with torch.no_grad():
            var_params = self.network(gbdt_embed)
            fit = self._compute_fit(var_params, Z)

        return fit

    def _num_class(self) -> int:
        """LightGBM output dimension: the tree-embedding dimension."""

        return self.embedding_dim

    def _reset_training_state(self) -> None:
        """Reset per-training state, including the network-specific parts."""
        super()._reset_training_state()
        self._fit = None
        self._target = None
        self.network = None
        self.optimizer = None

    def _post_datasets_setup(self, seed: int) -> None:
        """Construct the MLP decoder and optimizer once the panel dimensions are known.

        Called after ``_build_panel_datasets`` has set ``self.n_params``
        (``k * p`` for type="full", ``2 * p`` for type="factor"), which is
        the MLP output dimension.

        Parameters
        ----------
        seed : int
            Random seed forwarded from ``train()``.
        """
        network_params = self._network_params

        # Seed torch before constructing the MLP so initialization (and dropout
        # draws during training) are reproducible even when the random
        # projection layer -- whose constructor reseeds torch -- is disabled.
        torch.manual_seed(seed)

        self.network = MLP(
            tree_embed_dim=self.embedding_dim,
            output_dim=self.n_params,
            hidden_dim=network_params["hidden_dim"],
            use_random_projection=network_params["use_random_projection"],
            rp_embed_dim=network_params["rp_embed_dim"] if network_params["use_random_projection"] else None,
            dropout_rate=network_params["dropout"],
            seed=seed
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=network_params["learning_rate"])

    def train(
            self,
            lgb_params: dict = None,
            network_params: dict = None,
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
        Train the Hyper-TreeNet-VAR model on an aligned panel of time series.

        This method:
        1. Pivots the panel and builds the VAR design matrix
        2. Sets up LightGBM datasets (one row per series and time step)
        3. Trains the models

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
            LightGBM parameters
        network_params : dict
            Network parameters. Available parameters are:
                - "learning_rate": Learning rate for the neural network optimizer
                - "hidden_dim": Dimension of the hidden layer in the MLP
                - "embedding_dimension": Dimension of the tree embeddings from LightGBM
                - "use_random_projection": Whether to use random projection for embeddings
                - "rp_embed_dim": Dimension of the random projection embeddings (if used)
                - "dropout": Dropout rate for regularization
        num_iterations : int
            Number of boosting rounds for training
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
        if network_params is None:
            raise ValueError("network_params must be provided.")
        if not isinstance(network_params, dict):
            raise TypeError("network_params must be a dictionary.")
        required_net_keys = (
            "learning_rate", "embedding_dimension", "hidden_dim",
            "dropout", "use_random_projection",
        )
        missing_keys = [key for key in required_net_keys if key not in network_params]
        if missing_keys:
            raise ValueError(f"network_params is missing required keys: {missing_keys}")

        self.embedding_dim = network_params["embedding_dimension"]
        self._network_params = network_params

        # Select gradient computation based on Hessian method
        if self.hessian_method == "exact":
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_separate
        else:
            self.calculate_gradients_and_hessians = self._calculate_gradients_and_hessians_separate_gn

        def _model_factory():
            return HyperTreeNetVAR(
                p=self.p,
                freq=self.freq,
                fcst_h=self.fcst_h,
                loss_fn=self.loss_fn,
                scaling=self.scaling,
                type=self.type,
                device=self.device,
                hessian_method=self.hessian_method,
                n_hessian_probes=self.n_hessian_probes,
            )

        cal_train_kwargs = dict(
            lgb_params=lgb_params,
            network_params=network_params,
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

    def _tree_embeddings(self, features_df: pd.DataFrame) -> torch.Tensor:
        """Compute the GBDT tree embeddings for a feature frame.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature frame (training feature columns only).

        Returns
        -------
        torch.Tensor
            Tree embeddings, shape ``(n_rows, embedding_dim)``.
        """
        # Predict on the DataFrame (not .values) so pandas ``category``
        # dtype features keep their categorical encoding at forecast time.
        gbdt_embeds = torch.tensor(
            self.model.predict(features_df),
            dtype=self.dtype,
            device=self.device
        ).reshape(-1, self.embedding_dim)

        return gbdt_embeds

    def _forecast_params(self, features_df: pd.DataFrame) -> np.ndarray:
        """Forecast the ``(n_rows, n_params)`` coefficient matrix via GBDT + MLP.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature frame (training feature columns only).

        Returns
        -------
        np.ndarray
            VAR coefficients per row, shape ``(n_rows, n_params)``.
        """
        gbdt_embeds = self._tree_embeddings(features_df)

        # self.network holds this instance's trained weights (boosting
        # updated it in place; see NoDeepcopyObjective).
        self.network.eval()
        with torch.no_grad():
            params_fcst = (self.network(gbdt_embeds)
                           .cpu()
                           .detach()
                           .numpy())

        return params_fcst

    def _forecast_tree_embeddings(self, test_data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Build the ``type="tree_embeddings"`` output DataFrame.

        Parameters
        ----------
        test_data : pd.DataFrame
            Validated forecast input (features already injected).
        model_name : str
            Model name identifier for the ``model`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with one ``tree_embedding_{i}`` column per embedding
            dimension.
        """
        gbdt_embeds = self._tree_embeddings(test_data[self.features])

        out_df = pd.DataFrame({
            "series_id": test_data["series_id"].to_numpy().flatten(),
            "date": test_data["date"].to_numpy().flatten(),
            "model": model_name,
        })
        # Add tree embeddings to the dataframe
        for i in range(self.embedding_dim):
            out_df[f"tree_embedding_{i + 1}"] = gbdt_embeds[:, i].cpu().numpy().flatten()

        return out_df
