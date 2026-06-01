import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.autograd import grad as autograd
import time

from hypertrees.models import (
    HyperTreeETS,
    HyperTreeAR,
    HyperTreeNetAR,
)
from hypertrees.models.mlp import MLP
from sklearn.preprocessing import StandardScaler

from chronos import ChronosPipeline
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch import (
    DeepAREstimator,
    TemporalFusionTransformerEstimator
)

from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.compose import make_reduction
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.forecasting.compose import TransformedTargetForecaster

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA as SFAutoARIMA,
    ARIMA as SFARIMA,
    AutoETS as SFAutoETS,
)

from experiments.utils import (
    cleanup_lightning_artifacts,
    create_lag_features,
    to_offset_freq,
    to_period_freq,
    to_statsforecast_frames,
    to_statsforecast_output,
    col_order
)


def LightGBMForecast(
    params_lgb: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list
) -> pd.DataFrame:
        """
        Train a LightGBM model and generate forecasts.

        Parameters
        ----------
        params_lgb : dict
            Parameters for the LightGBM model.
        train : pd.DataFrame
            Training dataset containing the time series data.
        test : pd.DataFrame
            Test dataset for generating forecasts.
        features : list
            List of feature names to be used in the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the forecasted values.
        """
        # Dataset
        dtrain_lgb = lgb.Dataset(
            data=train[features].values,
            label=train["value"].to_numpy().reshape(-1,),
        )

        # Train model
        start = time.time()
        lgb_model = lgb.train(
            {k: v for k, v in params_lgb.items() if k not in ["num_boost_round", "use_time_index"]},
            dtrain_lgb,
            num_boost_round=params_lgb["num_boost_round"],
        )

        # Forecast
        lgb_fcst = pd.DataFrame.from_dict(
            {
                "series_id": test["series_id"],
                "date": test["date"],
                "fcst": lgb_model.predict(test[features]),
                "model": "LightGBM",
            }
        )
        end = time.time()
        runtime = (end - start) / 60
        lgb_fcst["runtime"] = runtime

        lgb_fcst = lgb_fcst[col_order]

        return lgb_fcst


def LightGBMARForecast(
        params_lgb: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        freq: str,
        fcst_h: int,
        lags: list,
        max_lag: int,
        n_series: int = 1
) -> pd.DataFrame:

    """
    Train an autoregressive LightGBM model and generate multi-step forecasts.

    Parameters
    ----------
    params_lgb : dict
        Parameters for the LightGBM model.
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    features : list
        List of feature names to be used in the model.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    lags : list
        List of lags to be used in the autoregressive model.
    max_lag : int
        Maximum lag to consider for the autoregressive model.
    n_series : int, optional
        Number of time series in the dataset (default is 1).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the autoregressive model.
    """

    # Create lag features
    preprocess_df = create_lag_features(train, lags=lags, features=features)
    train_lags = preprocess_df.filter(regex="lag").values
    train_targets = preprocess_df["value"].to_numpy().reshape(-1, 1)
    train_features = preprocess_df[features].values

    # Data
    dtrain_lgb_ar = lgb.Dataset(
        data=np.concatenate([train_features, train_lags], axis=1),
        label=train_targets.reshape(-1, )
    )

    # Train model
    start = time.time()
    lgb_ar_model = lgb.train(
        {k: v for k, v in params_lgb.items() if k not in ["num_boost_round", "use_time_index"]},
        dtrain_lgb_ar,
        num_boost_round=params_lgb["num_boost_round"],
    )

    # Forecast
    lgb_lags = train.groupby(["series_id"], sort=False).apply(lambda x: x["value"][-max_lag:][::-1]).reset_index(
        drop=True).to_numpy().reshape(n_series, -1)
    lgb_ar_fcsts = []
    x_test = test[features].to_numpy().reshape(n_series, fcst_h, -1)
    for h in range(fcst_h):
        next_val = lgb_ar_model.predict(np.concatenate([x_test[:, h, :], lgb_lags], axis=1)).reshape(-1, 1)
        lgb_ar_fcsts.append(next_val)
        lgb_lags = np.concatenate([next_val, lgb_lags[:, :-1]], axis=1)

    end = time.time()
    runtime = (end - start) / 60

    lgb_ar_fcst = pd.DataFrame.from_dict(
        {
            "series_id": test["series_id"].to_numpy().flatten(),
            "date": test["date"].to_numpy().flatten(),
            "fcst": np.hstack(lgb_ar_fcsts).flatten(),
            "runtime": runtime,
            "model": f"LightGBM-AR({max_lag})",
        }
    )

    return lgb_ar_fcst


def LightGBMSTLForecast(
        params_lgb: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        fcst_h: int,
        lags: list
) -> pd.DataFrame:
    """
    Train a LightGBM model with STL decomposition and generate forecasts.

    Parameters
    ----------
    params_lgb : dict
        Parameters for the LightGBM model.
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    features : list
        List of feature names to be used in the model.
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    lags : list
        List of lags to be used in the autoregressive model.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the LightGBM-STL model.
    """
    regressor = lgb.LGBMRegressor(
        **{k: v for k, v in params_lgb.items() if k not in ["num_boost_round", "use_time_index", "degree"]},
        n_estimators=params_lgb["num_boost_round"])
    kwargs = {
        "lag_feature": {
            "lag": [i + 1 for i in range(max(lags))],
        }
    }
    forecaster = TransformedTargetForecaster(
        [
            ("deseasonalise", Deseasonalizer(model="multiplicative", sp=max(lags))),
            ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=params_lgb["degree"]))),
            ("forecast", make_reduction(
                regressor,
                transformers=[WindowSummarizer(**kwargs, n_jobs=-1)],
                window_length=None,
                strategy="recursive",
                pooling="global",
            ),
             ),
        ]
    )
    fh = [i + 1 for i in range(fcst_h)]
    start = time.time()
    forecaster.fit(train["value"].to_numpy().reshape(-1, ), train[features].values, fh=fh)
    stl_fcst = forecaster.predict(fh, test[features].values)
    end = time.time()
    runtime = (end - start) / 60

    lgb_stl_fcst = pd.DataFrame({
        "series_id": test["series_id"].to_numpy().flatten(),
        "date": test["date"].to_numpy().flatten(),
        "fcst": stl_fcst.flatten(),
        "runtime": runtime,
        "model": f"LightGBM-STL",
    })

    return lgb_stl_fcst


def create_gluonts_dataset(
        train: pd.DataFrame,
        test: pd.DataFrame,
        meta: dict,
        freq: str
) -> tuple:
    """
    Create GluonTS ListDatasets for training and testing.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    meta : dict
        Metadata containing information about the dataset, such as dynamic and static covariates.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily, 'M' for monthly).

    Returns
    -------
    tuple of (ListDataset, ListDataset)
        Training and test GluonTS datasets.
    """

    # GluonTS uses pandas Period frequencies; normalize in case the caller
    # passed a DateOffset-style freq (e.g. "MS", "QS", "AS-JAN").
    freq_gluonts = to_period_freq(freq)

    # Train
    list_data_train = [i for _, i in train.groupby("series_id")]
    target_train = [np.array(item["value"]) for item in list_data_train]
    start_train = [pd.Timestamp(item["date"].min()) for item in list_data_train]
    series_id_train = [str(item["series_id"].iloc[0]) for item in list_data_train]
    feat_dynamic_real_train = [np.array(item[meta["dynamic_cov"] + meta["time_derived_feats"]]).T for item in
                               list_data_train]
    feat_static_real_train = np.array([np.array(item[meta["static_cov"]])[0] for item in list_data_train])
    train_ds = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.ITEM_ID: series_id,
        FieldName.FEAT_DYNAMIC_REAL: feat_dynamic_real,
        FieldName.FEAT_STATIC_REAL: feat_static_real
    } for
        target,
        start,
        series_id,
        feat_dynamic_real,
        feat_static_real
        in zip(
            target_train,
            start_train,
            series_id_train,
            feat_dynamic_real_train,
            feat_static_real_train
        )], freq=freq_gluonts)

    # Test
    gluonts_df = pd.concat([train, test], axis=0, ignore_index=True)
    list_data_test = [i for _, i in gluonts_df.groupby("series_id")]
    target_test = [np.array(item["value"]) for item in list_data_test]
    start_test = [pd.Timestamp(item["date"].min()) for item in list_data_test]
    series_id_test = [str(item["series_id"].iloc[0]) for item in list_data_test]
    feat_dynamic_real_test = [np.array(item[meta["dynamic_cov"] + meta["time_derived_feats"]]).T for item in
                              list_data_test]
    feat_static_real_test = np.array([np.array(item[meta["static_cov"]])[0] for item in list_data_test])
    test_ds = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.ITEM_ID: series_id,
        FieldName.FEAT_DYNAMIC_REAL: feat_dynamic_real,
        FieldName.FEAT_STATIC_REAL: feat_static_real,
    } for
        target,
        start,
        series_id,
        feat_dynamic_real,
        feat_static_real
        in zip(
            target_test,
            start_test,
            series_id_test,
            feat_dynamic_real_test,
            feat_static_real_test
        )], freq=freq_gluonts)

    return train_ds, test_ds


def DeepARForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        meta: dict,
        freq: str,
        fcst_h: int,
        lags: list,
        max_lag: int,
        config: dict,
        series_ids: list,
        device: torch.device
) -> pd.DataFrame:
    """
    Train a DeepAR model via GluonTS and generate forecasts.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    meta : dict
        Metadata containing information about the dataset, such as dynamic and static covariates.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily, 'M' for monthly).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    lags : list
        List of lags to be used in the autoregressive model.
    max_lag : int
        Maximum lag to consider for the autoregressive model.
    config : dict
        Configuration dictionary containing parameters for the model.
    series_ids : list
        List of series IDs for which forecasts are to be generated.
    device : torch.device
        Device to be used for training the model (e.g., 'cuda' or 'cpu').

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the model.
    """

    freq_gluonts = to_period_freq(freq)

    # Create GluonTS datasets
    train_ds, test_ds = create_gluonts_dataset(train, test, meta, freq_gluonts)

    # Initialize model
    deepar_model = DeepAREstimator(
        freq=freq_gluonts,
        prediction_length=fcst_h,
        context_length=2 * fcst_h,
        num_feat_dynamic_real=len(meta["dynamic_cov"] + meta["time_derived_feats"]),
        num_feat_static_real=len(meta["static_cov"]) if len(meta["static_cov"]) > 0 else 0,
        time_features=[],  # disable built-in time features and use the sames as in the HyperTree models
        lags_seq=lags,  # disable built-in lags and use the sames as in the HyperTree models
        nonnegative_pred_samples=True,
        scaling=True,
        lr=config["deep_learning"]["learning_rate"],
        batch_size=config["deep_learning"]["batch_size"],
        num_layers=config["deep_learning"]["num_layers"],
        hidden_size=config["deep_learning"]["hidden_size"],
        trainer_kwargs=dict(
            accelerator=device.type,
            max_epochs=config["deep_learning"]["num_epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
    )

    # Train model
    start = time.time()
    deepar_predictor = deepar_model.train(train_ds)

    # Forecast
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=deepar_predictor,
        num_samples=config["deep_learning"]["num_samples"],
    )
    forecasts_deepar = list(forecast_it)
    fcst_df_list = []
    for i in range(len(forecasts_deepar)):
        fcst_df_list.append(
            pd.DataFrame.from_dict(
                {
                    "series_id": forecasts_deepar[i].item_id,
                    "fcst": np.round(forecasts_deepar[i].samples.mean(axis=0)).flatten(),
                }
            )
        )
    deepar_fcst_df = pd.concat(fcst_df_list, axis=0, ignore_index=True)

    end = time.time()
    runtime = (end - start) / 60

    deepar_fcst_list = []
    for series in series_ids:
        test_gluonts = test[test["series_id"] == series]
        deepar_fcst = deepar_fcst_df[deepar_fcst_df["series_id"] == str(series)].copy()
        deepar_fcst_list.append(
            pd.DataFrame.from_dict({
                "series_id": series,
                "date": pd.to_datetime(test_gluonts["date"].values),
                "fcst": np.array(deepar_fcst["fcst"].values).flatten(),
                "runtime": runtime,
                "model": f"Deep-AR({max_lag})",
            })
        )

    deepar_fcst = pd.concat(deepar_fcst_list, axis=0)

    cleanup_lightning_artifacts()

    return deepar_fcst


def TFTForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        meta: dict,
        freq: str,
        fcst_h: int,
        config: dict,
        series_ids: list,
        device: torch.device
) -> pd.DataFrame:
    """
    Train a Temporal Fusion Transformer via GluonTS and generate forecasts.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    meta : dict
        Metadata containing information about the dataset, such as dynamic and static covariates.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily, 'M' for monthly).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    config : dict
        Configuration dictionary containing parameters for the model.
    series_ids : list
        List of series IDs for which forecasts are to be generated.
    device : torch.device
        Device to be used for training the model (e.g., 'cuda' or 'cpu').

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the model.
    """

    freq_gluonts = to_period_freq(freq)

    # Create GluonTS datasets
    train_ds, test_ds = create_gluonts_dataset(train, test, meta, freq_gluonts)

    # Initialize model
    dynamic_dims = len(meta["dynamic_cov"] + meta["time_derived_feats"])
    tft_model = TemporalFusionTransformerEstimator(
        freq=freq_gluonts,
        prediction_length=fcst_h,
        context_length=2 * fcst_h,
        dynamic_dims=[dynamic_dims] if dynamic_dims > 0 else [],
        static_dims=[len(meta["static_cov"])] if len(meta["static_cov"]) > 0 else [],
        time_features=[],  # disable built-in time features and use the sames as in the HyperTree models
        quantiles=config["deep_learning"]["quantiles_tft"],
        num_heads=config["deep_learning"]["num_heads"],
        hidden_dim=config["deep_learning"]["hidden_size"],
        variable_dim=config["deep_learning"]["variable_dim"],
        batch_size=config["deep_learning"]["batch_size"],
        lr=config["deep_learning"]["learning_rate"],
        trainer_kwargs=dict(
            accelerator=device.type,
            max_epochs=config["deep_learning"]["num_epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
    )

    # Train model
    start = time.time()
    tft_predictor = tft_model.train(train_ds)

    # Forecast
    forecast_it_tft, ts_it_tft = make_evaluation_predictions(
        dataset=test_ds,
        predictor=tft_predictor,
    )
    forecasts_tft = list(forecast_it_tft)
    fcst_df_list_tft = []
    for i in range(len(forecasts_tft)):
        fcst_df_list_tft.append(
            pd.DataFrame.from_dict(
                {
                    "series_id": forecasts_tft[i].item_id,
                    "fcst": np.round(forecasts_tft[i]["p50"]).flatten(),
                }
            )
        )
    tft_fcst_df = pd.concat(fcst_df_list_tft, axis=0, ignore_index=True)

    end = time.time()
    runtime = (end - start) / 60

    tft_fcst_list = []
    for series in series_ids:
        test_tft = test[test["series_id"] == series]
        tft_fcst = tft_fcst_df[tft_fcst_df["series_id"] == str(series)].copy()
        tft_fcst_list.append(
            pd.DataFrame.from_dict({
                "series_id": series,
                "date": pd.to_datetime(test_tft["date"].values),
                "fcst": np.array(tft_fcst["fcst"].values).flatten(),
                "runtime": runtime,
                "model": "TFT",
            })
        )

    tft_fcst = pd.concat(tft_fcst_list, axis=0)

    cleanup_lightning_artifacts()

    return tft_fcst

def ChronosForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        series_ids: list,
        fcst_h: int,
        config: dict,
        device: torch.device
) -> pd.DataFrame:
    """
    Generate forecasts using the pretrained Chronos foundation model.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    series_ids : list
        List of series IDs for which forecasts are to be generated.
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    config : dict
        Configuration dictionary containing parameters for the model.
    device : torch.device
        Device to be used for training the model (e.g., 'cuda' or 'cpu').

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the model.
    """
    # Initialize model
    chronos_model = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map=device,
        torch_dtype=torch.float32,
    )

    # Forecast
    chronos_fcst_list = []
    start = time.time()
    for series in series_ids:
        train_chronos = train[train["series_id"] == series]
        test_chronos = test[test["series_id"] == series]
        context = torch.tensor(train_chronos[train_chronos["series_id"] == series]["value"].to_numpy().reshape(-1, ))
        fcsts = np.array(
            chronos_model.predict(
                context,
                prediction_length=fcst_h,
                num_samples=config["deep_learning"]["num_samples_chronos"]
            )
        )
        chronos_fcst_list.append(
            pd.DataFrame.from_dict({
                "series_id": series,
                "date": pd.to_datetime(test_chronos["date"].values),
                "fcst": np.mean(fcsts, axis=1).flatten(),
                "model": "Chronos",
            })
        )
    end = time.time()
    runtime = (end - start) / 60
    chronos_fcst = pd.concat(chronos_fcst_list, axis=0)

    chronos_fcst["runtime"] = runtime
    chronos_fcst = chronos_fcst[col_order]

    return chronos_fcst


def HyperTreeARForecast(
        htar_params: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        freq: str,
        fcst_h: int,
        max_lag: int,
        loss_fn: nn.Module,
        seed: int,
        return_extras: bool = False,
) -> pd.DataFrame:
    """
    Train a Hyper-Tree-AR model and generate forecasts.

    Parameters
    ----------
    htar_params : dict
        Parameters for the HyperTree autoregressive model.
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    features : list
        List of feature names to be used in the model.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    max_lag : int
        Maximum lag to consider for the autoregressive model.
    loss_fn : nn.Module
        Loss function to be used for training the model.
    seed : int
        Random seed for reproducibility.
    return_extras : bool, default False
        If True, also return learned AR parameters over the full
        (train + test) date range.

    Returns
    -------
    pd.DataFrame or tuple
        ``fcst_df`` when ``return_extras=False``; otherwise
        ``(fcst_df, params_df)``.
    """
    # Initialize model
    ht_ar = HyperTreeAR(
        p=max_lag,
        freq=freq,
        fcst_h=fcst_h,
        loss_fn=loss_fn,
    )

    # Train model
    start = time.time()
    ht_ar.train(
        lgb_params={k: v for k, v in htar_params.items() if k != "num_boost_round"},
        num_iterations=htar_params["num_boost_round"],
        train_data=train[["series_id", "date", "value"] + features],
        seed=seed,
    )

    # Forecast
    ht_ar_fcst = ht_ar.forecast(
        test_data=test[["series_id", "date"] + features]
    )

    end = time.time()
    runtime = (end - start) / 60

    ht_ar_fcst["runtime"] = runtime
    ht_ar_fcst = ht_ar_fcst[col_order]

    if return_extras:
        full_range = pd.concat(
            [
                train[["series_id", "date"] + features],
                test[["series_id", "date"] + features],
            ],
            axis=0, ignore_index=True,
        ).sort_values(["series_id", "date"]).reset_index(drop=True)
        params_df = ht_ar.forecast(test_data=full_range, type="parameters")
        params_df["model"] = "Hyper-Tree-AR"
        return ht_ar_fcst, params_df

    return ht_ar_fcst


def HyperTreeNetARForecast(
        htnetar_params: dict,
        htnetar_params_lgb: dict,
        htnetar_network_params: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        freq: str,
        fcst_h: int,
        max_lag: int,
        loss_fn: nn.Module,
        seed: int,
        device: torch.device,
        return_extras: bool = False,
) -> pd.DataFrame:

    """
    Train a Hyper-TreeNet-AR model and generate forecasts.

    Parameters
    ----------
    htnetar_params : dict
        Parameters for the HyperTreeNet autoregressive model.
    htnetar_params_lgb : dict
        Parameters for the LightGBM model used in HyperTreeNet.
    htnetar_network_params : dict
        Parameters for the MLP used in HyperTreeNet.
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    features : list
        List of feature names to be used in the model.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    max_lag : int
        Maximum lag to consider for the autoregressive model.
    loss_fn : nn.Module
        Loss function to be used for training the model.
    seed : int
        Random seed for reproducibility.
    device : torch.device
        Device to be used for training the model (e.g., 'cuda' or 'cpu').
    return_extras : bool, default False
        If True, also return learned AR parameters and tree embeddings
        over the full (train + test) date range.

    Returns
    -------
    pd.DataFrame or tuple
        ``fcst_df`` when ``return_extras=False``; otherwise
        ``(fcst_df, params_df, embeddings_df)``.
    """
    # Initialize model
    htnet_ar = HyperTreeNetAR(
        p=max_lag,
        freq=freq,
        fcst_h=fcst_h,
        loss_fn=loss_fn,
        device=device
    )

    # Train model
    start = time.time()
    htnet_ar.train(
        lgb_params=htnetar_params_lgb,
        network_params=htnetar_network_params,
        gradient_mode="separate",
        num_iterations=htnetar_params["num_boost_round"],
        train_data=train[["series_id", "date", "value"] + features],
        seed=seed,
    )

    # Forecast
    htnet_ar_fcst = htnet_ar.forecast(
        test_data=test[["series_id", "date"] + features]
    )

    end = time.time()
    runtime = (end - start) / 60
    htnet_ar_fcst["runtime"] = runtime

    # Arrange columns
    htnet_ar_fcst = htnet_ar_fcst[col_order]

    if return_extras:
        full_range = pd.concat(
            [
                train[["series_id", "date"] + features],
                test[["series_id", "date"] + features],
            ],
            axis=0, ignore_index=True,
        ).sort_values(["series_id", "date"]).reset_index(drop=True)
        params_df = htnet_ar.forecast(test_data=full_range, type="parameters")
        params_df["model"] = "Hyper-TreeNet-AR"
        embeddings_df = htnet_ar.forecast(test_data=full_range, type="tree_embeddings")
        embeddings_df["model"] = "Hyper-TreeNet-AR"
        return htnet_ar_fcst, params_df, embeddings_df

    return htnet_ar_fcst

def HyperTreeETSForecast(
        htets_params: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        freq: str,
        fcst_h: int,
        seasonality_ets: str,
        loss_fn: nn.Module,
        manual_param: float = None
) -> pd.DataFrame:
    """
    Train a Hyper-Tree-ETS model and generate forecasts.

    Parameters
    ----------
    htets_params : dict
        Parameters for the HyperTreeETS model.
    train : pd.DataFrame
        Training dataset containing the time series data.
    test : pd.DataFrame
        Test dataset for generating forecasts.
    features : list
        List of feature names to be used in the model.
    freq : str
        Frequency of the time series data (e.g., 'D' for daily).
    fcst_h : int
        Forecast horizon, i.e., the number of time steps to forecast.
    seasonality_ets : str
        Feature to be used for seasonality.
    loss_fn : nn.Module
        Loss function to be used for training the model.
    manual_param : float, optional
        Manual parameter for the ETS model (default is None). If provided, the model is not trained and
        forecasts are generated directly using the provided parameter.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the forecasted values from the model.
    """
    start = time.time()
    if htets_params["train"]:
        # Warn on zero / near-zero values only when the multiplicative ETS
        # variant is used (``ets_type == "triple"`` in our implementation),
        # because that's where divergence actually happens.
        if htets_params["ets_type"] == "triple":
            _min = float(train["value"].min())
            if _min <= 1e-6:
                import warnings
                warnings.warn(
                    f"Hyper-Tree-ETS Forecast (ets_type='triple'): "
                    f"train['value'].min() = {_min:.6g} is zero or near-zero. "
                    "Multiplicative ETS components may diverge. Consider "
                    "enabling ``htets_params['scaling']`` or preprocessing "
                    "the series to be strictly positive.",
                    stacklevel=2,
                )

        # Copy to avoid modifying the caller's DataFrames
        train = train.copy()
        test = test.copy()

        # Scaling
        if htets_params["scaling"]:
            scaling_factor = np.mean(train["value"])
            train["value"] = train["value"] + scaling_factor

        # Initialize model
        ht_ets = HyperTreeETS(
            ets_type=htets_params["ets_type"],
            seasonality_feature=seasonality_ets,
            season_length=htets_params["season_length"],
            freq=freq,
            fcst_h=fcst_h,
            loss_fn=loss_fn
        )

        # Train model
        if manual_param is None:
            ht_ets.train(
                lgb_params={k: v for k, v in htets_params.items() if k not in ["num_boost_round", "season_length", "ets_type", "manual_param", "scaling", "train"]},
                num_iterations=htets_params["num_boost_round"],
                train_data=train[["series_id", "date", "value"] + features],
                seed=123,
            )

            # Forecast
            ht_ets_fcst = ht_ets.forecast(
                test_data=test[["series_id", "date"] + features]
            )

        else: # Generate forecasts using the manually provided parameter
            n_obs = train.shape[0] + test.shape[0]
            n_params = ht_ets.n_params
            n_series = train["series_id"].nunique()
            ht_ets.n_series = n_series
            params = torch.ones(
                (n_obs, ht_ets.n_params), dtype=ht_ets.dtype
            ).reshape(n_series, -1, n_params) * manual_param

            fit_params = params[:, :-fcst_h, :]
            fcst_params = params[:, -fcst_h:, :]

            # Create mask for training data
            train_mask = torch.tensor(
                np.ones_like(train["value"]),
                dtype=ht_ets.dtype
            ).reshape(n_series, -1)

            # Forward pass to update states using training data
            target = torch.tensor(
                train["value"].to_numpy().reshape(n_series, -1),
                dtype=ht_ets.dtype
            )
            ht_ets.is_trained = True
            dfit = lgb.Dataset(
                data=train[features],
                label=train["value"].to_numpy().reshape(-1, ),
                free_raw_data=False,
            )

            level, trend, seasonality, fit = ht_ets.forward(fit_params, dfit, target, train_mask)

            # Extract last states (forward returns final scalars directly)
            last_level = level
            last_trend = trend

            # Generate forecasts using roll-forward state-space recursion,
            # matching HyperTreeETS.forecast().
            if ht_ets.ets_type == "triple":
                seasonality_idxs = torch.tensor(
                    test[ht_ets.seasonality_feature].values - 1
                ).reshape(n_series, fcst_h)
                batch_idx = torch.arange(n_series, dtype=torch.long)
                alpha_fcst = fcst_params[:, :, 0]
                beta_fcst = fcst_params[:, :, 1]
                gamma_fcst = fcst_params[:, :, 2]
                phi_fcst = fcst_params[:, :, 3]

                fcsts = []
                level_h = last_level
                trend_h = last_trend
                for h in range(fcst_h):
                    alpha = alpha_fcst[:, h]
                    beta = beta_fcst[:, h]
                    gamma = gamma_fcst[:, h]
                    phi = phi_fcst[:, h]
                    s_idx = seasonality_idxs[:, h].long()
                    s_h = seasonality[batch_idx, s_idx]

                    pseudo_y = (level_h + phi * trend_h) * s_h
                    fcsts.append(pseudo_y.reshape(-1, 1))

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

            elif ht_ets.ets_type == "trend":
                alpha_fcst = fcst_params[:, :, 0]
                beta_fcst = fcst_params[:, :, 1]

                fcsts = []
                level_h = last_level
                trend_h = last_trend
                for h in range(fcst_h):
                    alpha = alpha_fcst[:, h]
                    beta = beta_fcst[:, h]

                    pseudo_y = level_h + trend_h
                    fcsts.append(pseudo_y.reshape(-1, 1))

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

            ht_ets_fcst = pd.DataFrame.from_dict({
                    "series_id": test["series_id"].to_numpy().flatten(),
                    "date": test["date"].to_numpy().flatten(),
                    "fcst": np.hstack(fcsts).flatten(),
                    "model": "Hyper-Tree-ETS",
                })

        end = time.time()
        runtime = (end - start) / 60
        ht_ets_fcst["runtime"] = runtime

        if htets_params["scaling"]:
            ht_ets_fcst["fcst"] = ht_ets_fcst["fcst"] - scaling_factor
    else:
        ht_ets_fcst = test[["series_id", "date"]].copy()
        ht_ets_fcst["fcst"] = np.nan
        ht_ets_fcst["runtime"] = np.nan
        ht_ets_fcst["model"] = "Hyper-Tree-ETS"
        ht_ets_fcst = ht_ets_fcst[["series_id", "date", "fcst", "runtime", "model"]]

    # Arrange columns
    ht_ets_fcst = ht_ets_fcst[col_order]

    return ht_ets_fcst


def AutoARIMAForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        fcst_h: int,
        freq: str,
        season_length: int = 1,
        features: list = None,
        n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast using statsforecast's AutoARIMA.

    StatsForecast fits one model per series internally; ``n_jobs`` is passed
    through to its parallel backend.

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame
        Test data (supplies ``date`` for forecast index and, if ``features`` is
        given, the exogenous regressor values for the forecast horizon).
    fcst_h : int
        Forecast horizon.
    freq : str
        Frequency of the time series (Period- or DateOffset-style).
    season_length : int
        Seasonal length of the time series.
    features : list
        List of features (exogenous regressors for AutoARIMA-X).
    n_jobs : int
        Number of parallel workers (``-1`` = all cores).

    Returns
    -------
    pd.DataFrame
        Columns ``["series_id", "date", "fcst", "runtime", "model"]``.
    """
    start = time.time()
    ts_train, ts_test_X = to_statsforecast_frames(train, test, features)
    sf = StatsForecast(
        models=[SFAutoARIMA(season_length=season_length)],
        freq=to_offset_freq(freq),
        n_jobs=n_jobs,
    )
    if ts_test_X is None:
        raw = sf.forecast(h=fcst_h, df=ts_train)
    else:
        raw = sf.forecast(h=fcst_h, df=ts_train, X_df=ts_test_X)
    runtime = (time.time() - start) / 60

    model_label = "AutoARIMA" + ("-X" if features else "")

    return to_statsforecast_output(raw, test, model_label, runtime, col_order)


def ARForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        fcst_h: int,
        freq: str,
        p: int = 1,
        features: list = None,
        n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast using a fixed-order AR(p) model via statsforecast's
    ``ARIMA(order=(p, 0, 0))``. Supports AR(p)-X when ``features`` is given.

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame
        Test data (supplies exogenous regressors for the forecast horizon if
        ``features`` is given).
    fcst_h : int
        Forecast horizon.
    freq : str
        Frequency of the time series (Period- or DateOffset-style).
    p : int
        Order of the AR part.
    features : list
        List of features (exogenous regressors for AR(p)-X).
    n_jobs : int
        Number of parallel workers (``-1`` = all cores).

    Returns
    -------
    pd.DataFrame
        Columns ``["series_id", "date", "fcst", "runtime", "model"]``.
    """
    start = time.time()
    ts_train, ts_test_X = to_statsforecast_frames(train, test, features)
    sf = StatsForecast(
        models=[SFARIMA(order=(p, 0, 0))],
        freq=to_offset_freq(freq),
        n_jobs=n_jobs,
    )
    if ts_test_X is None:
        raw = sf.forecast(h=fcst_h, df=ts_train)
    else:
        raw = sf.forecast(h=fcst_h, df=ts_train, X_df=ts_test_X)
    runtime = (time.time() - start) / 60

    model_label = f"AR({p})" + ("-X" if features else "")

    return to_statsforecast_output(raw, test, model_label, runtime, col_order)


def AutoETSForecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        fcst_h: int,
        freq: str,
        season_length: int = 1,
        damped: bool = True,
        features: list = None,
        n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Forecast using statsforecast's AutoETS.

    StatsForecast fits one model per series internally; ``n_jobs`` is passed
    through to its parallel backend.

    ETS does not use exogenous regressors in the classical formulation.
    ``features`` is accepted for API parity with AutoARIMA-X; it is ignored
    by the underlying model and only affects the model label.

    Parameters
    ----------
    train : pd.DataFrame
        Training data.
    test : pd.DataFrame
        Test data (``date`` is used to align the forecast index).
    fcst_h : int
        Forecast horizon.
    freq : str
        Frequency of the time series (Period- or DateOffset-style).
    season_length : int
        Seasonal length of the time series.
    damped : bool
        Whether to use a damped trend.
    features : list
        List of features (kept for API parity; ignored by ETS).
    n_jobs : int
        Number of parallel workers (``-1`` = all cores).

    Returns
    -------
    pd.DataFrame
        Columns ``["series_id", "date", "fcst", "runtime", "model"]``.
    """
    start = time.time()
    ts_train, _ = to_statsforecast_frames(train, test, None)
    sf = StatsForecast(
        models=[SFAutoETS(season_length=season_length, damped=damped)],
        freq=to_offset_freq(freq),
        n_jobs=n_jobs,
    )
    raw = sf.forecast(h=fcst_h, df=ts_train)
    runtime = (time.time() - start) / 60

    model_label = "AutoETS" + ("X" if features else "")

    return to_statsforecast_output(raw, test, model_label, runtime, col_order)


def MLPARForecast(
        htnetar_network_params: dict,
        n_epochs: int,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        lags: list,
        freq: str,
        fcst_h: int,
        loss_fn: nn.Module,
        seed: int,
        device: torch.device,
) -> pd.DataFrame:
    """
    Train a pure MLP model that maps features directly to AR(p)
    parameters, without the GBDT encoder of Hyper-TreeNet-AR. Used as an
    ablation baseline to isolate the contribution of the GBDT representation
    layer.

    The ``lags`` argument must be a contiguous list starting from 1
    (e.g., [1, 2, ..., p]); non-contiguous lags are not supported because the
    recursive multi-step forecast assumes consecutive positions.

    Parameters
    ----------
    htnetar_network_params : dict
        MLP hyper-parameters (``hidden_dim``, ``dropout``, ``learning_rate``).
    n_epochs : int
        Number of training epochs for the MLP.
    train : pd.DataFrame
        Training data containing ``series_id``, ``date``, ``value`` and feature columns.
    test : pd.DataFrame
        Test data for generating forecasts.
    features : list
        Feature column names passed to the MLP.
    lags : list
        Contiguous AR lag orders starting from 1 (e.g., ``[1, 2, ..., p]``).
    freq : str
        Frequency of the time series.
    fcst_h : int
        Forecast horizon.
    loss_fn : nn.Module
        PyTorch loss function for the MLP training loop.
    seed : int
        Random seed for reproducibility.
    device : torch.device
        Device for the MLP (e.g., ``'cuda'`` or ``'cpu'``).

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns ``series_id, date, fcst, model``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure contiguous lags starting from 1 (AR(p) assumption of the forecast loop)
    p = max(lags)
    if sorted(lags) != list(range(1, p + 1)):
        raise ValueError(
            f"MLPARForecast requires contiguous lags [1, 2, ..., p]; got {sorted(lags)}."
        )

    # Create lag features
    preprocess_df = create_lag_features(train, lags=lags, features=features)
    train_lags_ar = preprocess_df.filter(regex="lag").values
    train_targets = preprocess_df["value"].to_numpy().reshape(-1, 1)
    train_features = preprocess_df[features].values

    # Initialize MLP (no random projection; input is the raw feature vector)
    mlp_model = MLP(
        tree_embed_dim=len(features),
        output_dim=p,
        hidden_dim=htnetar_network_params["hidden_dim"],
        use_random_projection=False,
        rp_embed_dim=p,
        dropout_rate=htnetar_network_params["dropout"],
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(
        mlp_model.parameters(),
        lr=htnetar_network_params["learning_rate"],
    )

    # Standardize features (MLP is scale-sensitive)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
    lags_train_tensor = torch.tensor(train_lags_ar, dtype=torch.float32, device=device)

    X_test = scaler.transform(test[features].values)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Train the MLP end-to-end against the AR forecast loss
    mlp_model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        ar_params = mlp_model(X_train)
        fcst = torch.sum(ar_params * lags_train_tensor, dim=1, dtype=torch.float32).unsqueeze(1)
        loss = loss_fn(fcst, y_train)
        loss.backward()
        optimizer.step()

    # Predict AR parameters for every test observation
    mlp_model.eval()
    with torch.no_grad():
        params_fcst = (
            mlp_model(X_test).cpu().detach().numpy()
            .reshape(-1, fcst_h, p)
        )

    # Seed the recursive forecast with the last p training values per series
    n_series_test = len(params_fcst)
    ar_lags = (
        train.groupby(["series_id"], sort=False)
        .apply(lambda x: x["value"][-p:][::-1])
        .reset_index(drop=True)
        .values
        .reshape(n_series_test, -1)
    )
    forecasts = []
    for h in range(fcst_h):
        next_val = np.sum(params_fcst[:, h, :] * ar_lags, axis=1).reshape(-1, 1)
        forecasts.append(next_val)
        ar_lags = np.concatenate([next_val, ar_lags[:, :-1]], axis=1)

    mlp_fcst = pd.DataFrame({
        "series_id": test["series_id"].to_numpy().flatten(),
        "date": test["date"].to_numpy().flatten(),
        "fcst": np.hstack(forecasts).flatten(),
        "model": f"Hyper-TreeNet-AR",
    })

    return mlp_fcst


class HyperTreeNetDirectForecasting:
    """Hyper-TreeNet variant that trains the network to output forecasts
    directly, bypassing the AR(p) parametric layer.
    """

    from typing import Tuple, Callable
    from hypertrees.utils import CustomLogger
    import warnings
    lgb.register_logger(CustomLogger())

    warnings.filterwarnings(
        "ignore",
        message="Using backward\\(\\) with create_graph=True will create a reference cycle.*"
    )

    _network_states = {}  # Store network states for each instance
    def __init__(
            self,
            freq: str = "M",
            loss_fn: Callable = nn.MSELoss(),
            device: str = "cpu",
    ):
        """
        Initialize the Hyper-TreeNet-Direct model.

        Parameters
        ----------
        freq : str
            Frequency of the time series (e.g., ``'M'`` for monthly).
        loss_fn : Callable
            Loss function for training (default is ``nn.MSELoss()``).
        device : str
            Device for computation (e.g., ``'cpu'`` or ``'cuda'``).
        """

        self.freq = freq
        self.loss_fn = loss_fn
        self.loss_name = self.loss_fn.__class__.__name__
        self.dtype = torch.float32
        self.device = device
        self.model = None

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom objective function for LightGBM training with separate gradients (Option 2).

        Defines the gradients and Hessians for the LightGBM model. Converts the
        raw LightGBM outputs to PyTorch tensors, computes the loss via the MLP
        decoder, and backpropagates to get gradients.

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
        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=self.dtype, device=self.device)
        embeds, loss = self.get_embeds_loss(predt, target, requires_grad=True)
        grad, hess = self.calculate_gradients_and_hessians(loss, embeds)

        return grad, hess

    def get_embeds_loss(
            self,
            predt: np.ndarray,
            target: torch.Tensor,
            requires_grad: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Transform LightGBM outputs into embeddings and calculate loss.

        Parameters
        ----------
        predt : np.ndarray
            Raw outputs from LightGBM, representing the GBDT embeddings.
        target : torch.Tensor
            Target values (actual time series values).
        requires_grad : bool
            Whether to compute gradients (True during training).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Parameters tensor and loss value.
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
        fcst_net = self.network(gbdt_embed)
        network_loss = self.loss_fn(fcst_net, target)
        self.optimizer.zero_grad()
        network_loss.backward()
        self.optimizer.step()

        # Store network state
        HyperTreeNetDirectForecasting._network_states = self.network.state_dict()

        # Calculate loss for GBDT
        self.network.eval()
        fcst_gbdt = self.network(gbdt_embed)
        gbm_loss = self.loss_fn(fcst_gbdt, target)

        return gbdt_embed, gbm_loss

    def calculate_gradients_and_hessians(self, loss: torch.Tensor, embeds: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients and hessians for LightGBM optimization.

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

    def train(
            self,
            lgb_params: dict = None,
            network_params: dict = None,
            num_iterations: int = 100,
            train_data: pd.DataFrame = None,
            features: list = None,
            seed: int = 123,
            verbose: int = -1,
    ):
        """
        Train the HyperTreeNetDirectForecasting model on time series data.

        Parameters
        ----------
        lgb_params : dict
            LightGBM parameters.
        network_params : dict
            Network parameters. Available parameters are:
                - "learning_rate": Learning rate for the neural network optimizer.
                - "hidden_dim": Dimension of the hidden layer in the MLP.
                - "embedding_dimension": Dimension of the tree embeddings from LightGBM.
                - "use_random_projection": Whether to use random projection for embeddings.
                - "rp_embed_dim": Dimension of the random projection embeddings (if used).
                - "dropout": Dropout rate for regularization.
        num_iterations : int
            Number of boosting rounds for training.
        train_data : pd.DataFrame
            Training data containing series_id, date, value and feature columns.
        features : list
            List of feature column names to use for training.
        seed : int
            Random seed for reproducibility.
        verbose : int
            Verbosity level for LightGBM training.
        """
        # Set the network and optimizer
        gbdt_params = lgb_params.copy()
        self.embedding_dim = network_params["embedding_dimension"]
        self.features = features
        from hypertrees.models.mlp import MLP

        self.network = MLP(
            tree_embed_dim=self.embedding_dim,
            output_dim=1,
            hidden_dim=network_params["hidden_dim"],
            use_random_projection=network_params["use_random_projection"],
            rp_embed_dim=network_params["rp_embed_dim"] if network_params["use_random_projection"] else None,
            dropout_rate=network_params["dropout"],
            seed=seed
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=network_params["learning_rate"])

        # GBDT parameters
        self.lgb_params = {
            "num_class": self.embedding_dim,
            "objective": self.objective_fn,
            "metric": "None",
            "random_seed": seed,
            "verbose": verbose
        }

        # Update with user-provided LightGBM parameters
        self.lgb_params.update(gbdt_params)

        # Create LightGBM dataset
        dtrain = lgb.Dataset(
            data=train_data[self.features].values,
            label=train_data["value"].to_numpy().reshape(-1),
        )

        # Train model
        self.model = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=num_iterations,
        )

        # Set trained flag to True
        self.is_trained = True


    def forecast(
            self,
            test_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate forecasts using the trained model.

        Parameters
        ----------
        test_data : pd.DataFrame
            Test data for which to generate forecasts. Must contain the same
            feature columns used during training.

        Returns
        -------
        pd.DataFrame
            Forecasted data with columns:
            - series_id: Identifier for each time series
            - date: Forecast date/time
            - fcst: Forecasted value
            - model: Model name identifier
        """

        # Get tree embeddings
        gbdt_embeds = torch.tensor(
            self.model.predict(test_data[self.features].values),
            dtype=self.dtype,
            device=self.device
        ).reshape(-1, self.embedding_dim)

        # Load saved network state
        self.network.load_state_dict(HyperTreeNetDirectForecasting._network_states)

        self.network.eval()
        with torch.no_grad():
            forecasts = (self.network(gbdt_embeds)
                         .cpu()
                         .detach()
                         .numpy()
                         .reshape(-1, 1))

        out_df = pd.DataFrame({
            "series_id": test_data["series_id"].to_numpy().flatten(),
            "date": test_data["date"].to_numpy().flatten(),
            "fcst": np.hstack(forecasts).flatten(),
            "model": "Hyper-TreeNet-AR",
        })

        return out_df


def TwoStepForecast(
        lgb_params: dict,
        htnetar_network_params: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: list,
        lags: list,
        freq: str,
        fcst_h: int,
        loss_fn: nn.Module,
        seed: int,
        device: torch.device,
) -> pd.DataFrame:
    """
    Two-step ablation of Hyper-TreeNet-AR: sequentially-trained GBDT then
    MLP (no joint gradient flow).

    Step 1 trains a standalone LightGBM on the features and extracts
    standardized leaf-index embeddings. Step 2 trains an MLP that maps
    those fixed embeddings to AR(``max(lags)``) coefficients, optimising
    the one-step-ahead AR forecast loss. Multi-step forecasts are then
    produced recursively.

    Parameters
    ----------
    lgb_params : dict
        LightGBM hyper-parameters. The first three entries (insertion order:
        ``num_boost_round``, ``eta``, ``linear_tree``) are the standalone
        LightGBM training params; ``num_boost_round`` also doubles as MLP
        epochs in step 2.
    htnetar_network_params : dict
        MLP hyper-parameters (``hidden_dim``, ``dropout``, ``learning_rate``).
    train : pd.DataFrame
        Training data with ``series_id``, ``date``, ``value`` and the
        ``features`` columns.
    test : pd.DataFrame
        Test data with ``series_id``, ``date`` and the ``features`` columns.
    features : list
        Feature column names passed to LightGBM.
    lags : list
        AR lag orders; ``max(lags)`` is the AR order of the MLP decoder.
    freq : str
        Nixtla-compatible frequency string.
    fcst_h : int
        Forecast horizon.
    loss_fn : nn.Module
        PyTorch loss for the MLP training loop.
    seed : int
        Random seed.
    device : torch.device
        Device for the MLP.

    Returns
    -------
    pd.DataFrame
        Columns ``series_id, date, fcst, model``.
    """
    from itertools import islice
    from sklearn.preprocessing import StandardScaler
    from hypertrees.models.mlp import MLP

    # ------------------------------------------------------------------------
    # Step 1: Train LightGBM and extract standardized leaf-index embeddings
    # ------------------------------------------------------------------------

    # Build the LightGBM dataset from the raw features and values.
    dtrain = lgb.Dataset(
        data=train[features],
        label=train["value"].to_numpy().reshape(-1, ),
    )

    # Only the first three entries of lgb_params (num_boost_round, eta,
    # linear_tree) are the standalone-LightGBM parameters; the rest belong
    # to the MLP / network configuration.
    gbdt_params = dict(islice(lgb_params.items(), 3))
    gbdt_model = lgb.train(gbdt_params, dtrain)

    # Fit the scaler on the training leaf-index embeddings and transform them.
    scaler = StandardScaler()
    train_embeddings = gbdt_model.predict(train[features], pred_leaf=True)
    train_embeddings = scaler.fit_transform(train_embeddings)
    train_embeddings = pd.DataFrame(
        train_embeddings,
        columns=[f"embed_{i}" for i in range(train_embeddings.shape[1])],
    )
    embed_feats = train_embeddings.filter(regex="embed_").columns.tolist()
    train_embeddings["series_id"] = train["series_id"].values
    train_embeddings["date"] = train["date"].values

    # Apply the same scaler to the test embeddings (no re-fit).
    test_embeddings = gbdt_model.predict(test[features], pred_leaf=True)
    test_embeddings = scaler.transform(test_embeddings)

    # ------------------------------------------------------------------------
    # Step 2: Build the MLP training dataset (embeddings + lag features)
    # ------------------------------------------------------------------------

    # Merge the standardized embeddings back onto the training DataFrame and
    # use them as the MLP's input features.
    train_mlp = train[["series_id", "date", "value"]].copy()
    train_mlp = pd.merge(
        train_mlp, train_embeddings, on=["series_id", "date"], how="inner"
    )
    feats_mlp = embed_feats

    # Create lag features for the AR equation
    preprocess_df = create_lag_features(train_mlp, lags=lags, features=feats_mlp)
    train_lags = preprocess_df.filter(regex="lag").values
    train_targets = preprocess_df["value"].to_numpy().reshape(-1, 1)
    train_features = preprocess_df[feats_mlp].values

    # ------------------------------------------------------------------------
    # Step 2: Initialize and train the MLP decoder
    # ------------------------------------------------------------------------

    # The MLP consumes the fixed tree embeddings (random projection is off)
    # and outputs max(lags) AR coefficients per sample.
    mlp_model = MLP(
        tree_embed_dim=len(embed_feats),
        output_dim=max(lags),
        hidden_dim=htnetar_network_params["hidden_dim"],
        use_random_projection=False,
        rp_embed_dim=max(lags),
        dropout_rate=htnetar_network_params["dropout"],
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(
        mlp_model.parameters(),
        lr=htnetar_network_params["learning_rate"],
    )

    # Move the training tensors to the target device.
    X_train = torch.tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.float32, device=device)
    lags_train = torch.tensor(train_lags, dtype=torch.float32, device=device)

    # Train the MLP end-to-end against the AR forecast loss.
    # Number of epochs is reused from lgb_params["num_boost_round"].
    mlp_model.train()
    n_epochs = lgb_params["num_boost_round"]
    for _ in range(n_epochs):
        optimizer.zero_grad()
        ar_params = mlp_model(X_train)
        fcst = torch.sum(
            ar_params * lags_train, dim=1, dtype=torch.float32
        ).unsqueeze(1)
        loss = loss_fn(fcst, y_train)
        loss.backward()
        optimizer.step()

    # ------------------------------------------------------------------------
    # Step 3: Forecast AR coefficients and recursively roll forward
    # ------------------------------------------------------------------------

    # Forecast AR coefficients for every test observation.
    mlp_model.eval()
    X_test = torch.tensor(test_embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        params_fcst = (
            mlp_model(X_test)
            .cpu()
            .detach()
            .numpy()
            .reshape(-1, fcst_h, max(lags))
        )

    # Seed the recursive forecast with the last max(lags) training values
    # per series (reversed so index 0 is the most recent lag).
    n_series = len(params_fcst)
    lags_train = (
        train.groupby(["series_id"], sort=False)
        .apply(lambda x: x["value"][-max(lags):][::-1])
        .reset_index(drop=True)
        .values
        .reshape(n_series, -1)
    )

    # Roll forward through the forecast horizon, replacing the oldest lag
    # with the newly produced forecast at each step.
    forecasts = []
    for h in range(fcst_h):
        next_val = np.sum(params_fcst[:, h, :] * lags_train, axis=1).reshape(-1, 1)
        forecasts.append(next_val)
        lags_train = np.concatenate([next_val, lags_train[:, :-1]], axis=1)

    # ------------------------------------------------------------------------
    # Assemble the output DataFrame
    # ------------------------------------------------------------------------
    two_step_fcsts = pd.DataFrame({
        "series_id": test["series_id"].to_numpy().flatten(),
        "date": test["date"].to_numpy().flatten(),
        "fcst": np.hstack(forecasts).flatten(),
        "model": f"Hyper-TreeNet-AR",
    })

    return two_step_fcsts
