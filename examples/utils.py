import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_metrics(
    df: pd.DataFrame,
    value_col: str = "value",
    fcst_col: str = "fcst",
) -> pd.Series:
    r"""Compute point-forecast accuracy metrics from a DataFrame.

    Returns MAE, MAPE, sMAPE, WAPE, and RMSE. All percentage metrics
    are reported in percent (i.e. multiplied by 100).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain numeric columns named by ``value_col`` and ``fcst_col``,
        of equal length and free of NaN. Inputs are cast to ``float64`` to
        avoid integer overflow in the squared-error term of RMSE.
    value_col : str, default "value"
        Column holding the realized (ground-truth) values.
    fcst_col : str, default "fcst"
        Column holding the point forecasts.

    Returns
    -------
    pd.Series
        Indexed by ``["MAE", "MAPE", "sMAPE", "WAPE", "RMSE"]``. Two
        diagnostics are attached via ``Series.attrs``:

        * ``"mape_excluded_count"`` : number of rows dropped from MAPE
          because ``value == 0``.
        * ``"mape_excluded_frac"``  : the same as a fraction of the input.

        ``MAPE`` is ``NaN`` if every row has ``value == 0``. ``WAPE`` is
        ``NaN`` if ``sum(|value|) == 0``.

    Raises
    ------
    KeyError
        If either column is missing from ``df``.
    ValueError
        If ``df`` is empty or contains NaN in either column.

    Notes
    -----
    Let :math:`y` denote truth and :math:`\hat y` the forecast.

    .. math::

        \mathrm{MAE}   &= \tfrac{1}{n}\sum_i |y_i - \hat y_i| \\
        \mathrm{RMSE}  &= \sqrt{\tfrac{1}{n}\sum_i (y_i - \hat y_i)^2} \\
        \mathrm{MAPE}  &= \tfrac{100}{|S|}\sum_{i \in S}
                            \tfrac{|y_i - \hat y_i|}{|y_i|},
                            \quad S = \{i : y_i \neq 0\} \\
        \mathrm{sMAPE} &= \tfrac{100}{n}\sum_i
                            \tfrac{2\,|y_i - \hat y_i|}{|y_i| + |\hat y_i|}
                            \quad (\text{0/0} \to 0) \\
        \mathrm{WAPE}  &= 100 \cdot
                            \tfrac{\sum_i |y_i - \hat y_i|}{\sum_i |y_i|}

    sMAPE follows the M3/M4 convention with range :math:`[0, 200]`. MAPE is
    undefined at :math:`y = 0` and is therefore computed on the nonzero
    subset; the excluded count is reported via ``attrs`` rather than silently
    absorbed.

    Examples
    --------
    >>> df = pd.DataFrame({"value": [10.0, 20.0, 0.0], "fcst": [12.0, 18.0, 1.0]})
    >>> m = calculate_metrics(df)
    >>> round(m["MAE"], 4)
    1.6667
    >>> m.attrs["mape_excluded_count"]
    1
    """
    missing = {value_col, fcst_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}.")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    true = df[value_col].to_numpy(dtype=np.float64)
    fcst = df[fcst_col].to_numpy(dtype=np.float64)

    if np.isnan(true).any() or np.isnan(fcst).any():
        raise ValueError(
            f"NaN values found in '{value_col}' or '{fcst_col}'. "
            "Drop or impute before evaluating."
        )

    err = true - fcst
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # WAPE: sum|err| / sum|true|; NaN if denominator is exactly zero.
    sum_abs_true = np.sum(np.abs(true) + 1e-06)  # add small constant to avoid overflow in sum_abs_true
    wape = (
        float(np.sum(abs_err) / sum_abs_true * 100)
        if sum_abs_true > 0
        else np.nan
    )

    # MAPE: undefined at y == 0, computed only on the nonzero subset.
    nonzero = true != 0
    n_excluded = int((~nonzero).sum())
    if nonzero.any():
        mape = float(np.mean(abs_err[nonzero] / np.abs(true[nonzero])) * 100)
    else:
        mape = np.nan

    # sMAPE (M3/M4 form, range [0, 200]); 0/0 contributes 0.
    denom = np.abs(true) + np.abs(fcst)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_terms = np.where(denom == 0, 0.0, 200.0 * abs_err / denom)
    smape = float(np.mean(smape_terms))

    out = pd.Series(
        {"MAE": mae, "MAPE": mape, "sMAPE": smape, "WAPE": wape, "RMSE": rmse}
    )
    out.attrs["mape_excluded_count"] = n_excluded
    out.attrs["mape_excluded_frac"] = n_excluded / len(true)

    return out

def load_air_passengers() -> pd.DataFrame:
    """Load the Air Passengers dataset in Hyper-Trees format.

    Returns a DataFrame with columns: series_id, date, value, month.
    The ``month`` column is included as a seasonal feature for the
    quickstart example.

    Returns
    -------
    pd.DataFrame
        Monthly airline passenger counts (1949–1960).
    """
    df = pd.read_csv(
        "https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv",
        parse_dates=["ds"],
    ).rename(columns={"unique_id": "series_id", "ds": "date", "y": "value"})

    return df

def plot_example_forecast(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    levels: list = None,
) -> None:
    """Plot actuals vs. Hyper-Tree-AR forecast, shading any conformal intervals.

    If ``forecasts`` contains conformal interval columns named
    ``<model>-lo-<level>`` / ``<model>-hi-<level>`` (produced by
    ``forecast(..., level=[...])``), each band is shaded. Pass ``levels`` to
    restrict which are drawn; by default all detected levels are shown.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install hypertrees[plot]"
        ) from e

    plt.figure(figsize=(12, 5))
    plt.plot(actuals["date"], actuals["value"], label="Actual",
             color="#2E86AB", linestyle="-", linewidth=2, alpha=0.8)
    plt.plot(forecasts["date"], forecasts["fcst"], label="Hyper-Tree-AR Forecast",
             color="green", linestyle="--", linewidth=2, alpha=0.8)

    # Detect and shade conformal interval bands, if present.
    model = forecasts["model"].iloc[0] if "model" in forecasts.columns else None
    if model is not None:
        detected = sorted(
            int(m.group(1))
            for c in forecasts.columns
            for m in [re.fullmatch(rf"{re.escape(model)}-lo-(\d+)", c)]
            if m
        )
        if levels is not None:
            detected = [lv for lv in detected if lv in set(levels)]
        # Shade widest band first so narrower bands sit on top.
        alphas = [0.15, 0.28, 0.40]
        shade = {lv: a for lv, a in zip(sorted(detected, reverse=True), alphas)}
        for lv in sorted(detected, reverse=True):
            plt.fill_between(
                forecasts["date"],
                forecasts[f"{model}-lo-{lv}"],
                forecasts[f"{model}-hi-{lv}"],
                color="green", alpha=shade.get(lv, 0.2), label=f"{lv}% interval",
            )

    plt.axvline(x=forecasts["date"].min(), color="black", linestyle=":", alpha=0.7, label="Train/Test Split")
    plt.title("Forecasting Results - Air Passengers Dataset", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Passengers", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def simulate_var_panel(
    k: int = 10,
    n_train: int = 240,
    fcst_h: int = 12,
    seed: int = 42,
) -> tuple:
    """Simulate an aligned panel from a stable VAR(1) with a lead/lag chain.

    Each series follows its own past and its neighbor's previous month
    (``A[i, i-1] = 0.3``), so the panel carries genuine cross-series
    structure that a vector autoregression can exploit. A common monthly
    seasonal profile and strongly heterogeneous per-series scales are
    applied on top. Columns: ``series_id``, ``date``, ``value`` plus the
    features ``month``, ``quarter``, and ``series_num`` (pandas ``category``
    dtype, so LightGBM applies true categorical splits).

    Parameters
    ----------
    k : int, default 10
        Number of series.
    n_train : int, default 240
        Training observations per series.
    fcst_h : int, default 12
        Test observations per series (the forecast horizon).
    seed : int, default 42
        Seed for the random number generator.

    Returns
    -------
    tuple
        ``(df, train, test)`` where ``df`` is the full panel and
        ``train`` / ``test`` are the per-series head/tail splits.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2000-01-01", periods=n_train + fcst_h, freq="MS")

    # Stable VAR(1) with a lead/lag chain
    A = 0.5 * np.eye(k)
    for i in range(1, k):
        A[i, i - 1] = 0.3

    const = 10.0 * (np.eye(k) - A).sum(axis=1)
    Y = np.full((len(dates), k), 10.0)
    for t in range(1, len(dates)):
        Y[t] = const + A @ Y[t - 1] + 0.5 * rng.randn(k)

    # Common monthly seasonality and heterogeneous series scales
    season = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(1, 13) / 12)
    scales = rng.uniform(20, 2000, size=k)
    Y *= season[dates.month - 1][:, None] * scales[None, :] / 10.0

    df = pd.concat(
        [
            pd.DataFrame({
                "series_id": f"Series {i + 1}",
                "date": dates,
                "value": Y[:, i],
                "month": dates.month,
                "quarter": dates.quarter,
                "series_num": i,
            })
            for i in range(k)
        ],
        ignore_index=True,
    )
    df["series_num"] = df["series_num"].astype("category")

    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return df, train, test


def simulate_intermittent_panel(
    k: int = 20,
    n_train: int = 156,
    fcst_h: int = 12,
    seed: int = 42,
) -> tuple:
    """Simulate an aligned panel of intermittent (zero-inflated) demand series.

    Each SKU's weekly demand is the product of two feature-driven components,
    which is exactly the structure the TSB method targets:

    * a demand *probability* (occurrence) that rises during promotions and
      follows a mild monthly seasonal cycle, and
    * a demand *size* (units when demand occurs) that is larger during
      promotions.

    Both components depend on a binary ``promo`` feature and on ``month``, so a
    Hyper-Tree-TSB whose smoothing rates are functions of features has real
    structure to exploit. Columns: ``series_id``, ``date``, ``value`` plus the
    features ``month``, ``promo``, and ``series_num`` (pandas ``category``
    dtype, so LightGBM applies true categorical splits). All series share the
    same length, as TSB requires.

    Parameters
    ----------
    k : int, default 20
        Number of SKUs (series).
    n_train : int, default 156
        Training observations per series (weeks).
    fcst_h : int, default 12
        Test observations per series (the forecast horizon).
    seed : int, default 42
        Seed for the random number generator.

    Returns
    -------
    tuple
        ``(df, train, test)`` where ``df`` is the full panel and
        ``train`` / ``test`` are the per-series head/tail splits.
    """
    rng = np.random.RandomState(seed)
    T = n_train + fcst_h
    dates = pd.date_range("2015-01-05", periods=T, freq="W-MON")
    month = dates.month.to_numpy()

    # Per-SKU baselines for occurrence probability (logit) and size (log),
    # plus a shared monthly seasonal effect on the occurrence probability.
    base_logit = rng.uniform(-2.2, -0.4, size=k)   # mostly low occurrence
    base_logsize = rng.uniform(0.5, 2.0, size=k)   # heterogeneous sizes
    season_logit = 0.5 * np.sin(2 * np.pi * month / 12)

    frames = []
    for i in range(k):
        # Each SKU has its own promotion calendar: short recurring bursts.
        promo = np.zeros(T, dtype=int)
        start = rng.randint(2, 8)
        while start < T:
            promo[start:start + rng.randint(1, 3)] = 1
            start += rng.randint(6, 14)

        # Probability of demand and mean demand size, both functions of features.
        p = 1.0 / (1.0 + np.exp(-(base_logit[i] + season_logit + 1.3 * promo)))
        mu = np.exp(base_logsize[i] + 0.6 * promo)

        occurrence = (rng.uniform(size=T) < p).astype(float)
        size = rng.poisson(mu) + 1.0          # at least one unit when demand occurs
        value = occurrence * size

        frames.append(pd.DataFrame({
            "series_id": f"SKU {i + 1}",
            "date": dates,
            "value": value.astype(float),
            "month": month,
            "promo": promo,
            "series_num": i,
        }))

    df = pd.concat(frames, ignore_index=True)
    df["series_num"] = df["series_num"].astype("category")

    train = df.groupby("series_id", sort=False).head(n_train).reset_index(drop=True)
    test = df.groupby("series_id", sort=False).tail(fcst_h).reset_index(drop=True)

    return df, train, test


def plot_forecasts(
    datasets: list,
    split_date=None,
    title: str = "Forecasting Results",
    xlabel: str = "Date",
    ylabel: str = "Value",
    level: int = None,
) -> None:
    """Plot actuals and model forecasts on a single axis.

    Parameters
    ----------
    datasets : list of tuple
        Entries ``(data, x_col, y_col, label, color, style)``, one per line.
    split_date : optional
        If given, a dotted vertical line marks the train/test split.
    title, xlabel, ylabel : str
        Axis annotations.
    level : int, optional
        If given, every forecast DataFrame carrying
        ``<model>-lo-<level>`` / ``<model>-hi-<level>`` columns gets its
        conformal interval shaded in the line's color.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install hypertrees[plot]"
        ) from e

    plt.figure(figsize=(12, 5))
    for data, x_col, y_col, label, color, style in datasets:
        plt.plot(data[x_col], data[y_col], label=label, color=color,
                 linestyle=style, linewidth=2, alpha=0.8)
        if level is not None and "model" in data.columns:
            model = data["model"].iloc[0]
            lo, hi = f"{model}-lo-{level}", f"{model}-hi-{level}"
            if lo in data.columns and hi in data.columns:
                plt.fill_between(data[x_col], data[lo], data[hi], color=color,
                                 alpha=0.15, label=f"{level}% Interval ({model})")
    if split_date is not None:
        plt.axvline(x=split_date, color="black", linestyle=":", alpha=0.7, label="Train/Test Split")

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_panel_forecasts(
    datasets: list,
    series_ids: list,
    split_date=None,
    interval_fcst: pd.DataFrame = None,
    level: int = 80,
    history: int = 100,
    title: str = "Forecasting Results - Simulated VAR Panel",
) -> None:
    """Plot actuals and model forecasts for several panel series side by side.

    One subplot per entry of ``series_ids`` with a single global legend
    below the panels. Each DataFrame in ``datasets`` must carry a
    ``series_id`` column.

    Parameters
    ----------
    datasets : list of tuple
        Entries ``(data, x_col, y_col, label, color, style)``, one per line.
        Actual-value entries (``y_col == "value"``) are truncated to the
        last ``history`` rows per series.
    series_ids : list
        Series to plot, one subplot each.
    split_date : optional
        If given, a dotted vertical line marks the train/test split.
    interval_fcst : pd.DataFrame, optional
        Forecast output with ``<model>-lo-<level>`` / ``<model>-hi-<level>``
        columns; the band is shaded in each subplot.
    level : int, default 80
        Confidence level of the shaded interval.
    history : int, default 72
        Number of trailing actual observations to show per series.
    title : str
        Figure title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install hypertrees[plot]"
        ) from e

    fig, axes = plt.subplots(1, len(series_ids), figsize=(5 * len(series_ids), 5), sharex=True)
    axes = np.atleast_1d(axes)
    model = interval_fcst["model"].iloc[0] if interval_fcst is not None else None

    for ax, sid in zip(axes, series_ids):
        for data, x_col, y_col, label, color, style in datasets:
            series = data[data["series_id"] == sid]
            if y_col == "value":
                series = series.tail(history)
            ax.plot(series[x_col], series[y_col], label=label, color=color,
                    linestyle=style, linewidth=2, alpha=0.8)
        if interval_fcst is not None:
            band = interval_fcst[interval_fcst["series_id"] == sid]
            ax.fill_between(band["date"], band[f"{model}-lo-{level}"], band[f"{model}-hi-{level}"],
                            color="green", alpha=0.15, label=f"{level}% Interval ({model})")
        if split_date is not None:
            ax.axvline(x=split_date, color="black", linestyle=":", alpha=0.7, label="Train/Test Split")
        ax.set_title(sid, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(title, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout(rect=(0, 0.12, 1, 0.96))
    plt.show()


def plot_model_intervals(
    actuals: pd.DataFrame,
    forecasts: dict,
    levels: list = None,
    title: str = None,
) -> None:
    """Plot actuals, forecast, and conformal interval bands, one subplot per model.

    Interval columns named ``<model>-lo-<level>`` / ``<model>-hi-<level>``
    (produced by ``forecast(..., level=[...])``) are detected automatically
    and shaded, widest level first. A single global legend sits below the
    panels.

    Parameters
    ----------
    actuals : pd.DataFrame
        Realized values with ``date`` and ``value`` columns.
    forecasts : dict
        Mapping of subplot title to a forecast DataFrame containing
        ``date``, ``fcst``, and a ``model`` column.
    levels : list of int, optional
        Restrict which detected levels are shaded; by default all are shown.
    title : str, optional
        Figure title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install hypertrees[plot]"
        ) from e

    fig, axes = plt.subplots(1, len(forecasts), figsize=(6 * len(forecasts), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (name, fcst_df) in zip(axes, forecasts.items()):
        model = fcst_df["model"].iloc[0]
        ax.plot(actuals["date"], actuals["value"], label="Actual",
                color="#2E86AB", linewidth=2, alpha=0.8)
        ax.plot(fcst_df["date"], fcst_df["fcst"], label="Forecast",
                color="green", linestyle="--", linewidth=2)

        detected = sorted(
            int(m.group(1))
            for c in fcst_df.columns
            for m in [re.fullmatch(rf"{re.escape(model)}-lo-(\d+)", c)]
            if m
        )
        if levels is not None:
            detected = [lv for lv in detected if lv in set(levels)]
        # Shade widest band first so narrower bands sit on top.
        alphas = [0.15, 0.28, 0.40]
        shade = {lv: a for lv, a in zip(sorted(detected, reverse=True), alphas)}
        for lv in sorted(detected, reverse=True):
            ax.fill_between(
                fcst_df["date"],
                fcst_df[f"{model}-lo-{lv}"],
                fcst_df[f"{model}-hi-{lv}"],
                color="green", alpha=shade.get(lv, 0.2), label=f"{lv}% Interval",
            )

        ax.axvline(x=fcst_df["date"].min(), color="black", linestyle=":",
                   alpha=0.7, label="Train/Test Split")
        ax.set_title(name, fontsize=14)
        ax.grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=11)
    plt.tight_layout(rect=(0, 0.08, 1, 0.96 if title is not None else 1.0))
    plt.show()


def coverage(
    df: pd.DataFrame,
    level: int,
    model: str,
    value_col: str = "value",
) -> float:
    r"""Empirical coverage of a conformal prediction interval.

    Computes the fraction of realized values that fall within the
    ``[<model>-lo-<level>, <model>-hi-<level>]`` band, reported in percent.
    For a well-calibrated ``level``% interval this should be close to ``level``.

    Parameters
    ----------
    df : pd.DataFrame
        Forecast output containing ``value_col`` plus the interval columns
        ``f"{model}-lo-{level}"`` and ``f"{model}-hi-{level}"``.
    level : int
        Nominal confidence level (e.g. ``90``).
    model : str
        Model-name prefix used for the interval columns (the ``model`` column
        value, e.g. ``"Hyper-Tree-AR(12)"``).
    value_col : str, default "value"
        Column holding the realized values.

    Returns
    -------
    float
        Empirical coverage in percent, in ``[0, 100]``.
    """
    lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
    missing = {value_col, lo_col, hi_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}.")

    true = df[value_col].to_numpy(dtype=np.float64)
    lo = df[lo_col].to_numpy(dtype=np.float64)
    hi = df[hi_col].to_numpy(dtype=np.float64)
    inside = (true >= lo) & (true <= hi)
    return float(np.mean(inside) * 100)


def mean_interval_width(
    df: pd.DataFrame,
    level: int,
    model: str,
) -> float:
    """Mean width of a conformal prediction interval.

    Parameters
    ----------
    df : pd.DataFrame
        Forecast output containing ``f"{model}-lo-{level}"`` and
        ``f"{model}-hi-{level}"``.
    level : int
        Nominal confidence level (e.g. ``90``).
    model : str
        Model-name prefix used for the interval columns.

    Returns
    -------
    float
        Mean of ``hi - lo`` across all rows.
    """
    lo_col, hi_col = f"{model}-lo-{level}", f"{model}-hi-{level}"
    missing = {lo_col, hi_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}.")

    lo = df[lo_col].to_numpy(dtype=np.float64)
    hi = df[hi_col].to_numpy(dtype=np.float64)

    return float(np.mean(hi - lo))


def plot_stl(
    df,
    date_col="date",
    cols=("trend", "seasonality"),
    group_col="model",
    base_size=14,
    figsize=(10, 5),
):
    """Stacked plot of the trend and seasonality columns, one panel each.

    Expects a long DataFrame with a date column, one numeric column per
    component, and an optional grouping column (``model``) that gets one line
    per level. Complexity is O(n) per column, trivial next to the upstream
    decomposition.

    Assumes a single ``series_id`` per call. If several series are stacked in
    the frame, filter to one first or the lines will connect unrelated points.

    Returns
    -------
    (fig, axes)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    has_groups = group_col in df.columns and df[group_col].nunique() > 1

    fig, axes = plt.subplots(
        nrows=len(cols), ncols=1, sharex=True,
        figsize=figsize, constrained_layout=True,
    )
    axes = [axes] if len(cols) == 1 else list(axes)

    for ax, col in zip(axes, cols):
        if has_groups:
            for key, g in df.groupby(group_col):
                g = g.sort_values(date_col)
                ax.plot(g[date_col], g[col], linewidth=1.4, label=str(key))
        else:
            g = df.sort_values(date_col)
            ax.plot(g[date_col], g[col], linewidth=1.4, color="#1f77b4")

        ax.set_ylabel(col.capitalize(), fontsize=base_size)
        ax.grid(True, color="grey", alpha=0.12, linewidth=0.6)
        ax.tick_params(labelsize=base_size * 0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    if has_groups:
        axes[0].legend(frameon=True, fontsize=base_size * 0.7)

    x = df[date_col]
    span_years = max(1.0, (x.max() - x.min()).days / 365.0)
    step = max(1, round(span_years / 8))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=step))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Hyper-Tree-STL Decomposition", fontsize=base_size * 1.1)
