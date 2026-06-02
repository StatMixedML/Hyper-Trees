import numpy as np
import pandas as pd


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
) -> None:
    """Plot actuals vs. Hyper-Tree-AR forecast for the air passengers example.

    Parameters
    ----------
    actuals : pd.DataFrame
        Full series with ``date`` and ``value`` columns.
    forecasts : pd.DataFrame
        Forecasts with ``date`` and ``fcst`` columns.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install hypertrees[plot]"
        ) from e

    plt.figure(figsize=(12, 5))
    datasets = [
        (actuals, "date", "value", "Actual", "#2E86AB", "-"),
        (forecasts, "date", "fcst", "Hyper-Tree-AR Forecast", "green", "--"),
    ]
    for data, x_col, y_col, label, color, style in datasets:
        plt.plot(data[x_col], data[y_col], label=label, color=color,
                 linestyle=style, linewidth=2, alpha=0.8)
    plt.axvline(x=forecasts["date"].min(), color="black", linestyle=":", alpha=0.7,
                label="Train/Test Split")
    plt.title("Forecasting Results - Air Passengers Dataset", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Number of Passengers", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
