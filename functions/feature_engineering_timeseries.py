import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# SETUP !!!
def time_series_setup(df, datetime_col, uniform_sampling):
    """
    Prepares a dataframe for time series feature engineering and machine learning.
    Parameters:
        df (dataframe): Input dataframe.
        datetime_col (str): Name of the datetime column.
        uniform_sampling (str): Frequency string for resampling.
    Returns:
        pd.DataFrame: Time series ready dataframe with datetime index, correct dtype and uniform spacing.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    df = df.resample(uniform_sampling).mean()
    return df

# DECOMPOSITION !!!
def decomposition_featuring(df, target_col, period, lags, trend_smooth):
    """
    Decompose time series manually for high frequency noisy data and return trend, seasonal, and residual features for ML.
    Parameters:
        df (dataframe): Time series dataframe with datetime index (from time_series_setup())
        target_col (str): Target variable
        period (int): Seasonal period length
        lags (int): ACF/PACF lags
        trend_smooth (int): Rolling window to smooth trend
    """
    y = df[target_col].copy()
    y_filled = y.ffill().bfill()

    # Rolling trend
    trend = y_filled.rolling(trend_smooth, center=True, min_periods=1).mean()
    # Detrended signal
    detrended = y_filled - trend
    # Seasonality from detrended signal
    seasonal_avg = detrended.groupby(detrended.index.hour).mean()
    seasonal_avg -= seasonal_avg.mean()
    seasonal = seasonal_avg[detrended.index.hour].values
    # Rest
    resid = detrended - seasonal

    # ACF PACF
    plot_acf(y_filled, lags=lags)
    plt.title("ACF")
    plt.show()
    plot_pacf(y_filled, lags=lags)
    plt.title("PACF")
    plt.show()
    # Original with trend
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, y_filled, label="Original")
    plt.plot(df.index, trend, label="Trend")
    plt.legend()
    plt.title("Original series with trend")
    plt.show()
    # Period
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, seasonal)
    plt.title("Seasonality")
    plt.show()
    # Extracted period
    one_period = seasonal[:period]
    plt.figure(figsize=(15, 5))
    plt.plot(one_period)
    plt.title("Period")
    plt.show()
    # Residuals
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, resid)
    plt.title("Residuals")
    plt.show()
    # Features
    decomp_features = pd.DataFrame({
        'trend': trend,
        'seasonal': seasonal,
        'resid': resid
    }, index=df.index)
    return decomp_features