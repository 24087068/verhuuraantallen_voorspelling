import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, filtfilt

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

# LAGS !!!
def lag_featuring(df, target_col=None, lags=None):
    """
    Create lag features for a time series.
    Parameters:
        df (dataframe): Time series dataframe with datetime index
        target_col (str, optional): Target column to create lags for. If None, function does nothing.
        lags (list, optional): list of integer lags (in periods)
    Returns:
        pd.DataFrame: Lagged features
    """
    if target_col is None or target_col not in df.columns:
        # Return empty DataFrame if target column missing
        return pd.DataFrame(index=df.index)

    y = df[target_col]

    # Create lagged features
    # Prompt 1: Fix lag_featuring function's iteration
    lagged_df = pd.DataFrame(index=df.index)
    for lag in lags:
        lagged_df[f"{target_col}_lag{lag}"] = y.shift(lag)

    # Plot original with lagged features
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, y, label="Original")
    for col in lagged_df.columns:
        plt.plot(df.index, lagged_df[col], label=col)
    plt.legend()
    plt.title("Original Vs Lag Features")
    plt.show()

    return lagged_df

# FTT !!!
def frequency_featuring(df, target_col=None, sample_interval=1, filter_cutoff=0.1):
    if target_col is None or target_col not in df.columns:
        # Return None for all outputs if target column missing
        return None, None, None, None

    signal = df[target_col].ffill().bfill().values
    n = len(signal)

    # Low-pass filter
    # Prompt 2: Fix error and get low-pass criteria filtering
    b, a = butter(2, filter_cutoff, btype='low')
    signal = filtfilt(b, a, signal)

    # FFT
    fft_vals = fft(signal)
    magnitude = np.abs(fft_vals[:n//2])
    freqs = fftfreq(n, d=sample_interval)[:n//2]

    # Peaks
    peaks, _ = find_peaks(magnitude, prominence=np.max(magnitude)*0.1)

    # Periods
    periods = 1 / freqs[peaks]

    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(freqs, magnitude)
    plt.plot(freqs[peaks], magnitude[peaks], 'ro')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

    return freqs, magnitude, peaks, periods