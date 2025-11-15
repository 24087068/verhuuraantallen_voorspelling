import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# EDA !!!
def eda(df, target_col, nominal_cols, ordinal_cols, timeseries_col):
    """
    eda overzicht: basis statistiek, distributies, skew/kurtosis,
    relaties met target en tijdreeksanalyse.
    Parameters:
        df (dataframe): Input dataset
        target_col (str): Doelvariabele
        nominal_cols (list): Nominale kolommen
        ordinal_cols (list): Ordinale kolommen
        timeseries_col (str): Tijdkolom
    """
    # Labeling cols
    excluded = nominal_cols + ordinal_cols + [target_col] + [timeseries_col]
    continuous_cols = [c for c in df.columns if c not in excluded]
    # Basic statistics for the continuous data
    print("# Basis statistiek:")
    print("Continuous:")
    desc = df[continuous_cols + [target_col]].describe()
    print(desc)
    # Basic statistics for the ordinal data
    print("Ordinal:")
    for c in ordinal_cols:
        print(f"{df[c].value_counts().sort_index()}")
    # Basic statistics for the nominal data
    print("Nominal:")
    for c in nominal_cols:
        print(f"{df[c].value_counts()}")

    # Frequency leaning distribution (and just general spread) for continuous data
    for c in continuous_cols:
        sns.histplot(df[c])
        plt.title(f"{c} distribution")
        plt.show()
        print(f"{c}: skew={df[c].skew():.2f}, kurtosis={df[c].kurt():.2f}")

    # Relationship Measures
    # Prompt 1: Fix categorical Relationship Measures
    for c in continuous_cols:
        rho, p = stats.spearmanr(df[c], df[target_col])
        print(f"Spearman ({c} vs {target_col}): rho={rho:.2f}, p={p:.4f}")
    # Ordinal vs target
    for c in ordinal_cols:
        rho, p = stats.spearmanr(df[c], df[target_col])
        print(f"Spearman ({c} vs {target_col}): rho={rho:.2f}, p={p:.4f}")
    # Nominal vs target
    for c in nominal_cols:
        if df[c].nunique() == 2:
            mapping = {v:i for i,v in enumerate(df[c].unique())}
            r, p = stats.pointbiserialr(df[c].map(mapping), df[target_col])
            print(f"Point-Biserial ({c} vs {target_col}): r={r:.2f}, p={p:.4f}")
        else:
            groups = [df[df[c]==val][target_col] for val in df[c].unique()]
            f, p = stats.f_oneway(*groups)
            print(f"ANOVA ({c} vs {target_col}): F={f:.2f}, p={p:.4f}")
    # Correlations
    num_cols = continuous_cols + [target_col]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True)
    plt.title("Continuous correlaties")
    plt.show()

    # Time Series
    print("Time series:")
    df_ts = df.sort_values(timeseries_col)
    df_ts[timeseries_col] = pd.to_datetime(df_ts[timeseries_col])
    # Prompt 1: Efficiently compute hourly autocorrelation and seasonality for tons of samples
    df_ts['hour'] = df_ts[timeseries_col].dt.hour
    # Average target per hour of day
    hourly_profile = df_ts.groupby('hour')[target_col].mean()
    plt.figure(figsize=(15,5))
    hourly_profile.plot(marker='o')
    plt.title("Hourly average pattern")
    plt.xlabel("Hour of day")
    plt.ylabel(f"Average {target_col}")
    plt.grid(True)
    plt.show()
    # Fast ACF on hourly-diffed series (reduces noise)
    plt.figure(figsize=(15,5))
    y_hourly = df_ts[target_col].diff().dropna()
    plot_acf(y_hourly, lags=24)
    plt.title("Hourly ACF")
    plt.show()

    # Date time info
    t = df_ts[timeseries_col]
    sampl = t.diff().unique()
    sampl_main = t.diff().mode()[0]
    print("Sampling frequency:", sampl) # all different samples
    print("Main Sampling frequency:", sampl_main) # most frequent one (sfreq)
    print("Total samples:", len(t)) # all the samples (N)
    print("Total time:", t.max() - t.min()) # Time (t)

    # Resample to daily
    df_ts = df_ts.set_index(timeseries_col)
    df_daily = df_ts.resample('D').mean()
    x = df_daily.index
    y = df_daily[target_col]

    # Original
    plt.figure(figsize=(15,5))
    plt.plot(x, y)
    plt.title("Original time series (Daily)")
    plt.show()
    # Decomposition
    result = seasonal_decompose(y, period=30, model='additive')
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    # Trend
    plt.figure(figsize=(15,5))
    plt.plot(x, trend)
    plt.title("Trend")
    plt.show()
    # Seasonality
    plt.figure(figsize=(15,5))
    plt.plot(x, seasonal)
    plt.title("Seasonality")
    plt.show()
    # Seasonality for one period
    one_season = seasonal[:30]
    plt.figure(figsize=(15,5))
    plt.plot(one_season.index, one_season.values)
    plt.title("One period")
    plt.show()
    # Noise
    plt.figure(figsize=(15,5))
    plt.plot(x, resid)
    plt.title("Noise")
    plt.show()
    # Autocorrelation and PACF for seasonality
    plt.figure(figsize=(15,5))
    plot_acf(y.dropna(), lags=90)
    plt.title("ACF")
    plt.show()
    plt.figure(figsize=(15,5))
    plot_pacf(y.dropna(), lags=90)
    plt.title("PACF")
    plt.show()



