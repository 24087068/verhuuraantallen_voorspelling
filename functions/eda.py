import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def eda(df, target_col, nominal_cols, ordinal_cols, timeseries_col):
    """
    eda overzicht: basis statistiek, distributies, skew/kurtosis,
    relaties met target en tijdreeksanalyse.

    Parameters:
        df (pd.DataFrame): Input dataset
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
    print("# Numerical kolommen correlaties:")
    num_cols = continuous_cols + [target_col]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True)
    plt.title("Continuous correlaties")
    plt.show()

    # Time series
    print("Time series:")
    plt.figure(figsize=(15,5))
    df_sorted = df.sort_values(timeseries_col)
    sns.lineplot(x=timeseries_col, y=target_col, data=df_sorted)
    plt.title(f"{target_col} over {timeseries_col}")
    plt.show()