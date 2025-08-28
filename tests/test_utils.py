import pandas as pd
import numpy as np

def load_titanic_dataset():
    """
    Loads the Titanic dataset from the data directory.

    Returns:
    -------
    pd.DataFrame
        The Titanic dataset as a pandas DataFrame.
    """
    return pd.read_csv('data/titanic.csv')

def introduce_missing_values(df, missing_fraction=0.1, cols_to_affect=None):
    """
    Introduces missing values (NaN) into a DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame to introduce missing values into.
    missing_fraction : float, optional
        The fraction of values to replace with NaN. Defaults to 0.1.
    cols_to_affect : list, optional
        A list of columns to introduce missing values into. If None, it will
        introduce missing values into a random subset of columns. Defaults to None.
    """
    df_missing = df.copy()
    if cols_to_affect is None:
        # Choose a random subset of columns to introduce missing values into
        n_cols_to_affect = np.random.randint(1, len(df.columns))
        cols_to_affect = np.random.choice(df.columns, n_cols_to_affect, replace=False)

    for col in cols_to_affect:
        n_missing = int(len(df_missing) * missing_fraction)
        missing_indices = np.random.choice(df_missing.index, n_missing, replace=False)
        df_missing.loc[missing_indices, col] = np.nan
    return df_missing
