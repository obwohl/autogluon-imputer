import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

def load_titanic_dataset():
    """
    Loads the Titanic dataset from the data directory.
    """
    return pd.read_csv('data/titanic.csv')

def load_adult_dataset():
    """
    Loads the Adult Income dataset from the data directory.
    """
    return pd.read_csv('data/adult.csv')

def load_abalone_dataset():
    """
    Loads the Abalone dataset from the data directory.
    """
    df = pd.read_csv('data/abalone.csv', header=None)
    df.columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    return df

def load_breast_cancer_dataset():
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset from the data directory.
    """
    df = pd.read_csv('data/breast-cancer-wisconsin.data', header=None)
    df.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']
    df = df.drop(columns=['id'])
    return df

def MAR_mask(X, p, p_obs):
    """
    Missing at Random mask generation.

    Parameters:
    ----------
    X : pd.DataFrame
        The DataFrame to generate the mask for.
    p : float
        The probability of a value being missing.
    p_obs : float
        The proportion of variables that are fully observed.

    Returns:
    -------
    np.ndarray
        A boolean mask indicating the missing values.
    """
    n, d = X.shape
    mask = np.zeros((n, d))
    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs

    # Randomly select observed and missing variables
    idx_obs = np.random.choice(d, d_obs, replace=False)
    idx_na = np.array([i for i in range(d) if i not in idx_obs])

    # Generate missing values based on a logistic model
    # The probability of being missing depends on the observed variables
    W = np.random.rand(d_obs, d_na)
    b = np.random.rand(1, d_na)

    X_obs = X.iloc[:, idx_obs].values

    # Sigmoid function
    h = 1 / (1 + np.exp(-(X_obs @ W + b)))

    mask[:, idx_na] = np.random.binomial(1, h * p)

    return mask.astype(bool)

def MNAR_mask_logistic(X, p, p_obs):
    """
    Missing Not at Random mask generation using a logistic model.

    Parameters:
    ----------
    X : pd.DataFrame
        The DataFrame to generate the mask for.
    p : float
        The probability of a value being missing.
    p_obs : float
        The proportion of variables that are fully observed.

    Returns:
    -------
    np.ndarray
        A boolean mask indicating the missing values.
    """
    n, d = X.shape
    mask = np.zeros((n, d))
    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs

    # Randomly select observed and missing variables
    idx_obs = np.random.choice(d, d_obs, replace=False)
    idx_na = np.array([i for i in range(d) if i not in idx_obs])

    # Generate missing values based on a logistic model
    # The probability of being missing depends on the values themselves
    W = np.random.rand(d, d)
    b = np.random.rand(1, d)

    # Sigmoid function
    h = 1 / (1 + np.exp(-(X.values @ W + b)))

    mask = np.random.binomial(1, h * p)

    return mask.astype(bool)


def introduce_missing_values(df, mechanism="MCAR", missing_fraction=0.1, p_obs=0.5, cols_to_affect=None):
    """
    Introduces missing values (NaN) into a DataFrame.
    """
    df_missing = df.copy()

    if mechanism == "MCAR":
        if cols_to_affect is None:
            cols_to_affect = df.columns
        for col in cols_to_affect:
            n_missing = int(len(df_missing) * missing_fraction)
            if n_missing > 0:
                missing_indices = np.random.choice(df_missing.index, n_missing, replace=False)
                df_missing.loc[missing_indices, col] = np.nan
    else:
        # For MAR and MNAR, we need to work with a numerical dataframe
        df_numeric = df.copy()
        for col in df_numeric.columns:
            if not pd.api.types.is_numeric_dtype(df_numeric[col]):
                df_numeric[col] = pd.factorize(df_numeric[col])[0]

        if mechanism == "MAR":
            mask = MAR_mask(df_numeric, missing_fraction, p_obs)
        elif mechanism == "MNAR":
            mask = MNAR_mask_logistic(df_numeric, missing_fraction, p_obs)
        else:
            raise ValueError("Unknown missing data mechanism")

        df_missing[mask] = np.nan

    return df_missing
