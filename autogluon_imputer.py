import warnings

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon.*")
sep = "-" * 50
longsep = "-" * 100


def train_on_column(df, column, time=5, quality="medium_quality"):
    """
    Train a model on a specific column of a DataFrame using AutoGluon.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to train the model on.
    time (int): The time limit for the fit process in seconds.
    quality (str): The quality level of the model training. Accepted values are "medium_quality", "good_quality", "high_quality", "best_quality".

    Returns:
    Tuple[TabularPredictor, dict]: A tuple containing the trained predictor and its evaluation metrics.
    """

    # Determine if the column is for a regression problem
    unique_numerical_values = pd.to_numeric(df[column], errors="coerce").nunique(
        dropna=True
    )
    PROBLEM_TYPE = "regression" if unique_numerical_values > 10 else None
    if PROBLEM_TYPE == "regression":
        EVAL_METRIC = "r2"
    else:
        EVAL_METRIC = "accuracy"

    print(f"fitting {column}")

    # Prepare training data
    df_train = df.dropna(subset=[column])
    autogluon_df = TabularDataset(df_train)

    # Train the predictor
    predictor = TabularPredictor(
        label=column, verbosity=2, problem_type=PROBLEM_TYPE, eval_metric=EVAL_METRIC
    ).fit(autogluon_df, presets=quality, time_limit=time)

    # Evaluate the predictor
    metrics = predictor.evaluate(autogluon_df)

    return predictor, metrics


def fit_impute_single_feature(df, column, time=5, quality="medium_quality"):
    """
    Fit and impute a single feature in a DataFrame using a trained model.

    This function trains a predictor on a DataFrame with the specified column, imputes missing values in that column, and reports the prediction metrics.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data with missing values.
    column (str): The name of the column to fit and impute.
    time (int): The time limit for the fit process in seconds.
    quality (str): The quality level of the model training. Accepted values are "medium_quality", "good_quality", "high_quality", "best_quality".

    Returns:
    pd.DataFrame: The DataFrame with the specified column imputed.
    """

    # fit
    non_nan_df = df.dropna(subset=[column])
    nan_df = df[pd.isna(df[column])]
    predictor, metrics = train_on_column(non_nan_df, column, time, quality)

    # report how good the NaN-feature can be predicted
    print("sep")
    print(f"The feature {column} was fitted and imputed with these metrics: {metrics}")
    print("sep")

    # impute
    predictions = predictor.predict(nan_df)
    df_imput = df.copy(deep=True)
    df_imput.loc[predictions.index, column] = predictions

    return df_imput


def iterative_autogluon_imputer(
    df,
    label,
    ERROR_METRIC="mcc",
    time=5,
    quality="medium_quality",
    always_use_best=False,
    use_fallback_imputation=False,
):
    """
    Perform iterative imputation on a DataFrame using AutoGluon.

    This function iteratively imputes missing values in the DataFrame and trains a predictor on the specified label column

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data with missing values.
    label (str): The name of the label column for training the predictor.
    ERROR_METRIC (str): The metric name to evaluate the improvement of the imputation.
    time (int): The time limit for the training process in seconds.
    quality (str): The quality level of the model training. Accepted values are "low_quality", "medium_quality", and "high_quality".
    always_use_best (bool): Whether to always continue with highest imputation eval, regardless of its performance compared to the baseline. This guarantees full df imputation.
    use_fallback_imputation (bool): Whether to use a fallback imputation method if no improvement is found. (uses median/mode for numeric/categorical). This guarantees full df imputation.

    Returns:
    Tuple[pd.DataFrame, TabularPredictor, dict]: A tuple containing the imputed DataFrame, the final trained predictor, and its evaluation metrics.
    """

    columns_with_nan = df.columns[df.isna().any()].tolist()

    f = len(columns_with_nan)

    print(
        f"Your df has {f} NaN - columns. The imputation might take up to {time*(2*f**2+2*f+1)} seconds."
    )
    baseline_predictor, baseline_metric = train_on_column(df, label, time)

    while f > 0:
        metrics_label_imputed = {}
        predictors_label_imputed = {}
        imputed_dfs = {}

        for col in columns_with_nan:
            df_imput = fit_impute_single_feature(df, col)
            label_on_imputed_predictor, label_on_imputed_metric = train_on_column(
                df_imput, label, time, quality
            )
            print("sep")
            print(f"{label} on baseline df:      {baseline_metric} ({ERROR_METRIC})")
            print(
                f"{label} on {col}-imputed df: {label_on_imputed_metric} ({ERROR_METRIC}"
            )
            print("sep\n")
            predictors_label_imputed[col] = label_on_imputed_predictor

            metrics_label_imputed[col] = label_on_imputed_metric

            imputed_dfs[col] = df_imput

        # finding the best improvement and updating the baseline
        max_metric = float("-inf")
        improver_key = None
        for col, metrics in metrics_label_imputed.items():
            if metrics.get(ERROR_METRIC, float("-inf")) > max_metric:
                max_metric = metrics[ERROR_METRIC]
                improver_key = col

        if improver_key is not None and (
            always_use_best or max_metric > baseline_metric[ERROR_METRIC]
        ):
            print("longsep")
            print(f"New imputed baseline found with {improver_key}")
            if always_use_best and max_metric < baseline_metric[ERROR_METRIC]:
                print(
                    f"{improver_key} was NOT better than baseline but best of all candidates and always_use_best is set to 'True')"
                )
            print(
                f"METRICS({ERROR_METRIC})\n base: {baseline_metric[ERROR_METRIC]}\n {improver_key}: {metrics_label_imputed[improver_key][ERROR_METRIC]}"
            )
            print("longsep\n")
            baseline_predictor = predictors_label_imputed[improver_key]
            df = imputed_dfs[improver_key]
            columns_with_nan = df.columns[df.isna().any()].tolist()
            baseline_metric = metrics_label_imputed[improver_key]
            print(
                f"Updated baseline_metric: {baseline_metric}, type: {type(baseline_metric)}"
            )

        elif use_fallback_imputation:
            # Identify the column with the smallest number of missing values
            min_nan_col = columns_with_nan[0]
            min_nan_count = df[min_nan_col].isna().sum()
            for col in columns_with_nan:
                nan_count = df[col].isna().sum()
                if nan_count < min_nan_count:
                    min_nan_col = col
                    min_nan_count = nan_count

            # Check if the column is numeric or non-numeric
            if pd.api.types.is_numeric_dtype(df[min_nan_col]):
                # Impute numeric columns with median
                df[min_nan_col].fillna(df[min_nan_col].median(), inplace=True)
            else:
                # Impute non-numeric columns with mode (most frequent value)
                mode_value = df[min_nan_col].mode().iloc[0]
                df[min_nan_col].fillna(mode_value, inplace=True)

            print("longsep")
            print(
                f"Imputed column {min_nan_col} with {'median' if pd.api.types.is_numeric_dtype(df[min_nan_col]) else 'mode'} due to lack of improvement in prediction."
            )
            print("longsep")

            # Update
            columns_with_nan = df.columns[df.isna().any()].tolist()
            baseline_predictor, baseline_metric = train_on_column(
                df, label, time, quality
            )

        else:
            print("No improvement found and fallback imputation is disabled.")
            break

        f -= 1
        print("sep")
        print(f"{f} columns left to impute")
        print("sep")

    print("longsep")
    print("Finished")
    print("longsep")
    return df, baseline_predictor, baseline_metric


def impute_df_with_least_nans(df, time=25, quality="medium_quality"):
    """
    Imputes missing values in columns of a DataFrame by iteratively focusing on the column with the fewest missing values.

    This function iterates over the DataFrame's columns with missing values, starting with the column that has the least number of missing values. For each column, it trains an AutoGluon model to predict the missing values based on the other columns and imputes these values. It repeats this process, moving on to the next column with the least number of missing values until all missing values in the DataFrame are imputed.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data with missing values that need to be imputed.
    time (int, optional): The time limit in seconds for the model training process for each column. Defaults to 25.
    quality (str, optional): The quality level of the model training process. Accepted values are "medium_quality", "good_quality", "high_quality", "best_quality". Defaults to "medium_quality".

    Returns:
    pd.DataFrame: The DataFrame with missing values imputed.

    Notes:
    - The function selects the column with the minimum number of missing values and imputes it first, progressively moving to columns with more missing values.
    - The imputation model's performance can vary based on the 'time' and 'quality' parameters provided. Higher time limits and quality levels may improve the accuracy of imputation at the cost of longer computation time.
    - If a column's data type is numeric, the function may use regression models for imputation, and if it is categorical, classification models may be used.
    """
    while True:
        # count NaN and ignore Nan==0
        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]

        # break if no NaN is left
        if nan_counts.empty:
            break

        # get column with min NaNs
        column_with_least_nans = nan_counts.idxmin()

        # impute this column
        df = fit_impute_single_feature(df, column_with_least_nans, time, quality)

    return df
