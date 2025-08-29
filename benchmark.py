import pandas as pd
import numpy as np
import logging
import argparse
import time
from src.autogluon_imputer import AutoGluonImputer
from src.advanced_imputer import AdvancedImputer
from tests.test_utils import (
    load_titanic_dataset,
    load_adult_dataset,
    load_abalone_dataset,
    load_breast_cancer_dataset,
    introduce_missing_values,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_benchmark(dataset_name, mechanism, quality, verbosity, time_limit, imputer_to_test="AutoGluon", iterations=3):
    """
    Runs a benchmark comparison of different imputation methods on a specific dataset and missing data mechanism.
    """
    datasets = {
        "Titanic": load_titanic_dataset,
        "Adult": load_adult_dataset,
        "Abalone": load_abalone_dataset,
        "Breast Cancer": load_breast_cancer_dataset,
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(datasets.keys())}")

    df_original = datasets[dataset_name]()

    results = {}
    timing_results = {}

    print(f"\n--- Running benchmark on {dataset_name} dataset with {mechanism} mechanism, {quality} quality and {time_limit}s time limit ---")

    # Exclude columns with high cardinality from evaluation
    cols_to_exclude = [col for col in df_original.columns if df_original[col].count() > 0 and df_original[col].nunique() / df_original[col].count() > 0.95]

    # Label encode categorical features for scikit-learn imputers
    df_encoded = df_original.copy()
    le_dict = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le

    df_missing = introduce_missing_values(df_encoded, mechanism=mechanism, missing_fraction=0.2)

    imputers = {
        "Mean": SimpleImputer(strategy='mean'),
        "KNN": KNNImputer(),
        "MICE": IterativeImputer(max_iter=10, random_state=0)
    }
    if imputer_to_test == "AutoGluon":
        imputers["AutoGluon"] = AutoGluonImputer(quality=quality, verbosity=verbosity, time=time_limit)
    elif imputer_to_test == "Advanced":
        imputers["Advanced"] = AdvancedImputer(quality=quality, verbosity=verbosity, time=time_limit)


    for name, imputer in imputers.items():
        print(f"Running {name} imputer...")
        start_time = time.time()
        if name in ["AutoGluon", "Advanced"]:
            df_imputed = imputer.impute(df_missing)
        else:
            df_imputed_values = imputer.fit_transform(df_missing)
            df_imputed = pd.DataFrame(df_imputed_values, columns=df_missing.columns)
        end_time = time.time()
        timing_results[name] = end_time - start_time

        # Evaluate the imputation quality
        for col in df_original.columns:
            if col in cols_to_exclude:
                continue

            missing_mask = df_missing[col].isnull()
            original_values = df_encoded.loc[missing_mask, col]
            imputed_values = df_imputed.loc[missing_mask, col]

            if original_values.empty:
                continue

            key = (name, col)

            if pd.api.types.is_numeric_dtype(df_original[col]) and df_original[col].nunique() > 10:
                # Fill any remaining NaNs with 0 before calculating RMSE
                original_values = original_values.fillna(0)
                imputed_values = imputed_values.fillna(0)

                rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
                results[key] = rmse
            else:
                # Treat as categorical

                # Round imputed values to nearest integer
                imputed_values = np.round(imputed_values).astype(int)

                accuracy = accuracy_score(original_values, imputed_values)
                results[key] = accuracy

    # Print results
    print(f"\n--- Benchmark Results for {dataset_name} with {mechanism} and {quality} quality ---")
    results_df = pd.Series(results).unstack(level=0)
    print("--- Accuracy/RMSE Results ---")
    print(results_df.to_string())

    timing_df = pd.Series(timing_results)
    print("\n--- Timing Results (seconds) ---")
    print(timing_df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The name of the dataset to run the benchmark on.", choices=["Titanic", "Adult", "Abalone", "Breast Cancer"])
    parser.add_argument("mechanism", help="The missing data mechanism to use.", choices=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--quality", help="The quality setting for the AutoGluon imputer.", default="medium_quality", choices=["low_quality", "medium_quality", "good_quality", "high_quality", "best_quality"])
    parser.add_argument("--verbosity", help="The verbosity level for the AutoGluon imputer.", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--time_limit", help="The time limit for the AutoGluon imputer.", type=int, default=25)
    parser.add_argument("--imputer_to_test", help="The imputer to test.", default="AutoGluon", choices=["AutoGluon", "Advanced"])
    parser.add_argument("--iterations", help="The number of iterations for the AdvancedImputer.", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.dataset, args.mechanism, args.quality, args.verbosity, args.time_limit, args.imputer_to_test, args.iterations)
