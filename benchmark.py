import pandas as pd
import numpy as np
from src.autogluon_imputer import AutoGluonImputer
from tests.test_utils import load_titanic_dataset, introduce_missing_values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

def run_benchmark():
    """
    Runs a benchmark comparison of different imputation methods.
    """
    df_original = load_titanic_dataset()

    # Label encode categorical features for scikit-learn imputers
    df_encoded = df_original.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    cols_to_affect = ['age', 'fare']
    df_missing = introduce_missing_values(df_encoded, missing_fraction=0.2, cols_to_affect=cols_to_affect)

    imputers = {
        "AutoGluon": AutoGluonImputer(),
        "Mean": SimpleImputer(strategy='mean'),
        "KNN": KNNImputer(),
        "MICE": IterativeImputer(max_iter=10, random_state=0)
    }

    results = {}

    for name, imputer in imputers.items():
        print(f"Running {name} imputer...")
        if name == "AutoGluon":
            # AutoGluonImputer works with the original dataframe with categorical features
            df_missing_ag = introduce_missing_values(df_original, missing_fraction=0.2, cols_to_affect=cols_to_affect)
            df_imputed = imputer.impute(df_missing_ag)
        else:
            df_imputed_values = imputer.fit_transform(df_missing)
            df_imputed = pd.DataFrame(df_imputed_values, columns=df_missing.columns)

        rmse_scores = {}
        for col in cols_to_affect:
            missing_mask = df_missing[col].isnull()
            original_values = df_encoded.loc[missing_mask, col]
            imputed_values = df_imputed.loc[missing_mask, col]

            rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
            rmse_scores[col] = rmse

        results[name] = rmse_scores

    # Print results
    print("\n--- Benchmark Results (RMSE) ---")
    results_df = pd.DataFrame(results).T
    print(results_df)

if __name__ == '__main__':
    run_benchmark()
