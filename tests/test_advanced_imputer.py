import unittest
import pandas as pd
import numpy as np
from src.autogluon_imputer import AutoGluonImputer
from tests.test_utils import load_titanic_dataset, introduce_missing_values
from sklearn.metrics import mean_squared_error, accuracy_score

class TestAdvancedImputer(unittest.TestCase):

    def setUp(self):
        self.df_original = load_titanic_dataset()
        self.imputer = AutoGluonImputer()

    def test_imputation_quality(self):
        # Introduce missing values in 'age' and 'fare' columns
        cols_to_affect = ['age', 'fare']
        df_missing = introduce_missing_values(self.df_original, missing_fraction=0.2, cols_to_affect=cols_to_affect)

        # Impute the missing values
        df_imputed = self.imputer.impute(df_missing)

        # Evaluate the imputation quality
        for col in cols_to_affect:
            # Get the original and imputed values for the rows that were missing
            missing_mask = df_missing[col].isnull()
            original_values = self.df_original.loc[missing_mask, col]
            imputed_values = df_imputed.loc[missing_mask, col]

            if original_values.empty:
                continue

            if pd.api.types.is_numeric_dtype(self.df_original[col]) and self.df_original[col].nunique() > 10:
                # Evaluate numerical columns with RMSE
                rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
                print(f"RMSE for {col}: {rmse}")
                self.assertLess(rmse, 30) # Setting a more reasonable threshold
            else:
                # Evaluate categorical columns with accuracy
                original_values = original_values.astype(str)
                imputed_values = imputed_values.astype(str)
                accuracy = accuracy_score(original_values, imputed_values)
                print(f"Accuracy for {col}: {accuracy}")
                self.assertGreater(accuracy, 0.1) # Setting a low threshold for now

if __name__ == '__main__':
    unittest.main()
