import unittest
import pandas as pd
import numpy as np
from src.advanced_imputer import AdvancedImputer

class TestAdvancedImputer(unittest.TestCase):

    def setUp(self):
        self.data = {
            'A': [1, 2, np.nan, 2, 1],
            'B': [1, 1, 0, 0, 1],
            'C': [0, 0, 1, 1, np.nan]
        }
        self.df = pd.DataFrame(self.data)
        self.imputer = AdvancedImputer()

    def test_impute(self):
        # Impute the missing values
        imputed_df = self.imputer.impute(self.df)

        # Check that there are no missing values in the imputed DataFrame
        self.assertFalse(imputed_df.isnull().values.any())

        # Check if the imputed values are reasonable
        imputed_value_A = imputed_df.loc[2, 'A']
        self.assertTrue(isinstance(imputed_value_A, (int, float)))

        imputed_value_C = imputed_df.loc[4, 'C']
        self.assertIn(imputed_value_C, [0, 1])

    def test_impute_no_missing_values(self):
        df_no_missing = self.df.dropna()
        imputed_df = self.imputer.impute(df_no_missing)
        self.assertTrue(df_no_missing.equals(imputed_df))

    def test_impute_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.imputer.impute("not a dataframe")

    def test_high_cardinality_exclusion(self):
        df_high_cardinality = self.df.copy()
        df_high_cardinality['high_card'] = [str(i) for i in range(len(df_high_cardinality))]
        imputed_df = self.imputer.impute(df_high_cardinality)
        self.assertTrue(imputed_df['high_card'].equals(df_high_cardinality['high_card']))

    def test_regression_imputation(self):
        df = pd.DataFrame({
            'A': np.linspace(0, 1, 20),
            'B': np.linspace(0, 2, 20),
            'C': np.linspace(0, 3, 20),
        })
        df.loc[5, 'B'] = np.nan
        imputed_df = self.imputer.impute(df)
        self.assertFalse(imputed_df.isnull().values.any())
        self.assertAlmostEqual(imputed_df.loc[5, 'B'], 0.5, delta=0.5)

    def test_classification_imputation(self):
        df = pd.DataFrame({
            'A': [1, 1, 1, 0, 0, 0],
            'B': [1, 1, 1, 0, 0, 0],
            'C': [1, 1, 1, 0, 0, np.nan]
        })
        imputed_df = self.imputer.impute(df)
        self.assertFalse(imputed_df.isnull().values.any())
        self.assertIn(imputed_df.loc[5, 'C'], [0, 1])

if __name__ == '__main__':
    unittest.main()
