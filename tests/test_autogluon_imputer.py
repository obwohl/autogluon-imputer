import unittest
import pandas as pd
import numpy as np
from src.advanced_imputer import AdvancedImputer

class TestAutoGluonImputer(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
