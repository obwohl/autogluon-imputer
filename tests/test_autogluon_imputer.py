import unittest
import pandas as pd
import numpy as np
from src.autogluon_imputer import AutoGluonImputer

class TestAutoGluonImputer(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with missing values
        self.data = {
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [1, np.nan, 3, 4, 5, np.nan, 7, 8, 9, 10],
            'feature3': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        self.df = pd.DataFrame(self.data)
        self.imputer = AutoGluonImputer()

    def test_impute(self):
        # Impute the missing values
        imputed_df = self.imputer.impute(self.df)

        # Check that there are no missing values in the imputed DataFrame
        self.assertFalse(imputed_df.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
