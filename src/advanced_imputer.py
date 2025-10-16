import warnings
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon.*")

class AdvancedImputer:
    """
    A class to impute missing values in a pandas DataFrame using AutoGluon.

    This imputer iteratively finds the column with the fewest missing values,
    trains a model to predict those values based on the other columns,
    and then imputes them. This process is repeated until all missing values
    in the DataFrame are imputed.
    """
    def __init__(self, time=25, quality="medium_quality", verbosity=0):
        """
        Initializes the AdvancedImputer.
        """
        self.time = time
        self.quality = quality
        self.verbosity = verbosity

    def impute(self, df):
        """
        Imputes missing values in a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df_imputed = df.copy()
        logging.info("Starting imputation process.")

        # Exclude high-cardinality object columns
        cols_to_exclude = [
            col for col in df.columns
            if df[col].dtype == 'object' and df[col].count() > 0 and df[col].nunique() / df[col].count() > 0.95
        ]
        if cols_to_exclude:
            logging.info(f"Excluding columns with high cardinality: {cols_to_exclude}")

        cols_to_impute = [col for col in df.columns if col not in cols_to_exclude]

        while True:
            nan_counts = df_imputed[cols_to_impute].isna().sum()
            nan_counts = nan_counts[nan_counts > 0]

            if nan_counts.empty:
                logging.info("No missing values found in columns to impute. Imputation complete.")
                break

            column_with_least_nans = nan_counts.idxmin()
            logging.info(f"Imputing column: {column_with_least_nans}")

            df_imputed = self._fit_impute_single_feature(df_imputed, column_with_least_nans)

        logging.info("Imputation process finished.")
        return df_imputed

    def _fit_impute_single_feature(self, df, column):
        """
        Fits a model and imputes a single feature in the DataFrame.
        """
        logging.info(f"Training model for column: {column}")
        non_nan_df = df.dropna(subset=[column])
        nan_df = df[pd.isna(df[column])]
        predictor, _ = self._train_on_column(non_nan_df, column)

        logging.info(f"Predicting missing values for column: {column}")
        predictions = predictor.predict(nan_df)
        df_imput = df.copy(deep=True)
        df_imput.loc[predictions.index, column] = predictions
        shutil.rmtree(predictor.path)

        return df_imput

    def _train_on_column(self, df, column):
        """
        Trains a model on a specific column of a DataFrame using AutoGluon.
        """
        # Determine if the column is for a regression problem
        unique_numerical_values = pd.to_numeric(df[column], errors="coerce").nunique(
            dropna=True
        )
        if unique_numerical_values > 10:
            problem_type = "regression"
        else:
            problem_type = "multiclass"

        autogluon_df = TabularDataset(df)

        predictor = TabularPredictor(
            label=column, verbosity=self.verbosity, problem_type=problem_type
        ).fit(autogluon_df, presets=self.quality, time_limit=self.time)

        metrics = predictor.evaluate(autogluon_df, silent=True)
        logging.info(f"Model for column {column} trained. Metrics: {metrics}")

        return predictor, metrics
