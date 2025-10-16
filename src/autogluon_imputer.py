import warnings
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=FutureWarning, module="autogluon.*")

class AutoGluonImputer:
    """A class to impute missing values in a pandas DataFrame using AutoGluon.

    This imputer works by iteratively finding the column with the fewest missing
    values, training a model to predict those values based on the other columns,
    and then imputing them. This process is repeated until all missing values
    in the DataFrame are imputed.

    Note:
        This imputer can be memory-intensive as it keeps all trained models
        in memory until the imputation process is complete. For a more
        memory-efficient version, consider using `AdvancedImputer`.

    Attributes:
        time (int): The time limit in seconds for the model training process for
            each column.
        quality (str): The quality preset for the AutoGluon TabularPredictor.
        verbosity (int): The verbosity level for AutoGluon's logging.
    """
    def __init__(self, time=25, quality="medium_quality", verbosity=0):
        """Initializes the AutoGluonImputer.

        Args:
            time (int, optional): The time limit in seconds for the model
                training process for each column. Defaults to 25.
            quality (str, optional): The quality level of the model training
                process. Accepted values are "low_quality", "medium_quality",
                "good_quality", "high_quality", "best_quality". Defaults to
                "medium_quality".
            verbosity (int, optional): Verbosity level of AutoGluon. Can be 0,
                1, 2, 3, or 4. Defaults to 0.
        """
        self.time = time
        self.quality = quality
        self.verbosity = verbosity

    def impute(self, df):
        """Imputes missing values in a pandas DataFrame.

        This method iteratively imputes missing values in the DataFrame, starting
        with the column that has the fewest missing values.

        Args:
            df (pd.DataFrame): The DataFrame with missing values to be imputed.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
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
        """Fits a model and imputes a single feature in the DataFrame.

        This method trains a model on the non-missing values of a single column
        and then uses this model to predict the missing values.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be imputed.
            column (str): The name of the column to impute.

        Returns:
            pd.DataFrame: The DataFrame with the specified column imputed.
        """
        logging.info(f"Training model for column: {column}")
        non_nan_df = df.dropna(subset=[column])
        nan_df = df[pd.isna(df[column])]
        predictor, _ = self._train_on_column(non_nan_df, column)

        logging.info(f"Predicting missing values for column: {column}")
        predictions = predictor.predict(nan_df)
        df_imput = df.copy(deep=True)
        df_imput.loc[predictions.index, column] = predictions

        return df_imput

    def _train_on_column(self, df, column):
        """Trains a model on a specific column of a DataFrame using AutoGluon.

        This method determines the problem type (regression or letting AutoGluon
        decide) based on the number of unique values in the column and then
        trains an AutoGluon TabularPredictor.

        Args:
            df (pd.DataFrame): The DataFrame to train the model on. This should
                not contain missing values in the target `column`.
            column (str): The name of the target column for which to train the
                model.

        Returns:
            tuple: A tuple containing:
                - TabularPredictor: The trained AutoGluon predictor.
                - dict: A dictionary of evaluation metrics for the trained model.
        """
        # Determine if the column is for a regression problem
        unique_numerical_values = pd.to_numeric(df[column], errors="coerce").nunique(
            dropna=True
        )
        if unique_numerical_values > 10:
            problem_type = "regression"
            eval_metric = "r2"
        else:
            problem_type = None
            eval_metric = None # Let AutoGluon decide

        autogluon_df = TabularDataset(df)

        predictor = TabularPredictor(
            label=column, verbosity=self.verbosity, problem_type=problem_type, eval_metric=eval_metric
        ).fit(autogluon_df, presets=self.quality, time_limit=self.time)

        metrics = predictor.evaluate(autogluon_df, silent=True)
        logging.info(f"Model for column {column} trained. Metrics: {metrics}")

        return predictor, metrics
