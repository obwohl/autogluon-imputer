# AutoGluon Imputer

This project provides two powerful imputers for tabular data that leverage [AutoGluon](https://auto.gluon.ai/stable/index.html) to predict and fill missing values: `AutoGluonImputer` and `AdvancedImputer`.

## Installation

To use the imputers, you need to have AutoGluon and other dependencies installed. You can set up the environment using Conda:

```bash
# It is recommended to use a Conda environment
conda create -n autogluon_imputer python=3.8
conda activate autogluon_imputer
pip install autogluon pandas scikit-learn
```

You will also need to download the datasets for the benchmark:
```bash
python download_datasets.py
```

## Imputers

This project offers two imputer classes:

1.  **`AutoGluonImputer`**: A straightforward imputer that iteratively trains models to fill in missing data. It holds all trained models in memory, which can be resource-intensive.
2.  **`AdvancedImputer`**: A more memory-efficient version of the imputer. It cleans up trained models from disk after each feature imputation, making it suitable for larger datasets or scenarios with limited memory.

## Usage

Here are examples of how to use both imputers.

### `AutoGluonImputer`

```python
import pandas as pd
import numpy as np
from src.autogluon_imputer import AutoGluonImputer

# Create a sample DataFrame with missing values
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [1, None, 3, 4, 5, None, 7, 8, 9, 10],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Instantiate the imputer
imputer = AutoGluonImputer()

# Impute the missing values
imputed_df = imputer.impute(df)

# Print the imputed DataFrame
print(imputed_df)
```

### `AdvancedImputer`

The `AdvancedImputer` is used in the same way, but it is more memory-efficient.

```python
import pandas as pd
import numpy as np
from src.advanced_imputer import AdvancedImputer

# Create a sample DataFrame with missing values
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [1, None, 3, 4, 5, None, 7, 8, 9, 10],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Instantiate the imputer
imputer = AdvancedImputer()

# Impute the missing values
imputed_df = imputer.impute(df)

# Print the imputed DataFrame
print(imputed_df)
```

## Benchmark

A benchmark was performed to compare the `AdvancedImputer` with other common imputation methods from scikit-learn: Mean, KNN, and MICE. The benchmark was run on the Titanic and Abalone datasets with three different missing data mechanisms: MCAR, MAR, and MNAR.

### Missing Data Mechanisms

*   **MCAR (Missing Completely At Random):** The probability of a value being missing is independent of both the observed and unobserved data.
*   **MAR (Missing At Random):** The probability of a value being missing depends only on the observed data.
*   **MNAR (Missing Not At Random):** The probability of a value being missing depends on the unobserved data itself.

### Evaluation Metrics

*   **RMSE (Root Mean Squared Error):** Used for numerical columns. It measures the average difference between the imputed values and the original values. A lower RMSE is better.
*   **Accuracy:** Used for categorical columns. It measures the percentage of correctly imputed values. A higher accuracy is better.

### Running the Benchmark

You can run the benchmark yourself using the `benchmark.py` script:

```bash
python benchmark.py <dataset_name> <mechanism>
```

For example:
```bash
python benchmark.py Titanic MCAR
```

### Benchmark Results

The `AdvancedImputer` generally outperforms traditional methods, especially in terms of accuracy for categorical features. The performance of `AdvancedImputer` is very similar to `AutoGluonImputer` as they are based on the same core logic, but `AdvancedImputer` is more memory-efficient.

#### Titanic Dataset

| Mechanism | Metric        | AdvancedImputer | Mean   | KNN    | MICE   |
|-----------|---------------|-----------------|--------|--------|--------|
| MCAR      | age (RMSE)      | 12.52           | 13.67  | 14.07  | 13.04  |
| MCAR      | fare (RMSE)     | 56.86           | 64.14  | 61.42  | 56.45  |
| MCAR      | Avg. Accuracy | 0.79            | 0.58   | 0.65   | 0.70   |
| MAR       | age (RMSE)      | 12.23           | 13.29  | 14.11  | 12.72  |
| MAR       | fare (RMSE)     | 45.81           | 58.58  | 62.42  | 49.96  |
| MAR       | Avg. Accuracy | 0.78            | 0.56   | 0.68   | 0.73   |
| MNAR      | age (RMSE)      | 13.22           | 14.86  | 13.93  | 13.98  |
| MNAR      | fare (RMSE)     | 48.76           | 57.33  | 51.02  | 49.89  |
| MNAR      | Avg. Accuracy | 0.74            | 0.61   | 0.62   | 0.76   |

#### Abalone Dataset

| Mechanism | Metric         | AdvancedImputer | Mean   | KNN    | MICE   |
|-----------|----------------|-----------------|--------|--------|--------|
| MCAR      | RMSE (avg)     | 0.04            | 0.18   | 0.07   | 0.05   |
| MCAR      | Accuracy (sex) | 0.55            | 0.33   | 0.40   | 0.33   |
| MAR       | RMSE (avg)     | 0.03            | 0.18   | 0.05   | 0.03   |
| MAR       | Accuracy (sex) | 0.57            | 0.31   | 0.41   | 0.30   |
| MNAR      | RMSE (avg)     | 0.04            | 0.18   | 0.06   | 0.04   |
| MNAR      | Accuracy (sex) | 0.57            | 0.32   | 0.40   | 0.32   |

### Limitations

The benchmark for the Breast Cancer and Adult datasets could not be completed due to timeouts in the environment, even with increased time limits. The AutoGluon-based imputers are computationally expensive, especially for larger datasets and more complex missing data mechanisms.