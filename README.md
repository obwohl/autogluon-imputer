# AutoGluon Imputer

This project provides a simple to use imputer for tabular data that leverages the power of [AutoGluon](https://auto.gluon.ai/stable/index.html).

## Installation

To use the AutoGluon Imputer, you need to have AutoGluon installed. You can install it using pip:

```bash
pip install autogluon
```

## Usage

Here is a simple example of how to use the `AutoGluonImputer`:

```python
import pandas as pd
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

## Benchmark

A benchmark was performed on the Titanic dataset to compare the `AutoGluonImputer` with other common imputation methods from scikit-learn. Missing values were introduced into the `age` and `fare` columns (20% missing values). The Root Mean Squared Error (RMSE) was used to evaluate the imputation quality.

The results are as follows:

| Imputer         |      age |       fare |
|-----------------|----------|------------|
| AutoGluon       | 6.341141 |  11.664696 |
| Mean            | 14.981022|  61.157868 |
| KNN             | 15.563928|  62.346365 |
| MICE            | 13.071308|  49.365534 |

As the results show, the `AutoGluonImputer` significantly outperforms the other methods, achieving a much lower RMSE on both columns. This indicates a more accurate and reliable imputation.
