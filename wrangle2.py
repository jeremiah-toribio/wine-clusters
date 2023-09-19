
# ==--==--== Imports ==--==--==
import os
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling: Scaling
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer




def remove_outliers(df, k=1.5, exclude_cols=None):
    """
    Remove rows from a DataFrame that contain outliers based on the fences calculated
    using the get_fences function. You can specify a list of columns to exclude from outlier detection.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    k (float): The multiplier for the inner quartile range in get_fences.
    exclude_cols (list or None): A list of column names to exclude from outlier detection.
                                Defaults to None (no exclusion).

    Returns:
    pandas.DataFrame: A new DataFrame with outliers removed.
    """
    # Create a copy of the original DataFrame
    cleaned_df = df.copy()

    # Create a set of column names to exclude for faster lookup
    exclude_set = set(exclude_cols) if exclude_cols else set()

    # Iterate over each column in the DataFrame
    for col in df.columns:
        if col not in exclude_set:
            lower_bound, upper_bound = get_fences(df, col, k)
        
            # Remove rows where the column value is below the lower_bound or above the upper_bound
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df