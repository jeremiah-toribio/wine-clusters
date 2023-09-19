
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




############################################ OUTLIER FUNCTIONS #############################################

def get_fences(df, col, k=1.5) -> tuple:
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

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


############################################ SPLIT FUNCTION #############################################


def splitter(df, stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=df[stratify])

    train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=train[stratify])
    print(f'Dataframe: {df.shape}', '100%')
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')

    return train, validate, test


############################################ SCALING FUNCTION #############################################



def QuickScale(x_train, x_validate, x_test, linear=True, scaler='Standard', exclude_cols=None):
    '''
    Produces data scaled with each respective style, will utilize all unless specified otherwise.

    Arguments:
    x_train, x_validate, x_test: DataFrames to be scaled.
    linear: True for linear scaling (MinMax, Standard, Robust), False for non-linear (Quantile).
    scaler: Scaling method (MinMax, Standard, Robust, or Quantile).
    exclude_cols: List of column names to exclude from scaling.

    Returns: Original dataframes with the new scaled data added as new columns.
    '''
    
    # Check for valid scaler choice.
    if scaler not in ['MinMax', 'Standard', 'Robust', 'Quantile']:
        raise ValueError('Select a valid scaler.')
    
    # Initialize scalers.
    if linear:
        if scaler == 'MinMax':
            mmscaler = MinMaxScaler()
        elif scaler == 'Standard':
            nscaler = StandardScaler()
        elif scaler == 'Robust':
            rscaler = RobustScaler()
    else:
        qscaler = QuantileTransformer()
    
    # Exclude columns from scaling if specified.
    if exclude_cols is not None:
        cols_to_scale = [col for col in x_train.columns if col not in exclude_cols]
    else:
        cols_to_scale = x_train.columns
    
    # Scale the specified columns and add the scaled columns to the original dataframes.
    if linear:
        if scaler == 'MinMax':
            x_train_scaled = x_train.copy()
            x_val_scaled = x_validate.copy()
            x_test_scaled = x_test.copy()
            
            for col in cols_to_scale:
                col_name = f"{col}_scaled"
                x_train_scaled[col_name] = mmscaler.fit_transform(x_train_scaled[[col]])
                x_val_scaled[col_name] = mmscaler.transform(x_val_scaled[[col]])
                x_test_scaled[col_name] = mmscaler.transform(x_test_scaled[[col]])
        elif scaler == 'Standard':
            x_train_scaled = x_train.copy()
            x_val_scaled = x_validate.copy()
            x_test_scaled = x_test.copy()
            
            for col in cols_to_scale:
                col_name = f"{col}_scaled"
                x_train_scaled[col_name] = nscaler.fit_transform(x_train_scaled[[col]])
                x_val_scaled[col_name] = nscaler.transform(x_val_scaled[[col]])
                x_test_scaled[col_name] = nscaler.transform(x_test_scaled[[col]])
        elif scaler == 'Robust':
            x_train_scaled = x_train.copy()
            x_val_scaled = x_validate.copy()
            x_test_scaled = x_test.copy()
            
            for col in cols_to_scale:
                col_name = f"{col}_scaled"
                x_train_scaled[col_name] = rscaler.fit_transform(x_train_scaled[[col]])
                x_val_scaled[col_name] = rscaler.transform(x_val_scaled[[col]])
                x_test_scaled[col_name] = rscaler.transform(x_test_scaled[[col]])
    else:
        x_train_scaled = x_train.copy()
        x_val_scaled = x_validate.copy()
        x_test_scaled = x_test.copy()
        
        for col in cols_to_scale:
            col_name = f"{col}_scaled"
            x_train_scaled[col_name] = qscaler.fit_transform(x_train_scaled[[col]])
            x_val_scaled[col_name] = qscaler.transform(x_val_scaled[[col]])
            x_test_scaled[col_name] = qscaler.transform(x_test_scaled[[col]])

    return x_train_scaled, x_val_scaled, x_test_scaled


#################################### SPLIT SCALED AND UNSCALED FUNCTION #####################################


def retrieve_dataframes(train_scaled_added, validate_scaled_added, test_scaled_added):
    '''
    Inputs: scaled df's for train, validate, test.
    Returns: a scaled and unscaled df with the target column 'quality' and unscaled column 'is_red' retained for train, validate, test
    '''
    train = train_scaled_added.drop(columns=['fixed_acidity_scaled', 'volatile_acidity_scaled', 'citric_acid_scaled',
       'residual_sugar_scaled', 'chlorides_scaled',
       'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled',
       'density_scaled', 'ph_scaled', 'sulphates_scaled', 'alcohol_scaled'])

    train_scaled = train_scaled_added.drop(columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'ph', 'sulphates', 'alcohol'])

    validate = validate_scaled_added.drop(columns=['fixed_acidity_scaled', 'volatile_acidity_scaled', 'citric_acid_scaled',
           'residual_sugar_scaled', 'chlorides_scaled',
           'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled',
           'density_scaled', 'ph_scaled', 'sulphates_scaled', 'alcohol_scaled'])

    validate_scaled = validate_scaled_added.drop(columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'ph', 'sulphates', 'alcohol'])

    test = test_scaled_added.drop(columns=['fixed_acidity_scaled', 'volatile_acidity_scaled', 'citric_acid_scaled',
           'residual_sugar_scaled', 'chlorides_scaled',
           'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled',
           'density_scaled', 'ph_scaled', 'sulphates_scaled', 'alcohol_scaled'])

    test_scaled = test_scaled_added.drop(columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'ph', 'sulphates', 'alcohol'])
    return train, train_scaled, validate, validate_scaled, test, test_scaled


####################################  #####################################


def organize_columns(train):
    '''
    Distinguishes between numeric and categorical data types
    Only selecting columns that would be relevant to visualize, no encoded data.
    '''
    cat_cols, num_cols = [], []
    explore = train
    for col in explore:
        # check to see if its an object type,
        # if so toss it in categorical
        if train[col].dtype == 'O':
            cat_cols.append(col)
        # otherwise if its numeric:
        else:
            # check to see if we have more than just a few values:
            # if thats the case, toss it in categorical
            if train[col].nunique() < 5:
                cat_cols.append(col)
            else:
                num_cols.append(col)
    return cat_cols, num_cols