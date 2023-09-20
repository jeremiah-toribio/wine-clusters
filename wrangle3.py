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

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(f' {df[col][(df[col] > upper_bound) | (df[col] < lower_bound)]} ')
            print('----------')
            
        
def remove_outliers(df, k=1.5, exclude_col='quality'):
    """
    Remove rows from a DataFrame that contain outliers based on the fences calculated
    using the get_fences function.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    k (float): The multiplier for the inner quartile range in get_fences.

    Returns:
    pandas.DataFrame: A new DataFrame with outliers removed.
    """
    # Create a copy of the original DataFrame
    cleaned_df = df.copy()

    # Iterate over each column in the DataFrame
    for col in df.columns:
        if col != exclude_col:
            lower_bound, upper_bound = get_fences(df, col, k)
        
        # Remove rows where the column value is below the lower_bound or above the upper_bound
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df


################### SPLIT FUNCTION ##################

def splitter(df, stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    if stratify == None:
        train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=None)
        print(f'Dataframe: {df.shape}', '100%')
        print(f'Train: {train.shape}', '| ~60%')
        print(f'Validate: {validate.shape}', '| ~20%')
        print(f'Test: {test.shape}','| ~20%')

    else:
        train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=df[stratify])

        train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=train[stratify])
        print(f'Dataframe: {df.shape}', '100%')
        print(f'Train: {train.shape}', '| ~60%')
        print(f'Validate: {validate.shape}', '| ~20%')
        print(f'Test: {test.shape}','| ~20%')

    return train, validate, test



def summarize_df(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.

    returns:
        pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].dtype
            ]
        )

        check_columns = pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "dtype"],
    ).sort_values(by="Number of Unique Values")
   
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return check_columns



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

def check_cat_distribution(df,target='quality'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with categorical datatypes, since the only 
    plot used is a countplot, with a target used as the hue to compare.
    '''
    
    for col in df:
        plt.figure(figsize=(12.5,8))
        sns.countplot(data=df,x=col,alpha=0.8,linewidth=.4,edgecolor='black')
        plt.title('# of Observations by County')
        plt.show()
        #print('''-------------------------------------------------------------''')\
    

def check_num_distribution(df,target='quality'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with numerical datatypes, since the 
    plots used are hist plot and box plot, with a target used as the hue to compare.
    '''
    for col in df:
        sns.histplot(data=df, x=df[col],hue=target)
        t = col.lower()
        plt.title(t)
        plt.show()
        sns.boxplot(data=df, x=col,hue=target)
        plt.title(t)
        plt.show()
        print('''-------------------------------------------------------------''')

###               ###
# #     Scaler    # #
###               ###

def QuickScale(x_train, x_validate, x_test, linear=True, scaler='Standard'):
    '''
    Produces data scaled with each respective style, will utilize all unless specificied otherwise.

    Arguments: x_train = desired data frame; and respected validate and test, Linear= True or False

    Returns: 6 or 2 arrays that would need to be assigned
    '''
    # Check for linear keyword argument to choose how to scale.
    if linear==True:
        if scaler == 'MinMax':
            mmscaler = MinMaxScaler()
        elif scaler == 'Standard':
            nscaler = StandardScaler()
        elif scaler == 'Robust':
            rscaler = RobustScaler()
        else:
            raise ValueError('Select a valid scaler.')
    else:
        # Non Linear Scaler 
        qscaler = QuantileTransformer()
        # train
        x_train_scaled = qscaler.fit_transform(x_train)
        # validate
        x_val_scaled = qscaler.transform(x_validate)
        # test
        x_test_scaled = qscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

        ##### change scaler to be kwarg to reduce output -- , scaler= #####
        # train
    if scaler == 'MinMax':
        x_train_scaled = mmscaler.fit_transform(x_train)
        x_val_scaled = mmscaler.transform(x_validate)
        x_test_scaled = mmscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

    elif scaler == 'Standard':
        x_train_scaled = nscaler.fit_transform(x_train)
        x_val_scaled = nscaler.transform(x_validate)
        x_test_scaled = nscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled

    elif scaler == 'Robust':
        x_train_scaled = rscaler.fit_transform(x_train)
        x_val_scaled = rscaler.transform(x_validate)
        x_test_scaled = rscaler.transform(x_test)
        return x_train_scaled, x_val_scaled, x_test_scaled
    
    else:
        raise ValueError('Select a valid scaler.')
    
def handle_missing_values(df, column_percent=.6, row_percent=.6):
    '''
    Values in 'column_percent' & 'row_percent' should be a decimal between 0-1
    this will indicate how much of the values or columns will be retained based on percentage of missing values.

    The higher the decimal the lower the threshold, vice versa. It will be the inverse of what is put -- (ex. 0.8 means values that have at least 80%
    of none nulls will not be dropped.)
    '''

    col_limit = int(len(df.columns) * column_percent)
    row_limit = int(len(df.columns) * row_percent)
    df.dropna(thresh=col_limit,axis=1,inplace=True)
    df.dropna(thresh=row_limit,axis=0,inplace=True)

    return df

def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df) -> pd.DataFrame:
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')

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

def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('=======================\n=====   SHAPE   =====\n=======================')
    print(df.shape)
    print('========================\n=====   INFO   =====\n========================')
    print(df.info())
    print('========================\n=====   DESCRIBE   =====\n========================')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('==========================\n=====   DATA TYPES   =====\n==========================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    #print(df.select_dtypes('O').describe().T.to_markdown())
    print('==============================\n=====   MISSING VALUES   =====\n==============================')
    print('==========================\n=====   BY COLUMNS   =====\n==========================')
    print(missing_by_col(df).to_markdown())
    print('=======================\n=====   BY ROWS   =====\n=======================')
    print(missing_by_row(df).to_markdown())
    print('========================\n=====   OUTLIERS   =====\n========================')
    print(report_outliers(df, k=k))
    print('================================\n=====   THAT IS ALL, BYE   =====\n================================')