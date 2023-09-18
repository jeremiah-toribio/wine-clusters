import pandas as pd
import numpy as np

import os

def acquire_wine():
    '''
    function first checks os for wine_csv.csv. If it exists it is read into a pandas dataframe; Else it checks os for winequality_red.csv and winequality_white.csv. If they exists they are red into separate dataframes. Then each df is encoded with a new column 'is_red'. Concat both df's into a single df.
    '''
    if os.path.exists('wine_csv.csv'):
        df = pd.read_csv('wine_csv.csv')
    else:
        if os.path.exists('winequality_red.csv'):
            red_df = pd.read_csv('winequality_red.csv')

        if os.path.exists('winequality_white.csv'):
            white_df = pd.read_csv('winequality_white.csv')

        red_df['is_red'] = 1
        white_df['is_red'] = 0

        df = pd.concat([white_df, red_df], axis=0, ignore_index=True)
        df.to_csv('wine_csv.csv')
    df = df.drop(columns=['Unnamed: 0'])
    return df

def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df):
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