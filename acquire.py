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
    return df