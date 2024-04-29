import pandas as pd
from typing import Callable


def check_inputs(func) -> Callable:
    """
    Decorator function to validate inputs to precision_at_k & recall_at_k
    """
    def checker(df: pd.DataFrame, k: int=3, y_test: str='y_actual', y_pred: str='y_recommended') -> float:
        # check we have a valid entry for k
        if k <= 0:
            raise ValueError(f'Value of k should be greater than 1, read in as: {k}')
        # check y_test & y_pred columns are in df
        if y_test not in df.columns:
            raise ValueError(f'Input dataframe does not have a column named: {y_test}')
        if y_pred not in df.columns:
            raise ValueError(f'Input dataframe does not have a column named: {y_pred}')
        return func(df, k, y_test, y_pred)
    return checker

@check_inputs
def precision_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    """
    Function to compute precision@k for an input boolean dataframe
    
    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output
        
    Output:
        Floating-point number of precision value for k items
    """       
    # extract the k rows
    dfK = df.head(k)
    # compute number of recommended items @k
    denominator = dfK[y_pred].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator/denominator
    else:
        return None

@check_inputs
def recall_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    """
    Function to compute recall@k for an input boolean dataframe
    
    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output
        
    Output:
        Floating-point number of recall value for k items
    """    
    # extract the k rows
    dfK = df.head(k)
    # compute number of all relevant items
    denominator = df[y_test].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator/denominator
    else:
        return None