import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Union
from scipy.stats import boxcox
import requests
from bs4 import BeautifulSoup
import json
from sqlalchemy import create_engine
import pickle as pkl
from db import get_table


def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from the given path.

    Args:
    - path (str): The path to the dataset file.
    
    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(path)

def tidy_up_columns(df: pd.DataFrame)  -> pd.DataFrame:
    """
    Tidy up column names of the given dataframe by lowering the column names and removing any leading/trailing whitespaces.


    Args:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with tidied column names.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def set_index(col_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the specified column as the index of the DataFrame.

    Args:
    col_name (str): The name of the column to be set as the index.
    df (pd.DataFrame): The DataFrame to be modified.

    Returns:
    pd.DataFrame: The modified DataFrame with the specified column as the index.
    """
    
    df = df.set_index(col_name)
    return df

def get_orig_col_name(col_name: str) -> str:
    """
    Get the original column name from a column name that includes any transformations.

    Args:
    col_name (str): The column name that may include transformations.

    Returns:
    str: The original column name.
    """
    
    extra_names = ['standardized', 'cleaned', 'imputed', 'labelEncoded', 'log', 'sqrt', 'norm', 'log1p', 'boxcox']
    
    orig_col_name = col_name
    for extra_name in extra_names:
        orig_col_name = orig_col_name.replace(f'_{extra_name}', '')
    return orig_col_name

def convert_to_z_score(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert the values in a column to z-scores.

    Args:
    df (pd.DataFrame): The DataFrame containing the column to convert.
    column (str): The name of the column to convert.

    Returns:
    pd.DataFrame: The DataFrame with the column converted to z-scores.
    """
    
    df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def log_transform_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Applies a natural logarithm transformation to a specified column in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to be transformed.
    col (str): The column to be transformed.

    Returns:
    pd.DataFrame: The DataFrame with the specified column transformed by a natural logarithm.
    """

    orig_col_name = get_orig_col_name(col)
    if df[col].min() <= 0:
        df[f'{orig_col_name}_log1p'] = np.log1p(df[col])
    else:
        df[f'{orig_col_name}_log'] = np.log(df[col])
    
    return df

def sqrt_transform_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Applies a square root transformation to a specified column in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to be transformed.
    col (str): The column to be transformed.

    Returns:
    pd.DataFrame: The DataFrame with the specified column transformed by a square root.
    """

    orig_col_name = get_orig_col_name(col)
    df[f'{orig_col_name}_sqrt'] = np.sqrt(df[col])
    return df


boxcox_lambdas = pd.DataFrame(columns=['Column', 'Lambda'])

def apply_boxcox(df: pd.DataFrame, column_name: str) -> Tuple[pd.DataFrame, float]:
    """
    Applies Box-Cox transformation to a specified column to make it more normally distributed.
    
    Parameters:
    - df: DataFrame containing the column to be transformed
    - column_name: Name of the column to transform
    
    Returns:
    - Transformed column and lambda value used for the transformation
    """
    if (df[column_name] <= 0).any():
        df[column_name] = df[column_name] + abs(df[column_name].min()) + 1

    transformed_data, fitted_lambda = boxcox(df[column_name])

    df[f'{column_name}_boxcox'] = transformed_data

    global boxcox_lambdas
    orig_col_name = get_orig_col_name(column_name)
    new_row = {'Column': orig_col_name, 'Lambda': fitted_lambda}
    boxcox_lambdas = pd.concat([boxcox_lambdas, pd.DataFrame([new_row])], ignore_index=True)

    return df, fitted_lambda


def normalization(df: pd.DataFrame, col: str, type_of_norm: str) -> pd.DataFrame:
    """
    Normalize the specified column in the DataFrame using either standardization or min-max scaling.

    Args:
    df (pd.DataFrame): The input DataFrame.
    col (str): The column to normalize.
    type_of_norm (str): The type of normalization to apply. Can be either 'standardization
    or 'min-max scaling'.
    
    Returns:
    pd.DataFrame: The DataFrame with the normalized column.
    """
    
    assert type_of_norm in ['standard', 'min_max'], 'type_of_norm should be either standard or min_max'
    orig_col_name = get_orig_col_name(col)
    if type_of_norm == 'standard':
        scaler = StandardScaler()
    elif type_of_norm == 'min_max':
        scaler = MinMaxScaler()
    df[f'{orig_col_name}_norm'] = scaler.fit_transform(df[[col]])

    return df

global_lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Encoded'])
multivariate_lookup_table = pd.DataFrame(columns=['column_used_for_imputation', 'technique', 'column_imputed', 'bins_used', 'groups_with_corresponding_values'])

def encode_col(df: pd.DataFrame, type_of_encoding: str, col_name: str, need_to_sort: bool=False) -> pd.DataFrame:
    """
    This function encodes a column in a DataFrame based on the type of encoding specified. It encodes the column 
    and stores the original and encoded values in a lookup table if type_of_encoding = 'label'.

    Args:
    - df (pd.DataFrame): The DataFrame that contains the column to be encoded.
    - type_of_encoding (str): The type of encoding to be used. It can be 'ohe' or 'label'.
    - col_name (str): The name of the column to be encoded.
    - need_to_sort (bool): Whether the column needs to be sorted before encoding. Default is False.

    Returns:
    - pd.DataFrame: The DataFrame with the encoded column.
    """

    assert type_of_encoding in ['label', 'onehot'], 'Invalid type of encoding. Please choose either "label" or "onehot"'
    
    global global_lookup_table

    orig_col_name = get_orig_col_name(col_name)

    lookup_table = pd.DataFrame()
    if type_of_encoding == 'label':
        if need_to_sort:
            sorted_values = sorted(df[col_name].dropna().unique())
            df[f'{orig_col_name}_labelEncoded'] = pd.Categorical(df[col_name], categories=sorted_values, ordered=True).codes
            lookup_table['Original'] = sorted_values
        else:
            df[f'{orig_col_name}_labelEncoded'] = pd.Categorical(df[col_name]).codes
            lookup_table['Original'] = df[col_name].unique()

        lookup_table['Encoded'] = range(len(lookup_table['Original']))
        lookup_table['Column'] = orig_col_name
        global_lookup_table = pd.concat([global_lookup_table, lookup_table], ignore_index=True)
        
    elif type_of_encoding == 'onehot':
        one_hot_df = pd.get_dummies(df[col_name], prefix=orig_col_name, dtype=int)
        df = pd.concat([df, one_hot_df], axis=1)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to remove duplicates from.

    Returns:
    - pd.DataFrame: The DataFrame with duplicates removed.
    """
    
    return df.drop_duplicates()

def clean_str_in_col(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Clean the strings in the specified column by converting them to lowercase and stripping leading/trailing whitespaces.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.
    - col_name (str): The name of the column to clean.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned column.
    """
    
    df[col_name] = df[col_name].str.lower().str.strip()
    return df

# emp_titile column cleaning
def impute_emp_title(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function imputes the missing values in the 'emp_title' column of the given DataFrame
    with the value = 'unknown'.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - col_name (str): The name of the column to impute.

    Returns:
    - pd.DataFrame: The DataFrame with the missing values in the 'emp_title' column imputed
    """

    df[f'{col_name}_imputed'] = df[col_name].fillna('unknown')

    global global_lookup_table
    lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Encoded'])
    lookup_table['Column'] = [col_name]
    lookup_table['Original'] = [np.nan]
    lookup_table['Encoded'] = ['unknown']
    global_lookup_table = pd.concat([global_lookup_table, lookup_table], ignore_index=True)
    return df

def clean_emp_title(df: pd.DataFrame)  -> pd.DataFrame:
    """
    This function cleans the 'emp_title' column of the given DataFrame by cleaning the strings and imputing missing values with 'unknown'.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'emp_title' column.
    """
    df = clean_str_in_col(df, 'emp_title')
    df = impute_emp_title(df, 'emp_title')
    return df

# emp_length column cleaning
def clean_emp_length_values(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function cleans the employment lengths column in a given DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the column we want to clean.
    col_name (str): The name of the column containing the employment lengths.

    Returns:
    pd.DataFrame: The cleaned DataFrame with the emp_length corrected.
    """

    df[f'{col_name}_cleaned'] = df[col_name].str.replace('years', '').str.replace('year', '').str.replace('< 1', '0.5').str.replace('10+', '11').str.strip()
    df[f'{col_name}_cleaned'] = df[f'{col_name}_cleaned'].astype(float)
    return df

def impute_emp_length_by_income_bin(df: pd.DataFrame, bins: int=5, col_name: str='emp_length_cleaned', col_name_2: str='annual_inc') -> pd.DataFrame:
    """
    Imputes 'Emp Length' based on the mode within equal-frequency bins of 'Annual Inc'.
    
    Parameters:
    - data: DataFrame containing 'Emp Length' and 'Annual Inc' columns.
    - bins: Number of equal-frequency bins to create for 'Annual Inc'.
    - col_name: Name of the 'Emp Length' column.
    - col_name_2: Name of the 'Annual Inc' column.
    
    Returns:
    - DataFrame with imputed 'Emp Length' column.
    """
    
    df[f'{col_name_2}_bins'], bin_edges = pd.qcut(df[col_name_2], q=bins, labels=False, retbins=True)
    
    income_bin_mode_emp_length = df.groupby(f'{col_name_2}_bins')[col_name].agg(lambda x: x.mode().iloc[0])
    
    orig_col_name = get_orig_col_name(col_name)
    df[f'{orig_col_name}_imputed'] = df.apply(
        lambda row: income_bin_mode_emp_length[row[f'{col_name_2}_bins']] if pd.isna(row[col_name]) else row[col_name],
        axis=1
    )
    print(f'income_bin_mode_emp_length: {income_bin_mode_emp_length.to_dict()}')
    print(f'bin_edges: {bin_edges.tolist()}')
    global multivariate_lookup_table
    modes_list = income_bin_mode_emp_length.tolist()
    bin_edges_with_mode_values = [(f'{bin_edges[i-1]}-{bin_edges[i]}', modes_list[i - 1]) for i in range(1,len(bin_edges))]
    print(f'bin_edges_with_mode_values: {bin_edges_with_mode_values}')
    lookup_table = pd.DataFrame(columns=['column_used_for_imputation', 'technique', 'column_imputed', 'bins_used', 'groups_with_corresponding_values'])
    lookup_table['column_used_for_imputation'] = [col_name_2]
    lookup_table['technique'] = ['mode']
    lookup_table['column_imputed'] = [col_name]
    lookup_table['bins_used'] = [json.dumps(bin_edges.tolist())]
    lookup_table['groups_with_corresponding_values'] = [json.dumps(bin_edges_with_mode_values)]
    multivariate_lookup_table = pd.concat([multivariate_lookup_table, lookup_table], ignore_index=True)
    
    return df

def clean_emp_length(df: pd.DataFrame, bins: int=5, col_name: str='emp_length_cleaned', col_name_2: str='annual_inc') -> pd.DataFrame:
    """
    This function cleans the 'emp_length' column of the given DataFrame by cleaning the strings and 
    imputing missing values with the mode of each of the col_name_2 binned column.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - bins: Number of equal-frequency bins to create for 'Annual Inc'.
    - col_name: Name of the 'Emp Length' column.
    - col_name_2: Name of the 'Annual Inc' column.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'emp_length' column.
    """
    df = clean_emp_length_values(df, 'emp_length')
    df = impute_emp_length_by_income_bin(df, bins, col_name, col_name_2)
    return df

# home_ownership column cleaning
def clean_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'home_ownership' column of the given DataFrame by one hot encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'home_ownership' column.
    """
    df = encode_col(df, 'onehot', 'home_ownership')
    return df

# annual_inc column cleaning
def handle_outliers_of_annual_inc(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'annual_inc' column in the given DataFrame by applying log transform to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'annual_inc' column handled.
    """

    df = log_transform_col(df, 'annual_inc')
    return df

def clean_annual_inc(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'annual_inc' column of the given DataFrame by handling the outliers.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'annual_inc' column.
    """
    df = handle_outliers_of_annual_inc(df)
    return df

# annual_inc_joint column cleaning
def handle_outliers_of_annual_inc_joint(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'annual_inc_joint' column in the given DataFrame by applying log transform to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'annual_inc_joint' column handled.
    """

    df = log_transform_col(df, 'annual_inc_joint')
    return df

def impute_annual_inc_joint(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function imputes the missing values in the 'annual_inc_joint' column of the given DataFrame with value = 0.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'annual_inc_joint' column.
    - col_name (str): The name of the column to be imputed.

    Returns:
    pd.DataFrame: The DataFrame with the missing values in the 'annual_inc_joint' column imputed.
    """

    orig_col_name = get_orig_col_name(col_name)
    df[f'{orig_col_name}_imputed'] = df[col_name].fillna(0)

    global global_lookup_table
    lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Encoded'])
    lookup_table['Column'] = [orig_col_name]
    lookup_table['Original'] = [np.nan]
    lookup_table['Encoded'] = [0]
    global_lookup_table = pd.concat([global_lookup_table, lookup_table], ignore_index=True)
        
    return df

def clean_annual_inc_joint(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'annual_inc_joint' column in the given DataFrame by handling outliers and imputing missing values.

    Args:
    - df (pd.DataFrame): The DataFrame with the 'annual_inc_joint' column to be cleaned

    Returns:
    - pd.DataFrame: The DataFrame with the 'annual_inc_joint' column cleaned
    """

    df = handle_outliers_of_annual_inc_joint(df)
    df = impute_annual_inc_joint(df, 'annual_inc_joint_log')
    return df

# verification_status column cleaning
def clean_verification_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'verification_status' column of the given DataFrame by one hot encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'verification_status' column.
    """
    df = encode_col(df, 'onehot', 'verification_status')
    return df

# zip_code column cleaning
# nothing to do here


# addr_state column cleaning
def clean_addr_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'addr_state' column of the given DataFrame by one hot encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'addr_state' column.
    """
    df = encode_col(df, 'label', 'addr_state', need_to_sort=True)
    return df

# avg_cur_bal column cleaning
def handle_outliers_of_avg_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'avg_cur_bal' column in the given DataFrame by applying box-cox transformation to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'avg_cur_bal' column handled.
    """

    df, _ = apply_boxcox(df, 'avg_cur_bal')
    return df

def clean_avg_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'avg_cur_bal' column of the given DataFrame by handling outliers and applying box-cox transformation.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'avg_cur_bal' column.
    """
    df = handle_outliers_of_avg_cur_bal(df)
    return df

# tot_cur_bal column cleaning
def handle_outliers_of_tot_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'tot_cur_bal' column in the given DataFrame by applying box-cox transformation to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'tot_cur_bal' column handled.
    """

    df, _ = apply_boxcox(df, 'tot_cur_bal')
    return df

def clean_tot_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'tot_cur_bal' column of the given DataFrame by handling outliers and applying box-cox transformation.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'tot_cur_bal' column.
    """
    df = handle_outliers_of_tot_cur_bal(df)
    return df

# loan_id column cleaning
# nothing to do here

# loan_status column cleaning
def clean_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'loan_status' column of the given DataFrame by one hot encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'loan_status' column.
    """
    df = encode_col(df, 'onehot', 'loan_status')
    return df

# loan_amount column cleaning
def handle_outliers_of_loan_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'loan_amount' column in the given DataFrame by applying sqrt transform to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'loan_amount' column handled.
    """

    df = sqrt_transform_col(df, 'loan_amount')
    return df

def clean_loan_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'loan_amount' column of the given DataFrame by handling outliers.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'loan_amount' column.
    """
    df = handle_outliers_of_loan_amount(df)
    return df

# state column cleaning
def clean_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'state' column of the given DataFrame by label encoding the column values.
    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'state' column.
    """
    df = encode_col(df, 'label', 'state', need_to_sort=True)
    return df

# funded_amount column cleaning
def handle_outliers_of_funded_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'funded_amount' column in the given DataFrame by applying sqrt transform to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'funded_amount' column handled.
    """

    df = sqrt_transform_col(df, 'funded_amount')
    return df

def clean_funded_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'funded_amount' column of the given DataFrame by handling outliers.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'funded_amount' column.
    """
    df = handle_outliers_of_funded_amount(df)
    return df

# term column cleaning
def clean_term(df: pd.DataFrame, col_name: str='term') -> pd.DataFrame:
    """
    This function cleans the term column in a given DataFrame by removing any non-alphanumeric characters and converting it to an integer.

    Args:
    - df (pd.DataFrame): The DataFrame containing the term column to be cleaned.
    - col_name (str): The name of the term column to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame with the term column converted to an integer.
    """

    df[f'{col_name}_cleaned'] = df[col_name].str.replace('months', '').str.strip()
    df[f'{col_name}_cleaned'] = df[f'{col_name}_cleaned'].astype(int)
    return df

# int_rate column cleaning
def handle_outliers_of_int_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles the outliers of the 'int_rate' column in the given DataFrame by applying log transform to the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the outliers of 'int_rate' column handled.
    """

    df = log_transform_col(df, 'int_rate')
    return df

def impute_int_rate(df: pd.DataFrame, col_name: str, col_used_for_imputation: str) -> pd.DataFrame:
    """
    Impute the missing values in the 'int_rate' column based on the mean interest rate for each grade.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the 'int_rate' and 'grade'
    - col_name (str): The name of the column to be imputed
    - col_used_for_imputation (str): The column used for imputation
    
    Returns:
    - pd.DataFrame: The DataFrame with imputed 'int_rate' column
    """
    orig_name = get_orig_col_name(col_name)
    grade_mean_int_rate = df.groupby(col_used_for_imputation)[col_name].mean()

    df[f'{orig_name}_imputed'] = df.apply(
        lambda row: grade_mean_int_rate[row[col_used_for_imputation]] if pd.isna(row[col_name]) else row[col_name],
        axis=1
    )

    global multivariate_lookup_table
    grade_mean_int_rate_dict = grade_mean_int_rate.to_dict()
    grade_mean_int_rate_tuples = list(grade_mean_int_rate_dict.items())
    print(f'grade_mean_int_rate_tuples: {grade_mean_int_rate_tuples}')
    lookup_table = pd.DataFrame(columns=['column_used_for_imputation', 'technique', 'column_imputed', 'bins_used'])
    lookup_table['column_used_for_imputation'] = [col_used_for_imputation]
    lookup_table['technique'] = ['mean']
    lookup_table['column_imputed'] = [col_name]
    lookup_table['bins_used'] = [None]
    lookup_table['groups_with_corresponding_values'] = [json.dumps(grade_mean_int_rate_tuples)]
    multivariate_lookup_table = pd.concat([multivariate_lookup_table, lookup_table], ignore_index=True)

    return df

def clean_int_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'int_rate' column by applying log transform and imputing missing values
    based on the mean interest rate for each grade.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the 'int_rate' and 'grade' columns
    - col_name (str): The name of the column to be cleaned
    - col_used_for_imputation (str): The column used for imputation

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'int_rate' column
    """
    df = handle_outliers_of_int_rate(df)
    df = impute_int_rate(df, 'int_rate_log', 'grade')
    return df

# grade column cleaning
# nothing to do here

# issue_date  column cleaning
def clean_issue_date(df: pd.DataFrame, col_name: str='issue_date') -> pd.DataFrame:
    """
    Clean the issue date column in the given DataFrame by converting it to a datetime object.

    Args:
    - df (pd.DataFrame): The DataFrame containing the issue date column.
    - col_name (str): The name of the issue date column.

    Returns:
    pd.DataFrame: The DataFrame with the issue date column cleaned.
    """
    
    df[f'{col_name}_cleaned'] = pd.to_datetime(df[col_name])
    return df

# pymnt_plan column cleaning
def clean_pymnt_plan(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'pymnt_plan' column of the given DataFrame by label encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'pymnt_plan' column.
    """
    df = encode_col(df, 'label', 'pymnt_plan')
    return df

# type column cleaning

def standardize_loan_type(df: pd.DataFrame, type_col: str) -> pd.DataFrame:
    """
    Standardize loan type by mapping it to a correct value which has no inconsistencies.

    Args:
    - df (pd.DataFrame): The DataFrame containing the loan type column.
    - type_col (str): The name of the column containing the loan type.

    Returns:
    pd.DataFrame: The DataFrame with the loan type standardized.
    """

    type_mapping = {
        'INDIVIDUAL': 'Individual',
        'Individual': 'Individual',
        'JOINT': 'Joint',
        'Joint App': 'Joint',
        'DIRECT_PAY': 'Direct_Pay'
    }

    df[f'{type_col}_standardized'] = df[type_col].replace(type_mapping)
    
    return df

def clean_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'type' column of the given DataFrame by standardizing the column values and one hot encoding them.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'type' column.
    """
    df = standardize_loan_type(df, 'type')
    df = encode_col(df, 'onehot', 'type_standardized')
    return df

# purpose column cleaning
def clean_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'purpose' column of the given DataFrame by one hot encoding the column values.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'purpose' column.
    """
    df = encode_col(df, 'onehot', 'purpose')
    return df

# description column cleaning
def impute_description(df: pd.DataFrame, desc_col: str) -> pd.DataFrame:
    """
    This function imputes missing values in the description column of a given DataFrame with the value = 'No Description'.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - desc_col (str): The name of the column that contains the description.

    Returns:
    pd.DataFrame: The input DataFrame with missing values in the description column imputed.
    """

    df[f'{desc_col}_imputed'] = df[desc_col].fillna('No Description')
    global global_lookup_table
    lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Encoded'])
    lookup_table['Column'] = [desc_col]
    lookup_table['Original'] = [np.nan]
    lookup_table['Encoded'] = ['No Description']
    global_lookup_table = pd.concat([global_lookup_table, lookup_table], ignore_index=True)
    return df

def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'description' column of the given DataFrame by imputing missing values with 'No Description'.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'description' column.
    """
    df = impute_description(df, 'description')
    return df

# add 4 columns to the dataset
# month number
def add_month_col(df: pd.DataFrame, date_col_name: str='issue_date_cleaned',  month_col_name: str='issue_month') -> pd.DataFrame:
    """
    This function adds a new column to a given DataFrame with the month of the date in the specified
    date column.

    Args:
    - df (pd.DataFrame): The DataFrame to which the new column will be added.
    - date_col_name (str): The name of the column containing the date.
    - month_col_name (str): The name of the new column that will contain the month of the date.

    Returns:
    - pd.DataFrame: The DataFrame with the new column added.
    """
    
    df[month_col_name] = df[date_col_name].dt.month
    return df

# can salary cover

def can_salary_cover(df: pd.DataFrame, 
                     annual_income_col: str='annual_inc', 
                     loan_amount_col: str='loan_amount', 
                     annual_joint_col: str='annual_inc_joint',
                     type_col: str='type_standardized') -> pd.DataFrame:
    """
    This function checks if the annual income can cover the loan amount based on the loan type.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - annual_income_col (str): The column name for the annual income.
    - loan_amount_col (str): The column name for the loan amount.
    - annual_joint_col (str): The column name for the annual joint income.
    - type_col (str): The column name for the loan type.

    Returns:
    - pd.DataFrame: The input DataFrame with an additional column 'salary_can_cover' indicating whether the annual
    income can cover the loan amount.
    """

    def salary_cover(row: pd.Series) -> bool:
        """
        This function checks if the annual income can cover the loan amount.

        Args:
        - row (pd.Series): The row containing the annual income and loan amount.

        Returns:
        - bool: True if the annual income can cover the loan amount, False otherwise.
        """

        loan_type = row[type_col]
        loan_amount = row[loan_amount_col]
        
        if loan_type == "Joint":
            return row[annual_joint_col] >= loan_amount
        elif loan_type == "Individual" or loan_type == "Direct_Pay":
            return row[annual_income_col] >= loan_amount
        else:
            return False

    df['salary_can_cover'] = df.apply(salary_cover, axis=1).astype(int)
    
    return df

# letter grade
def change_number_to_grade(df: pd.DataFrame, col_name: str='grade') -> pd.DataFrame:
    """
    This function converts a numerical column to a letter grade column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the numerical column to be converted.
    - col_name (str): The name of the numerical column to be converted.

    Returns:
    pd.DataFrame: The DataFrame with the numerical column converted to a letter grade column.
    """

    grade_mapping = {
        'A': range(1,6),
        'B': range(6,11),
        'C': range(11,16),
        'D': range(16,21),
        'E': range(21,26),
        'F': range(26,31),
        'G': range(31,36)
    }
    def get_grade(x: int) -> str:
        for k, v in grade_mapping.items():
            if x in v:
                return k

    df['letter_grade'] = df[col_name].apply(lambda x: get_grade(x))
    global global_lookup_table
    lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Encoded'])
    rows = [{'Column': 'grade', 'Original': num, 'Encoded': k} for k, v in grade_mapping.items() for num in v]
    lookup_table = pd.concat([lookup_table, pd.DataFrame(rows)], ignore_index=True)
    global_lookup_table = pd.concat([global_lookup_table, lookup_table], ignore_index=True)

    return df

def add_letter_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds a new column to a given DataFrame with the letter grade based on the grade column.

    Args:
    - df (pd.DataFrame): The DataFrame to which the new column will be added.

    Returns:
    - pd.DataFrame: The DataFrame with the new column added.
    """
    
    df = change_number_to_grade(df, 'grade')
    df = encode_col(df, 'label', 'letter_grade', need_to_sort=True)
    return df

# installment per month
def calculate_installment_per_month(
        df: pd.DataFrame, 
        loan_amount_col: str='loan_amount', 
        term_col: str='term_cleaned', 
        int_rate_col_imputed: str='int_rate_imputed') -> pd.DataFrame:
    """
    This function calculates the monthly installment amount for each loan based on the loan amount, term, and interest rate.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the loan data.
    - loan_amount_col (str): The column name for the loan amount.
    - term_col (str): The column name for the loan term.
    - int_rate_col_imputed (str): The column name for the imputed interest rate.

    Returns:
    - pd.DataFrame: The input DataFrame with an additional column 'installment_per_month' containing the monthly installment amount.
    """
    
    r = np.exp(df[int_rate_col_imputed])/12
    n = df[term_col]
    P = df[loan_amount_col]
    numerator = r * (1+r)**n
    denominator = (1+r)**n - 1
    df['installment_per_month'] = P * (numerator/denominator)
    return df

# normalization
def normalize_columns(df: pd.DataFrame, numeric_cols: List[str], norm_type: str='minmax', is_fit: bool=False) -> pd.DataFrame:
    """
    Normalizes specified numeric columns in the dataframe using Min-Max or Z-score scaling.
    
    Parameters:
    - df: DataFrame to be normalized
    - numeric_cols: List of column names to be normalized
    - norm_type: Type of normalization ('minmax' or 'z-score')
    - is_fit: If False, fit the scaler to the data. If True, use the previously fitted scaler.
    
    Returns:
    - DataFrame with normalized columns
    """
    assert norm_type in ['minmax', 'z-score'], 'Invalid normalization type. Choose either "minmax" or "z-score".'
    
    if norm_type == 'minmax':
        scaler = MinMaxScaler()
    else: 
        scaler = StandardScaler()

    if not is_fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        with open('./data/cleaned_dataset/scaler.pkl', 'wb') as f:
            pkl.dump(scaler, f)
    else:
        with open('./data/cleaned_dataset/scaler.pkl', 'rb') as f:
            scaler = pkl.load(f)
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def normalize_numeric_cols(df: pd.DataFrame, norm_type: str='minmax', is_fit: bool=False) -> pd.DataFrame:
    """
    Normalizes all numeric columns in the dataframe using Min-Max or Z-score scaling.

    Args:
    - df (pd.DataFrame): The input DataFrame.
    - norm_type (str): The type of normalization. Defaults to 'minmax'.
    - is_fit (bool): If False, fit the scaler to the data. If True, use the previously fitted scaler.

    Returns:
    - pd.DataFrame: The input DataFrame with normalized numeric columns.
    """

    numeric_cols = ['annual_inc_log', 'annual_inc_joint_imputed', 'avg_cur_bal_boxcox', 'tot_cur_bal_boxcox','loan_amount_sqrt','funded_amount_sqrt', 'int_rate_imputed', 'installment_per_month']
    df = normalize_columns(df, numeric_cols, norm_type=norm_type, is_fit=is_fit)
    return df

# bonus task
def get_state_names(url: str) -> dict:
    """
    This function takes a URL as input, scrapes the webpage, and returns a dictionary of state abbreviations and names.

    Args:
    url (str): The URL of the webpage to scrape.

    Returns:
    dict: A dictionary where the keys are state abbreviations and the values are state names.
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.find_all('tr')[1:] 
    state_dict = {}
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            state_name = cols[0].text.strip()
            state_abbreviation = cols[2].text.strip()
            state_dict[state_abbreviation] = state_name
    return state_dict

def add_state_names(df: pd.DataFrame, state_col: str, url: str) -> pd.DataFrame:
    """
    This function adds a new column to the DataFrame containing the state names based on the state abbreviations.

    Args:
    df (pd.DataFrame): The DataFrame to add the state names to.
    state_col (str): The column name containing the state abbreviations.
    url (str): The URL of the webpage to scrape.

    Returns:
    pd.DataFrame: The DataFrame with the added state names column.
    """

    state_dict = get_state_names(url)
    df['state_name'] = df[state_col].map(state_dict)
    return df

def do_bonus_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function performs the bonus task by adding a new column to the DataFrame containing the state names based
    on the state abbreviations.

    Args:
    - df (pd.DataFrame): The DataFrame to add the state names to.

    Returns:
    - pd.DataFrame: The DataFrame with the added state names column.
    """

    url = "https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971"
    df = add_state_names(df, 'state', url)
    return df

# keep wanted cols
def keep_wanted_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function keeps only the wanted columns in the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to keep only the wanted columns from.

    Returns:
    - pd.DataFrame: The DataFrame with only the wanted columns.
    """

    cols_to_keep = [
        'emp_title_imputed', 
        'emp_length_imputed',
        'home_ownership_ANY',
        'home_ownership_MORTGAGE',
        'home_ownership_OWN',
        'home_ownership_RENT',
        'annual_inc_log',
        'annual_inc_joint_imputed',
        'verification_status_Not Verified',
        'verification_status_Source Verified',
        'verification_status_Verified',
        'zip_code',
        'addr_state_labelEncoded',
        'avg_cur_bal_boxcox',
        'tot_cur_bal_boxcox',
        'loan_id',
        'loan_status_Charged Off',
        'loan_status_Current',
        'loan_status_Default',
        'loan_status_Fully Paid',
        'loan_status_In Grace Period',
        'loan_status_Late (16-30 days)',
        'loan_status_Late (31-120 days)',
        'loan_amount_sqrt',
        'state_labelEncoded',
        'funded_amount_sqrt',
        'term_cleaned',
        'int_rate_imputed',
        'grade',
        'issue_date_cleaned',
        'pymnt_plan_labelEncoded',
        'type_Direct_Pay',
        'type_Individual',
        'type_Joint',
        'purpose_car',
        'purpose_credit_card',
        'purpose_debt_consolidation',
        'purpose_home_improvement',
        'purpose_house',
        'purpose_major_purchase',
        'purpose_medical',
        'purpose_moving',
        'purpose_other',
        'purpose_renewable_energy',
        'purpose_small_business',
        'purpose_vacation',
        'purpose_wedding',
        'description_imputed',
        'issue_month',
        'salary_can_cover',
        'letter_grade_labelEncoded',
        'installment_per_month',
        'state_name'
    ]

    df = df[cols_to_keep]
    return df

# rename columns
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in a pandas DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to rename columns in.
    
    Returns:
    - pd.DataFrame: The DataFrame with renamed columns.
    """
    
    mapping_dict = {}
    for col in df.columns:
        orig_col_name = get_orig_col_name(col)
        mapping_dict[col] = orig_col_name
        # print(f'{col} => {orig_col_name}')
    df = df.rename(columns=mapping_dict)
    return df

def clean_data(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the data by removing rows with missing values and renaming columns.

    Args:
    - dataset_path (str): The path to the dataset to clean.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the cleaned DataFrame and the global lookup table.
    """
    # Load the data
    df = load_data(dataset_path)

    # tidy up the column names
    df = tidy_up_columns(df)

    # set the index
    df = set_index(df=df, col_name='customer_id')

    # Remove duplicates
    df = drop_duplicates(df)

    # Clean 'emp_title' column
    df = clean_emp_title(df)

    # Clean 'emp_length' column
    df = clean_emp_length(df)

    # Clean 'home_ownership' column
    df = clean_home_ownership(df)

    # Clean 'annual_inc' column
    df = clean_annual_inc(df)

    # Clean 'annual_inc_joint' column
    df = clean_annual_inc_joint(df)

    # Clean 'verification_status' column
    df = clean_verification_status(df)

    # Clean 'addr_state' column
    df = clean_addr_state(df)

    # Clean 'avg_cur_bal' column
    df = clean_avg_cur_bal(df)

    # Clean 'tot_cur_bal' column
    df = clean_tot_cur_bal(df)

    # Clean 'loan_status' column
    df = clean_loan_status(df)

    # Clean 'loan_amount' column
    df = clean_loan_amount(df)

    # Clean 'state' column
    df = clean_state(df)

    # Clean 'funded_amount' column
    df = clean_funded_amount(df)

    # Clean 'term' column
    df = clean_term(df)

    # Clean 'int_rate' column
    df = clean_int_rate(df)

    # clean 'issue_date' column
    df = clean_issue_date(df)

    # Clean 'pymnt_plan' column
    df = clean_pymnt_plan(df)

    # Clean 'type' column
    df = clean_type(df)

    # Clean 'purpose' column
    df = clean_purpose(df)

    # Clean 'description' column
    df = clean_description(df)

    # Add month column
    df = add_month_col(df)

    # Check if salary can cover loan amount
    df = can_salary_cover(df)

    # Add letter grade column
    df = add_letter_grade(df)

    # Calculate installment per month
    df = calculate_installment_per_month(df)
    
    # normalize numeric columns
    df = normalize_numeric_cols(df)

    # Bonus task
    df = do_bonus_task(df)

    # keep  only the columns we need
    df = keep_wanted_cols(df)

    # Rename columns
    df = rename_columns(df)

    # Save the cleaned dataset
    cleaned_dir_path = './data/cleaned_dataset'
    df.to_csv(f'{cleaned_dir_path}/fintech_data_MET_P2_52_1008_clean.csv')
    global_lookup_table.to_csv(f'{cleaned_dir_path}/lookup_table_MET_P2_52_1008.csv', index=False)
    multivariate_lookup_table.to_csv(f'{cleaned_dir_path}/multivariate_lookup_table_MET_P2_52_1008.csv', index=False)
    boxcox_lambdas.to_csv(f'{cleaned_dir_path}/boxcox_lambdas_MET_P2_52_1008.csv', index=False)

    return df, global_lookup_table, multivariate_lookup_table, boxcox_lambdas

##----------------------------------------------------------------------------
## Processing Messages
def apply_boxcox_for_processing(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Apply box-cox transformation to a column in the dataframe for processing.

    Args:
    - df (pd.DataFrame): Dataframe to process.
    - col_name (str): Name of the column to apply box-cox transformation.

    Returns:
    - pd.DataFrame: Dataframe with the column transformed.
    """
    boxcox_lambdas_df = get_table('boxcox_lambdas_MET_P2_52_1008')
    orig_col_name = get_orig_col_name(col_name)
    boxcox_lambda = boxcox_lambdas_df[boxcox_lambdas_df['Column'] == orig_col_name]['Lambda'].values[0]
    df[f'{orig_col_name}_boxcox'] = boxcox(df[col_name], boxcox_lambda)
    return df

def encode_cols_for_processing(df: pd.DataFrame, col_name: str, encoding_type: str) -> pd.DataFrame:
    """
    Encode a column in the dataframe for processing.

    Args:
    - df (pd.DataFrame): Dataframe to process.
    - col_name (str): Name of the column to encode.
    - encoding_type (str): Type of encoding to apply.

    Returns:
    - pd.DataFrame: Dataframe with the column encoded.
    """
    assert encoding_type in ['label', 'onehot'], "Invalid encoding type"
    if encoding_type == 'onehot':
        cleaned_df = get_table('fintech_data_MET_P2_52_1008_clean')
        orig_col_name = get_orig_col_name(col_name)
        col_names = [col for col in cleaned_df.columns if orig_col_name in col]
        # print(col_names)
        df = encode_col(df, encoding_type, col_name)
        for col in col_names:
            if col not in df.columns:
                df[col] = 0
    elif encoding_type == 'label':
        lookup_df = get_table('lookup_table_MET_P2_52_1008')
        label_values_df = lookup_df[lookup_df['Column'] == col_name]
        # print(label_values_df)
        def apply_label_encoding(col_value: Union[str,bool]) -> int:
            # print(col_value)
            # print(label_values_df[label_values_df['Original'] == str(col_value)]['Encoded'])
            return label_values_df[label_values_df['Original'] == str(col_value)]['Encoded'].values[0]
        
        df[f'{col_name}_labelEncoded'] = df[col_name].apply(apply_label_encoding)

    return df

# process emp_length
def impute_emp_length_for_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the 'emp_length' column based on the mode of each income bin.

    Args:
    - df (pd.DataFrame): DataFrame containing the 'emp_length' column.

    Returns:
    - pd.DataFrame: DataFrame with imputed 'emp_length' values.
    """
    multivariate_df = get_table('multivariate_lookup_table_MET_P2_52_1008')
    multivariate_df = multivariate_df[multivariate_df['column_imputed'] == 'emp_length_cleaned']
    bin_edges_with_mode_values = json.loads(multivariate_df['groups_with_corresponding_values'].values[0])

    bins = []
    mode_values = []
    for bin_range, mode in bin_edges_with_mode_values:
        lower, upper = map(float, bin_range.split('-'))
        bins.append((lower, upper))
        mode_values.append(mode)

    def get_mode_for_income(income: float) -> int:
        for i, (lower, upper) in enumerate(bins):
            if lower <= income <= upper:
                return mode_values[i]

    df['emp_length_imputed'] = df.apply(
        lambda row: get_mode_for_income(row['annual_inc']) if pd.isna(row['emp_length_cleaned']) else row['emp_length_cleaned'],
        axis=1
    )
    return df

def process_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'emp_length' column by cleaning the strings and imputing missing values with the mode of each income bin.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'emp_length' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'emp_length' column.
    """
    df = clean_emp_length_values(df, col_name='emp_length')
    df = impute_emp_length_for_processing(df)
    return df

# process home_ownership
def process_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'home_ownership' column by mapping the values to a more meaningful representation.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'home_ownership' column to be processed

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'home_ownership' column
    """
    df = encode_cols_for_processing(df, 'home_ownership', 'onehot')
    return df

# process annual_inc_joint

def process_annual_inc_joint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'annual_inc_joint' column by cleaning the strings and imputing missing values.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'annual_inc_joint' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'annual_inc_joint' column.
    """
    df = clean_annual_inc_joint(df)
    lookup_df = get_table('lookup_table_MET_P2_52_1008')
    impute_value = lookup_df[lookup_df['Column'] == 'annual_inc_joint']['Encoded'].values[0]
    df['annual_inc_joint_imputed'] = df['annual_inc_joint_log'].fillna(impute_value)
    return df 

# process verification_status
def process_verification_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'verification_status' column by mapping the values to a more meaningful representation.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'verification_status' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'verification_status' column.
    """

    df = encode_cols_for_processing(df, 'verification_status', 'onehot')
    return df

# process addr_state
def process_addr_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'addr_state' column by mapping the values to a more meaningful representation.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'addr_state' column to be processed.
    
    Returns:
    - pd.DataFrame: The DataFrame with the processed 'addr_state' column.
    """

    df = encode_cols_for_processing(df, 'addr_state', 'label')
    return df

# process avg_cur_bal 
def process_avg_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    applying boxcox using the fitted lambda value.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'avg_cur_bal' column to be processed

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'avg_cur_bal' column.
    """
    df = apply_boxcox_for_processing(df, 'avg_cur_bal')
    return df

# process tot_cur_bal
def process_tot_cur_bal(df: pd.DataFrame) -> pd.DataFrame:
    """
    applying boxcox using the fitted lambda value.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing the 'tot_cur_bal' column to be processed

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'tot_cur_bal' column.
    """
    df = apply_boxcox_for_processing(df, 'tot_cur_bal')
    return df

# process loan_status
def process_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'loan_status' column by mapping the values to a more meaningful representation.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'loan_status' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'loan_status' column.
    """
    df = encode_cols_for_processing(df, 'loan_status', 'onehot')
    return df

# process state column
def process_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'state' column by mapping the values to a more meaningful representation.
    
    Args:
    - df (pd.DataFrame): The DataFrame containing the 'state' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'state' column.
    """

    df = encode_cols_for_processing(df, 'state', 'label')
    return df

# process int_rate
def impute_int_rate_for_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the 'int_rate' column with the mean value of corresponding grade value.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'int_rate' column.

    Returns:
    - pd.DataFrame: The DataFrame with imputed 'int_rate' values.
    """
    multivariate_df = get_table('multivariate_lookup_table_MET_P2_52_1008')
    multivariate_df = multivariate_df[multivariate_df['column_imputed'] == 'int_rate_log']
    grade_mean_int_rate = json.loads(multivariate_df['groups_with_corresponding_values'].values[0])

    def get_mean_for_grade(grade: int) -> float:
        for k, v in grade_mean_int_rate:
            if k == grade:
                return v

    df['int_rate_imputed'] = df.apply(
        lambda row: get_mean_for_grade(row['grade']) if pd.isna(row['int_rate_log']) else row['int_rate_log'],
        axis=1
    )

    return df


def process_int_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'int_rate' column by converting it to a numeric type and scaling it to a
    more meaningful range.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'int_rate' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'int_rate' column.
    """

    df = handle_outliers_of_int_rate(df)
    df = impute_int_rate_for_processing(df)
    return df

# process pymnt_plan
def process_pymnt_plan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'pymnt_plan' column by mapping the values to a more meaningful representation.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'pymnt_plan' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'pymnt_plan' column.
    """
    df = encode_cols_for_processing(df, 'pymnt_plan', 'label')
    return df

# process type
def process_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the 'type' column of the given DataFrame by standardizing the column values and one hot encoding them.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'type' column.
    """
    df = standardize_loan_type(df, 'type')
    df = encode_cols_for_processing(df, 'type_standardized', 'onehot')
    return df

# process purpose
def process_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the 'purpose' column of the given DataFrame by one hot encoding them.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the cleaned 'purpose' column.
    """
    df = encode_cols_for_processing(df, 'purpose', 'onehot')
    return df

# process description
def process_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'description' column by imputing missing values.

    Args:
    - df (pd.DataFrame): The DataFrame containing the 'description' column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed 'description' column.
    """
    lookup_df = get_table('lookup_table_MET_P2_52_1008')
    impute_value = lookup_df[lookup_df['Column'] == 'description']['Encoded'].values[0]
    df['description_imputed'] = df['description'].fillna(impute_value)
    return df 

def process_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the messages column by removing any non-alphanumeric characters and converting it to lowercase.

    Args:
    - df (pd.DataFrame): The DataFrame containing the messages column to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with the processed messages column.
    """
    # tidy up the column names
    processed_df = tidy_up_columns(df)

    # set the index
    processed_df = set_index(df=processed_df, col_name='customer_id')

    # Remove duplicates
    processed_df = drop_duplicates(processed_df)

    # Clean 'emp_title' column
    processed_df = clean_emp_title(processed_df)

    # Clean 'emp_length' column
    processed_df = process_emp_length(processed_df)

    # Clean 'home_ownership' column
    processed_df = process_home_ownership(processed_df)

    # clean 'annual_inc' column
    processed_df = clean_annual_inc(processed_df)

    # clean 'annual_inc_joint' column
    processed_df = process_annual_inc_joint(processed_df)
    
    # clean 'verification_status' column
    processed_df = process_verification_status(processed_df)

    # clean 'addr_state' column
    processed_df = process_addr_state(processed_df)

    # process 'avg_cur_bal' column
    processed_df = process_avg_cur_bal(processed_df)

    # process 'tot_cur_bal' column
    processed_df = process_tot_cur_bal(processed_df)

    # process 'loan_status' column
    processed_df = process_loan_status(processed_df)

    # process 'loan_amount' column
    processed_df = clean_loan_amount(processed_df)

    # process 'state' column
    processed_df = process_state(processed_df)

    # process 'funded_amount' column
    processed_df = clean_funded_amount(processed_df)

    # process 'term' column
    processed_df = clean_term(processed_df)

    # process 'int_rate' column
    processed_df = process_int_rate(processed_df)

    # process 'issue_date' column
    processed_df = clean_issue_date(processed_df)

    # process 'pymnt_plan' column
    processed_df = process_pymnt_plan(processed_df)

    # process 'type' column
    processed_df = process_type(processed_df)

    # process 'purpose' column
    processed_df = process_purpose(processed_df)

    # process 'description' column
    processed_df = process_description(processed_df)

    # Add month column
    processed_df = add_month_col(processed_df)

    # Add salary can cover loan amount
    processed_df = can_salary_cover(processed_df)

    # Add letter grade column
    processed_df = add_letter_grade(processed_df)

    # Calculate installment per month
    processed_df = calculate_installment_per_month(processed_df)

    # normalize numeric columns
    processed_df = normalize_numeric_cols(processed_df, is_fit=True)

    # Bonus task
    processed_df = do_bonus_task(processed_df)

    # keep  only the columns we need
    processed_df = keep_wanted_cols(processed_df)

    # Rename columns
    processed_df = rename_columns(processed_df)

    return processed_df

# if __name__ == "__main__":

#     dataset_path = './data/dataset/fintech_data_29_52_1008.csv'
#     _ = clean_data(dataset_path=dataset_path)
