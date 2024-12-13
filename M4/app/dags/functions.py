from cleaning import *
from sqlalchemy import create_engine


engine = create_engine('postgresql://root:root@pgdatabase:5432/m4_etl')


def extract_clean(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the data by removing rows with missing values and handling outliers and encoding categorical variables.

    Args:
    - dataset_path (str): The path to the dataset to clean.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the cleaned DataFrame and the global lookup table.
    """
    # Load the data
    df = pd.read_parquet(dataset_path)

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

    df.to_parquet('./data/fintech_clean.parquet')

    return df, global_lookup_table

def transform(cleaned_dataset_path: str) -> pd.DataFrame:
    """
    Transform the cleaned data by normalizing the numeric columns and performing the bonus task.

    Args:
    - cleaned_dataset_path (str): The path to the cleaned dataset.

    Returns:
    - pd.DataFrame: the transformed DataFrame.
    """
    # Load the cleaned data
    df = pd.read_parquet(cleaned_dataset_path)

    # Normalize numeric columns
    df = normalize_numeric_cols(df)

    df.to_parquet('./data/fintech_transformed.parquet')

    return df

def load_to_db(dataset_path: str, table_name: str):
    """
    Load the cleaned dataset to the database.

    Args:
    - dataset_path (str): The path to the cleaned dataset.
    - table_name (str): The name of the table to create in the database.
    """
    cleaned_df = pd.read_parquet(dataset_path)
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            # cleaned.to_sql(table_name, con=engine, if_exists='fail')
            cleaned_df.to_sql(table_name, con=engine, if_exists='replace')
            print('Done writing to database')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')