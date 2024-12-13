from functions import *
from fintech_dashboard import create_dashboard



def main():
    cleaned_df = extract_clean('./data/fintech_data_29_52_1008.parquet')
    transformed_df = transform('./data/fintech_clean.parquet')
    # load_to_db('./data/fintech_transformed.parquet', 'fintech_data')
    create_dashboard()


# if __name__ == "__main__":
#     main()