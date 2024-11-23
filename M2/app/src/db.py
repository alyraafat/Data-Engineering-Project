from sqlalchemy import create_engine
import pandas as pd


engine = create_engine('postgresql://root:root@pgdatabase:5432/testdb')

def save_to_db(cleaned: pd.DataFrame, table_name: str):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            # cleaned.to_sql(table_name, con=engine, if_exists='fail')
            cleaned.to_sql(table_name, con=engine, if_exists='replace')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')
    
def get_table(table_name: str) -> pd.DataFrame:
    """
    Get data from database table.

    Args:
    - table_name (str): Name of the table to get data from.

    Returns:
    - pd.DataFrame: Data from the specified table.
    """
    # if(engine.connect()):
    #     print("Connected to database")
    #     try:
    #         with engine.connect() as conn, conn.begin(): 
    #             print(f"Reading {table_name} from database")
    #             df = pd.read_sql_table(table_name, conn)
    #             print(f"Done reading {table_name} from database")
    #     except:
    #         print(f"Failed to read {table_name} from database")
    #         df = pd.read_csv(f'./data/cleaned_dataset/{table_name}.csv')
    # else:
    #     df = pd.read_csv(f'./data/cleaned_dataset/{table_name}.csv')
    df = pd.read_csv(f'./data/cleaned_dataset/{table_name}.csv')
    return df
    
def append_to_table(df: pd.DataFrame, table_name: str) -> None:
    """
    Append data to a database table.

    Args:
    - df (pd.DataFrame): Data to append to the table.
    - table_name (str): Name of the table to append data to.
    """
    if(engine.connect()):
        print("Connected to database")
        try:
            with engine.connect() as conn, conn.begin():
                print(f"Appending data to {table_name}")
                df.to_sql(table_name, con=engine, if_exists='append')
                print(f"Done appending data to {table_name}")
        except Exception as ex:
            print(f"Failed to append data to {table_name}")
            print(ex)
    else:
        print("Failed to connect to database")