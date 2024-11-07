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