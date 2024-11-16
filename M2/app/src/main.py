import os
import pandas as pd
from cleaning import clean_data
from db import save_to_db
from run_producer import start_producer, stop_container
from consumer import consume, start_consumer
import time

def main():
    kafka_url = 'kafka:9092'
    topic_name = 'm2_topic'
    id = '52_1008'
    cleaned_dir_path = './data/cleaned_dataset'
    cleaned_dataset_path = f'{cleaned_dir_path}/fintech_data_MET_P2_52_1008_clean.csv'
    dataset_path = './data/dataset/fintech_data_29_52_1008.csv'
    if os.path.exists(cleaned_dataset_path):
        cleaned_df = pd.read_csv(cleaned_dataset_path)
        global_lookup_table = pd.read_csv(f'{cleaned_dir_path}/lookup_table_MET_P2_52_1008.csv', index_col=None)
        multivariate_lookup_table = pd.read_csv(f'{cleaned_dir_path}/multivariate_lookup_table_MET_P2_52_1008.csv', index_col=None)
        boxcox_lambdas_df = pd.read_csv(f'{cleaned_dir_path}/boxcox_lambdas_MET_P2_52_1008.csv', index_col=None)
    else:
        cleaned_df, global_lookup_table, multivariate_lookup_table, boxcox_lambdas_df = clean_data(dataset_path=dataset_path)
    
    save_to_db(cleaned_df, 'fintech_data_MET_P2_52_1008_clean')
    print('Data saved to database')
    save_to_db(global_lookup_table, 'lookup_table_MET_P2_52_1008')
    print('Global lookup table saved to database')
    save_to_db(multivariate_lookup_table, 'multivariate_lookup_table_MET_P2_52_1008')
    print('Multivariate lookup table saved to database')
    save_to_db(boxcox_lambdas_df, 'boxcox_lambdas_MET_P2_52_1008')
    print('Boxcox lambdas saved to database')

    consumer = start_consumer(topic_name=topic_name)
    producer_id = start_producer(id=id, kafka_url=kafka_url, topic_name=topic_name)
    consume(consumer=consumer)
    
    stop_container(producer_id)


if __name__ == '__main__':
    num_tries = 1
    while num_tries <= 5:
        try:
            main()
            break
        except Exception as e:
            print(e)
            num_tries += 1
            print(f'Try {num_tries} failed. Retrying...')
            time.sleep(5)
    if num_tries > 5:
        print(f'All {num_tries} tries failed. Exiting...')

