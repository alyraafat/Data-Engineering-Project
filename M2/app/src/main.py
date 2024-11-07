import os
import pandas as pd
from cleaning import clean_data
from db import save_to_db
from run_producer import start_producer, stop_container
from consumer import consume

def main():
    kafka_url = 'localhost:9093'
    topic_name = 'm2_topic'
    id = '52_1008'
    cleaned_dataset_path = './data/cleaned_dataset/fintech_data_MET_P2_52_1008_clean.csv'
    dataset_path = './data/dataset/fintech_data_29_52_1008.csv'
    if os.path.exists(cleaned_dataset_path):
        cleaned_df = pd.read_csv(cleaned_dataset_path)
        global_lookup_table = pd.read_csv('./data/cleaned_dataset/lookup_table_MET_P2_52_1008.csv', index_col=None)
    else:
        cleaned_df, global_lookup_table = clean_data(dataset_path=dataset_path)
    
    save_to_db(cleaned_df, 'fintech_data_MET_P2_52_1008_clean')
    print('Data saved to database')
    save_to_db(global_lookup_table, 'lookup_table_MET_P2_52_1008')
    print('Global lookup table saved to database')

    producer_id = start_producer(id=id, kafka_url=kafka_url, topic_name=topic_name)
    messages_cleaned_df,  messages_lookup_table_df = consume(topic_name=topic_name)
    print(messages_cleaned_df)
    print(messages_lookup_table_df)
    
    stop_container(producer_id)

if __name__ == '__main__':
    main()

