import pandas as pd
from kafka import KafkaConsumer
import json
from cleaning import clean_data
from typing import Tuple

def process_messages(messages_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process messages from a DataFrame.

    Args:
    messages_df (pd.DataFrame): DataFrame containing messages to be processed.

    Returns:
    pd.DataFrame: DataFrame containing processed messages.
    """

    print(f"Processing messages: {messages_df}")
    messages_cleaned_df, messages_lookup_table_df = clean_data(dataset_path=None, save_dfs=False, given_df=messages_df)
    return messages_cleaned_df,  messages_lookup_table_df

def consume(topic_name: str='m2_topic') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consume messages from a Kafka topic and process them.

    Args:
    - topic_name: Name of the Kafka topic to consume messages from

    Returns:
    - Tuple of two DataFrames: cleaned messages and lookup table
    """
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f"Listening for messages in {topic_name}...")
    df = pd.DataFrame()
    for message in consumer:
        print(f"Received: {message.value}")
        if message.value == 'EOF':
            print("Received EOF. Exiting.")
            break
        else:
            df = pd.concat([df,pd.DataFrame([message.value])], ignore_index=True)
    consumer.close()
    messages_cleaned_df,  messages_lookup_table_df = process_messages(df)
    return messages_cleaned_df,  messages_lookup_table_df

