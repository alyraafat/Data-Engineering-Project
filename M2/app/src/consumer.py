import pandas as pd
from kafka import KafkaConsumer
import json
from cleaning import process_messages
from typing import Tuple
from db import append_to_table


def start_consumer(topic_name: str='m2_topic') -> KafkaConsumer:
    """
    Start a Kafka consumer to listen for messages on a specified topic.

    Args:
    - topic_name (str): Name of the topic to listen for messages on.

    Returns:
    - KafkaConsumer: Kafka consumer object
    """
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers='kafka:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    return consumer

def consume(consumer: KafkaConsumer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consume messages from a Kafka topic and process them.

    Args:
    - consumer: Kafka consumer object

    Returns:
    - Tuple of two DataFrames: cleaned messages and lookup table
    """
    while True:
        messages_all = consumer.poll()
        for _, messages in messages_all.items():
            for message in messages:
                print(f"Received: {message.value}")
                if message.value == 'EOF':
                    print("Received EOF. Exiting.")
                    consumer.close()
                    return
                else:
                    df = pd.DataFrame([message.value])
                    processed_df = process_messages(df)
                    append_to_table(processed_df, 'fintech_data_MET_P2_52_1008_clean')
                    print(processed_df)
    

