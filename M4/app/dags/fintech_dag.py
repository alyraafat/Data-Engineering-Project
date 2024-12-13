from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# Import functions from functions.py
from functions import extract_clean, transform, load_to_db
from fintech_dashboard import create_dashboard

default_args = {
    "owner": "de_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    'start_date': days_ago(2),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fintech_dag_pipeline",
    default_args=default_args,
    description="A data pipeline for the Fintech Dashboard",
    schedule_interval='@once',
    tags=['fintech-pipeline'],
) as dag:

    extract_clean_task = PythonOperator(
        task_id="extract_clean",
        python_callable=extract_clean,
        op_kwargs={"dataset_path": "/opt/airflow/data/fintech_data_29_52_1008.parquet"},
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
        op_kwargs={"cleaned_dataset_path": "/opt/airflow/data/fintech_clean.parquet"},
    )

    load_to_db_task = PythonOperator(
        task_id="load_to_db",
        python_callable=load_to_db,
        op_kwargs={"dataset_path": "/opt/airflow/data/fintech_transformed.parquet", "table_name": "fintech_data"},
    )

    create_dashboard_task = BashOperator(
        task_id="create_dashboard",
        bash_command="python /opt/airflow/dags/fintech_dashboard.py --dataset_path {{ params.dataset_path }}",
        params={"dataset_path": "/opt/airflow/data/fintech_transformed.parquet"}
    )


    extract_clean_task >> transform_task >> load_to_db_task >> create_dashboard_task
