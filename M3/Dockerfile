FROM jupyter/pyspark-notebook:latest

RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

    
ADD https://jdbc.postgresql.org/download/postgresql-42.7.3.jar /opt/spark/jars/
ADD fintech_data_29_52_1008.parquet /home/jovyan/work/