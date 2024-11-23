FROM jupyter/pyspark-notebook:latest

# Switch to root user to install system packages
USER root

# Install required system packages
RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the default user
USER ${NB_UID}

# Add the JDBC driver and dataset
ADD https://jdbc.postgresql.org/download/postgresql-42.7.3.jar /opt/spark/jars/
ADD fintech_data_29_52_1008.parquet /home/jovyan/work/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
