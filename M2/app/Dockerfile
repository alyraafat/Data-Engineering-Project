FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY src /app/src
# COPY data /app/data

CMD ["python", "/app/src/main.py"]
