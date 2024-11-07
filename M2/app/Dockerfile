# Use the slim version of Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data into the container
COPY src /app/src
COPY data /app/data

# Set the entry point to main.py
CMD ["python", "/app/src/main.py"]
