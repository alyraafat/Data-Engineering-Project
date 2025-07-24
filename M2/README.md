# Fintech Lending – Milestone 2

## Overview

Milestone 2 productionizes the **Milestone 1** preprocessing pipeline and adds **streaming ingestion**. The solution is fully containerized and, when launched with `docker-compose`, will:

1. **Load & clean** the raw fintech lending dataset (or reuse an already–cleaned file).
2. **Persist** the cleaned dataset and lookup table into **PostgreSQL**.
3. **Start a Kafka producer container** (provided image) that streams incremental “messages”.
4. **Consume & process** each Kafka message, enrich/clean it, and append it to the cleaned dataset table until a terminal `EOF` message is received.
5. Leave PostgreSQL and pgAdmin running for inspection.

---

## Components

### Source Code (`src/`)

| File              | Responsibility                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cleaning.py`     | All reusable **functions** for preprocessing: column standardization, imputation, outlier handling, transformations (log/sqrt/Box‑Cox), normalization, encoding, lookup‑table creation, external state‑name enrichment, and message‐level cleaning (`process_messages`). Exposes a top‑level `clean_data(...)` (or similarly named) function that orchestrates the pipeline.                                 |
| `db.py`           | Database utilities: instantiates SQLAlchemy engine (`postgresql://root:root@pgdatabase:5432/testdb`), `save_to_db` to create/replace the **cleaned** and **lookup** tables, `append_to_table` to add new streaming rows, and helpers to read existing tables.                                                                                                                                                |
| `consumer.py`     | Kafka consumer logic. Subscribes to the configured topic (default `m2_topic`), continuously reads messages, applies `process_messages` to each, appends results to the cleaned table, and terminates gracefully when message value equals `'EOF'`.                                                                                                                                                           |
| `run_producer.py` | Starts/stops the external streaming producer container (`mmedhat1910/dew24_streaming_producer`) via the Docker SDK (`start_producer`, `stop_container`).                                                                                                                                                                                                                                                     |
| `main.py`         | Orchestrator. Checks for an existing cleaned CSV; if present, skips heavy cleaning and just loads/saves to DB. Otherwise: loads raw dataset (`./data/dataset/fintech_data_29_52_1008.csv`), runs `clean_data`, writes cleaned dataset + lookup table to disk & Postgres, launches the producer, runs the consumer until `EOF`, then stops the producer. Implements retry logic for transient startup errors. |

### Data

```
data/
 ├─ dataset/fintech_data_29_52_1008.csv          # Raw input
 └─ cleaned_dataset/fintech_data_MET_P2_52_1008_clean.csv  # Generated (if absent triggers full pipeline)
```

Lookup table CSV (e.g. `lookup_table_MET_P2_52_1008.csv`) is also created in the cleaned folder.

### Database Tables

| Table                                                                                                    | Purpose                                                          |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `fintech_data_MET_P2_52_1008_clean`                                                                      | Full cleaned dataset (initial batch + appended streaming rows).  |
| `lookup_fintech_data_MET_P2_52_1008`                                                                     | Lookup mappings (encodings, imputations, reversible categories). |
| *(Optional)* additional summary tables for statistics (e.g., means/lambdas) depending on implementation. |                                                                  |

### Docker Services (from `docker-compose.yaml`)

* **App (Cleaning Image)**: Builds from `Dockerfile` and runs `python /app/src/main.py`.
* **Postgres (`postgres:13`)**: Persistent database backend.
* **pgAdmin (`dpage/pgadmin4`)**: UI for validating that tables were created/updated.
* **Zookeeper (`confluentinc/cp-zookeeper:7.4.0`)**: Coordination service for Kafka.
* **Kafka (`confluentinc/cp-kafka:7.4.0`)**: Message broker used for streaming ingestion.

The streaming producer container is *not* a permanent service in the compose file; it is started dynamically via `run_producer.start_producer(...)` on the same Docker network (`app_default`) so it can reach Kafka.

---

## Installation & Requirements

### 1. Prerequisites

* Docker & Docker Compose installed.
* The raw dataset placed under `data/dataset/`.

### 2. Python Dependencies

Listed in `requirements.txt`

### 3. Build & Launch

From the Milestone‑2 directory (containing `docker-compose.yaml`):

```bash
docker-compose build           # Build the cleaning image
docker-compose up -d           # Start Postgres, Kafka stack, pgAdmin, and app
```

The application will:

1. Wait for dependencies to become reachable (retry logic).
2. If `./data/cleaned_dataset/fintech_data_MET_P2_52_1008_clean.csv` exists, **skip** full cleaning and immediately load + persist it; otherwise run the full pipeline and generate the file.
3. Write `fintech_data_MET_P2_52_1008_clean` and `lookup_fintech_data_MET_P2_52_1008` tables to Postgres.
4. Start the **producer** container (using your ID and topic).
5. Consume messages until `'EOF'`.
6. Stop and remove the producer container.

### 4. Inspecting Results

Open pgAdmin (URL/credentials as configured in `docker-compose.yaml`), connect to the `testdb` database, and verify:

* Tables exist with expected row counts.
* Streaming messages appear appended after initial batch load.

### 5. Re‑Running / Iteration

* To force a full re-clean: delete or rename the existing cleaned CSV and re-run `docker-compose up --build`.
* To reprocess streaming only: keep the cleaned CSV; the app will ingest and append new messages on next start.

### 6. Stopping

```bash
docker-compose down
```

Add `-v` to remove named volumes if you want a fresh Postgres state.

---

## Internal Processing Highlights

### Cleaning & Feature Engineering

Key operations inside `cleaning.py` include:

* **Column normalization** (`tidy_up_columns`, `set_index`).
* **Imputation** (`impute_emp_title`, employment length by income bins, interest rate by grade, etc.).
* **Outlier handling** (IQR and z‑score strategies).
* **Transformations** (`log_transform_col`, `sqrt_transform_col`, `apply_boxcox`, normalization).
* **Encoding** (`encode_col`) with reversible mappings stored in a global lookup table.
* **External enrichment** (scraping full U.S. state names).
* **Streaming message prep** (`process_messages`) to clean/standardize incoming message records before appending.

### Streaming Flow

1. **Producer** emits JSON‑serializable objects to Kafka topic.
2. **Consumer** deserializes, converts to DataFrame, calls `process_messages`, then `append_to_table`.
3. On receiving `'EOF'`, consumer closes gracefully—no busy loop remains.

---
