# Fintech Lending – Milestone 4 (Airflow ETL + Dashboard)

## Overview

Milestone 4 productionizes the batch pipeline with **Apache Airflow** and adds a **dashboard** for exploratory analytics. The solution orchestrates an end‑to‑end ETL flow (`extract_clean → transform → load_to_db → run_dashboard`) and serves an interactive dashboard (Plotly Dash) answering five business questions about the lending portfolio.

---

## 1. ETL Pipeline (Airflow)

**DAG file:** `fintech_dag.py`
**Schedule:** (typically `@once` / manual trigger for the milestone)
**Tasks:**

1. **`extract_clean`**

   * Reads the **raw CSV** from `data/`.
   * Executes the Milestone‑2 cleaning logic (column standardization, imputations, outlier handling, state name enrichment, encodings, etc.).
   * Persists intermediate output as `fintech_clean.csv`/`parquet`.

2. **`transform`**

   * Loads the cleaned file.
   * Applies all transformations from earlier milestones (e.g., log/Box‑Cox/sqrt transforms, normalization or standardization).
   * Saves result as `fintech_transformed.csv`/`parquet`.

3. **`load_to_db`**

   * Reads the transformed dataset.
   * Loads it into **PostgreSQL** (`pgdatabase`) using SQLAlchemy/psycopg2, creating or replacing the target table.

4. **`run_dashboard`**

   * Invokes the dashboard’s `create_dashboard(...)` function (see below) so the web app is available after ETL succeeds.

> Airflow cannot pass in‑memory DataFrames between tasks; each step writes to disk so downstream tasks re‑load from file.

### Functions Module

`functions.py` centralizes all reusable logic. For simplicity (per milestone instructions) it lives in the `dags/` folder and is imported inside the DAG. Each top‑level function wraps lower‑level helpers from Milestone 2. This keeps the DAG file minimal and testable.

---

## 2. Dashboard

**File:** `fintech_dashboard.py`
**Entry Point:** `create_dashboard(cleaned_or_transformed_path: str, host: str="0.0.0.0", port: int=8050)`

Implemented with **Plotly Dash** (or Streamlit if you chose that) and launched from the Airflow task. The dashboard loads the **transformed dataset** and renders:

1. **Distribution of Loan Amounts by Grade**

   * Box/violin plot showing spread of `loan_amount` across letter (or encoded) grades.

2. **Loan Amount vs Annual Income Across States (Interactive)**

   * Scatter plot with `annual_income` (original) vs `loan_amount`.
   * Color: `loan_status`.
   * Dropdown filter: *State* (`All` + individual states). Selecting `All` removes filtering.

3. **Trend of Loan Issuance Over Months (Interactive)**

   * Line chart of count (or sum) of loans per `issue_month`.
   * Year dropdown filters the series dynamically.

4. **States with Highest Average Loan Amount**

   * Bar chart (or optional choropleth) of average `loan_amount` by state, sorted descending.

5. **Percentage Distribution of Loan Grades**

   * Pie/donut or histogram showing proportion of each grade.


---

## 3. Technology Stack

| Component                         | Purpose                                                  |
| --------------------------------- | -------------------------------------------------------- |
| **Airflow**                       | Orchestrates ETL tasks as a DAG.                         |
| **PostgreSQL + pgAdmin**          | Persistent analytical store and verification UI.         |
| **Plotly Dash**                   | Interactive web dashboard served from the container.     |
| **Pandas / scikit‑learn / SciPy** | Cleaning, transformations, statistical encodings.        |
| **SQLAlchemy / psycopg2**         | Database connectivity.                                   |
| **Docker / docker‑compose**       | Reproducible environment for Airflow, DB, and dashboard. |

`requirements.txt` lists all Python dependencies bundled into the Airflow image (dashboard library included). The Dockerfile exposes the dashboard port so it is reachable from the host.

---

## 4. Running the Project

### Using Docker Compose

```bash
docker-compose up -d --build
```

This spins up:

* Airflow webserver & scheduler
* PostgreSQL (`pgdatabase`)
* pgAdmin
* (Your) Airflow worker/container hosting the dashboard

Navigate to the Airflow UI, trigger `fintech_dag`, and monitor task execution. After success, open the dashboard in your browser at the exposed host port (e.g., `http://localhost:8050`).

### Manual / Local (Optional)

If running outside Docker, install dependencies and set Airflow variables/connection for Postgres, then run the DAG via `airflow dags trigger fintech_dag`.

---

## 5. Data Flow Summary

```
Raw CSV
  │
  ├─(extract_clean)─> fintech_clean.parquet
  │
  ├─(transform)─────> fintech_transformed.parquet
  │
  ├─(load_to_db)────> PostgreSQL table
  │
  └─(run_dashboard)→ Dashboard queries transformed dataset / DB
```

Each stage is idempotent: deleting an intermediate file and re‑running the DAG recomputes only downstream artifacts.

---

## 6. Implementation Highlights

* **Modularization:** Encapsulation of previous milestone logic into three high‑level functions enables concise DAG definitions.
* **Reproducibility:** Deterministic file naming and transformation order simplify debugging and future automation.
* **Observability:** Airflow logs show per‑task runtime; Postgres provides persisted state; dashboard offers immediate visual validation.
* **Extensibility:** New tasks (e.g., model training) can be appended as downstream dependencies after `load_to_db`.

