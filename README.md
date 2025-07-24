# Fintech Lending – Project Summary

A four‑milestone data engineering & analytics project that evolves from local preprocessing to a fully orchestrated, distributed, and dashboarded pipeline.

---

## Milestones Overview

### **Milestone 1 – Exploratory Data & Batch Cleaning**

* Performed EDA answering analytical questions.
* Implemented **all steps as functions**: column normalization, missing value imputation, outlier handling.
* Engineered features (e.g., monthly installment, salary coverage, grade label).
* Built a reversible **lookup table** for encodings/imputations.
* Saved cleaned dataset (`csv/parquet`) for reuse.

### **Milestone 2 – Containerized Batch + Streaming Pipeline**

* Dockerized the cleaning code and persisted outputs to **PostgreSQL**.
* Consumed real‑time loan records from **Kafka** (producer container) until `EOF`.
* Enriched/cleaned streaming messages and appended them to the batch table.
* Provided pgAdmin access for verification.

### **Milestone 3 – Distributed Processing with PySpark**

* Migrated pipeline to **PySpark**.
* Re‑implemented cleaning & encoding using Spark ML transformers.
* Engineered **lag features** via window functions (previous loan date/amount per grade and per state+grade).
* Answered business questions twice: **Spark SQL** and **DataFrame API**.
* Stored results & lookup table as Parquet (optionally loaded into Postgres).

### **Milestone 4 – Orchestrated ETL & Analytics Dashboard**

* Built an **Apache Airflow** DAG: `extract_clean → transform → load_to_db → run_dashboard`.
* Tasks exchange artifacts via on‑disk files for reproducibility.
* Served an interactive **Plotly Dash** dashboard (filters, charts) answering five portfolio questions.
* Optional Postgres loading + screenshots for validation.

---

## End‑to‑End Data Flow

```
Raw CSV/Parquet
   └─(M1 Functions)→ Cleaned Dataset + Lookup
        └─(M2 Kafka Streaming)→ Updated Cleaned Table in Postgres
             └─(M3 PySpark)→ Distributed Clean/Features/Analysis (Parquet)
                  └─(M4 Airflow)→ ETL Orchestration + Dashboard Visualization
```

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/BeautifulSoup-4B8BBE?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Apache%20Kafka-231F20?logo=apachekafka&logoColor=white"/>
  <img src="https://img.shields.io/badge/ZooKeeper-FF6A00?logo=apache&logoColor=white"/>
  <img src="https://img.shields.io/badge/PySpark-E25A1C?logo=apachespark&logoColor=white"/>
  <img src="https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=apacheairflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly%20Dash-3F4F75?logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white"/>
</p>

---

## Highlights

* **Modular Functions:** Reusability across milestones.
* **Streaming + Batch Integration:** Unified table with incremental updates.
* **Distributed Processing:** Scalable transformation & feature engineering.
* **Orchestrated ETL:** Airflow ensures deterministic, observable runs.
* **Interactive Insights:** Dashboard enables exploratory analysis for stakeholders.

