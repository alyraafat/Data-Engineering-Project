# Fintech Lending – Milestone 3 (PySpark)

## Overview

Milestone 3 re‑implements the previous lending data pipeline in **PySpark**, taking advantage of distributed processing and Spark SQL.
The notebook `m3_spark_52_1008.ipynb`:

1. Loads the assigned Parquet dataset.
2. Cleans and imputes data with reusable functions.
3. Encodes selected categorical features (mix of One‑Hot and Label Encoding).
4. Engineers *lag* features using Spark window functions (no Python UDFs).
5. Answers five business questions **twice** (Spark SQL **and** DataFrame API).
6. Saves a cleaned dataset and a lookup table as Parquet (and optionally to Postgres).

---

## Key Features & What Was Implemented

### 1. Data Loading & Partitioning

The raw dataset (`fintech_spark_52_1008.parquet`) is read into a Spark DataFrame.
Initial partition count is inspected and the DataFrame is **repartitioned** to match the machine’s logical CPU cores for balanced parallelism.

### 2. Cleaning Pipeline

All logic is implemented as pure functions to keep the process reproducible:

| Function                     | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `detect_missing`             | Computes percentage of missing values per column.                            |
| `handle_missing_numerical`   | Imputes null numeric columns with `0`.                                       |
| `handle_missing_categorical` | Imputes null string/categorical columns with the mode (most frequent value). |
| Column renaming              | All columns converted to `lower_snake_case`.                                 |

After imputation, an assertion verifies there are no remaining nulls.

### 3. Categorical Encoding

Only the required columns are encoded:

| Column                 | Encoding                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `emp_length`           | Converted to numeric via regex/digit extraction (`convert_emp_length_to_numeric`).   |
| `home_ownership`       | One‑Hot (`one_hot_encode`).                                                          |
| `verification_status`  | One‑Hot.                                                                             |
| `state` (`addr_state`) | Label Encoding (`label_encode`).                                                     |
| `type`                 | One‑Hot.                                                                             |
| `purpose`              | Label Encoding.                                                                      |
| `grade`                | Discretized to its **letter grade** via `create_letter_grade` (no further encoding). |

All other text fields (e.g., employment title, description) are left untouched.

A **lookup table** is assembled programmatically from the encoders so that transformations are reversible.

### 4. Feature Engineering (Lag Features)

Using Spark window functions (no UDFs), four “previous loan” features are created:

1. `prev_issue_date_grade` – previous loan’s issue date within the same grade.
2. `prev_amount_grade` – previous loan amount within the same grade.
3. `prev_issue_date_state_grade` – previous loan’s issue date within the same *state + grade*.
4. `prev_amount_state_grade` – previous loan amount within the same *state + grade*.

Supporting functions:

* `add_previous_loan_issue_date_form_same_grade`
* `add_prev_loan_amount_from_same_grade`
* `add_prev_loan_date_from_same_state_and_grade`
* `add_prev_loan_amount_from_same_state_and_grade`

### 5. Dual‑Mode Analysis (SQL & DataFrame API)

Each business question is answered twice—once with raw Spark SQL and once with the DataFrame API:

1. **Default Loans:** Average loan amount & interest rate by employment length and binned annual‑income ranges.
2. **Amount vs Funded Gap:** Average `(loan_amount - funded_amount)` per grade (descending).
3. **Verification Comparison:** Total loan amount for “Verified” vs “Not Verified” status across states.
4. **Inter‑Loan Time Gap:** Average days between consecutive loans per grade using engineered lag issue dates.
5. **Inter‑Loan Amount Change:** Average difference in loan amounts between consecutive loans within the same state–grade cluster.

Temporary views (`createOrReplaceTempView`) enable clean SQL; equivalent DataFrame transformations (groupBy, window, withColumn, etc.) provide the parallel API solutions.

### 6. Outputs

Two Parquet files are produced:

* `fintech_spark_52_1008_clean.parquet` – Final cleaned dataset including engineered features and encodings.
* `lookup_spark_52_1008.parquet` – Encoding/translation table for categorical transformations.

### 7. Optional: Postgres Load

A utility function `save_to_db` (with SQLAlchemy) demonstrates exporting the cleaned dataset (and optionally the lookup table) to a Postgres instance for inspection via pgAdmin.

---

## How to Run

### Local (PySpark)

```bash
pip install pyspark pandas sqlalchemy psycopg2-binary fastparquet
python  # or open Jupyter
```

Open and run `m3_spark_52_1008.ipynb` top‑to‑bottom.

### (Optional) Docker / Postgres

If using Docker Compose (with Postgres/pgAdmin):

```bash
docker-compose up -d
# Run the notebook pointing to the running Postgres service, then call save_to_db()
```

Inspect tables and engineered columns in pgAdmin.

