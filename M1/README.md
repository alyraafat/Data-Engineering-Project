# Fintech Lending – Milestone 1

## Overview

This repository contains the first milestone for a Fintech Lending data engineering / analytics pipeline.
The notebook **`M1_MET_P2_52_1008.ipynb`** performs:

1. **Exploratory Data Analysis (EDA)** – answering business questions with visualizations.
2. **Data Cleaning** – standardizing columns, indexing, imputing missing values, handling inconsistencies and outliers.
3. **Feature Engineering & Transformation** – creating new informative fields and applying statistical transforms.
4. **External Data Enrichment** – scraping U.S. state full names from the web.
5. **Encoding & Lookup Table** – consistent reversible encodings stored in a global lookup table.
6. **Dataset Export** – saving a clean, model‑ready dataset and lookup table.

Raw input file: `fintech_data_29_52_1008.csv`
Outputs:

- `fintech_data_MET_P2_52_1008_clean.csv`
- `lookup_table_MET_P2_52_1008.csv`

---

## Notebook Structure

All logic is implemented as reusable **functions** inside the notebook (can be refactored into a `src/` package later). Execution proceeds in logical phases:

### 1. EDA Functions

`analyze_loan_terms_by_emp_length`, `analyze_grade_vs_status`, `analyze_grade_status_by_term`,
`analyze_loan_status_by_purpose_and_income`, `analyze_loan_amount_by_state_and_income`,
`analyze_high_risk_states`

These functions generate bar charts / stacked bars to study relationships among **loan grade**, **status**, **purpose**, **employment length**, **income**, and **state**, helping uncover risk patterns.

### 2. Cleaning & Standardization

Key utilities:

- `tidy_up_columns(df)` – lowercase and replace spaces with `_`.
- `set_index(col_name, df)` – set a stable primary key.
- `standardize_loan_type(...)` (and similar) – harmonize categorical spellings.
- Duplicate / inconsistency checks (inline code).
- **Missing value imputation** functions:

  - `impute_emp_title` → `'unknown'`
  - `impute_emp_length_by_income_bin` → conditional fill based on income grouping
  - `impute_annual_inc_joint` → `0`
  - `impute_int_rate` → grade‑level mean
  - `impute_description` → `'No Description'`

### 3. Outlier Handling

Two strategies implemented:

- `remove_outliers_using_boxplot` (IQR whiskers)
- `remove_outliers_using_z_score` (standard score threshold)

### 4. Feature Engineering

New/derived columns:

| Feature                     | Description                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| `issue_month`               | Month extracted from `issue_date_cleaned` via `add_month_col`.                                         |
| `salary_can_cover`          | Boolean: annual income ≥ loan amount (`can_salary_cover` / `salary_cover`).                            |
| `letter_grade_labelEncoded` | Label‑encoded ordinal grade from numeric `grade` using `change_number_to_grade`/`get_grade`.           |
| `installment_per_month`     | Amortized monthly payment from principal, interest rate, and term (`calculate_installment_per_month`). |
| `state_name`                | Full state name scraped via `get_state_names` + `add_state_names`.                                     |

### 5. Transformations

Numeric skew mitigation:

- Log / log1p: `log_transform_col`
- Square root: `sqrt_transform_col`
- Box‑Cox: `apply_boxcox`
- Standardization helpers (`get_orig_col_name` ensures clean naming)

Examples present in final dataset: `annual_inc_log`, `avg_cur_bal_boxcox`, `tot_cur_bal_boxcox`.

### 6. Encoding & Lookup Table

`encode_col(df, type_of_encoding, col_name, need_to_sort=False)` performs **one‑hot** (`ohe`) or **label** encoding.
A global `global_lookup_table` (`Column`, `Original`, `Encoded`) accumulates mappings for full reversibility. Saved as `lookup_table_MET_P2_52_1008.csv`.

### 7. Column Selection & Export

A curated subset (`cols_to_keep`) is assembled (categorical dummies, engineered features, transformations, imputed columns, etc.), then written to `fintech_data_MET_P2_52_1008_clean.csv`.

---

## Extending

- **Modeling:** Feed the clean CSV into ML experiments (classification of loan default risk).
- **Automation:** Convert notebook cells to Python modules and orchestrate with Airflow or a Makefile.
- **Data Validation:** Integrate `great_expectations` or `pydantic` for schema checks before export.
