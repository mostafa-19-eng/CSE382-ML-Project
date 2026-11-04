# Milestone 1: Data Understanding, EDA & Data Cleaning Report 

**Dataset:** City Hotel (`data/H2.csv` -> `data/H2_cleaned.csv`)



## 1. Executive Summary & Overview

This document details the exploratory data analysis (EDA), missing value normalization, and data cleaning executed on the City Hotel dataset (`H2.csv`). The processing pipeline handles space-padded null markers, filters pricing anomalies, removes invalid operational logs, and exports a fully sanitized dataset ready for downstream preprocessing and modeling.

* **Initial Raw Dataset Shape:** 79,330 rows × 31 columns

* **Final Cleaned Dataset Shape:** 78,091 rows × 31 columns

* **Total Rows Filtered / Removed:** 1,239 rows

* **Remaining Missing Values:** 0 total nulls across all features

## 2. Ingestion & Missing Value Parsing

In the raw `H2.csv` file, missing values across categorical and identifier columns are represented as literal strings with varying whitespace (e.g., `"NULL"`, `" NULL "`, `" "`). Default pandas ingestion misses these strings, underreporting missingness in columns like `Agent` and `Company`.

To ensure consistent missing-value parsing across EDA (`src/Eda_first.py`) and Cleaning (`src/data_cleaning.py`), explicit `na_values` and regular expressions are applied during loading:

```python
missing_markers = ['NULL', 'null', '', ' ', '   ', 'NaN', 'nan']
df = pd.read_csv(data_path, na_values=missing_markers, skipinitialspace=True)
df.replace(r'^\s*NULL\s*$', np.nan, regex=True, inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
```

### Confirmed Missingness Counts (Raw Ingestion):

* **`Company`**: 75,641 missing values (95.35%) $\rightarrow$ Imputed with `0` (represents direct/individual booking without a corporate sponsor).

* **`Agent`**: 8,131 missing values (10.25%) $\rightarrow$ Imputed with `0` (represents direct booking without a travel agency).

* **`Country`**: 24 missing values (0.03%) $\rightarrow$ Imputed with `'Unknown'`.

* **`Children`**: 4 missing values (0.005%) $\rightarrow$ Imputed with `0`.

## 3. Row Filtering & Exclusion Justifications

| Exclusion Rule | Rows Removed | Rationale & Domain Justification | 
| ----- | ----- | ----- | 
| **Non-Positive ADR (**$\text{ADR} \le 0$**)** | 1,208 rows | Non-positive Average Daily Rates distort cost scaling, log transformations, and error evaluation metrics ($\text{RMSE}/\text{MAE}$). | 
| **Extreme Pricing Outlier (**$\text{ADR} > 1000$**)** | 1 row | Severe single-transaction anomaly ($> \$1000$) that heavily skews linear regression modeling and variance calculations. | 
| **Zero-Guest Occupancy (**$\text{TotalGuests} = 0$**)** | 30 rows | Bookings logged with 0 adults, 0 children, and 0 babies represent invalid operational logging errors. | 

## 4. Duplicate Rows Handling

* **Raw Duplicate Count:** 25,902 exact duplicate feature rows

* **Cleaned Duplicate Count:** 25,946 exact duplicate feature rows

* **Policy Decision:** **Exact duplicate rows are preserved.**

* **Rationale:** In hotel reservation management, duplicate feature vectors represent legitimate distinct bookings made under identical parameters (e.g., corporate group bookings or standard room reservations for identical dates). Removing them would artificially flatten actual booking frequencies.

## 5. Scope of Data Cleaning & Feature Engineering 

1. **Deterministic Missing Value Imputation:** Fixed-sentinel mapping across all missing values (`Country` $\rightarrow$ `'Unknown'`, `Agent`/`Company`/`Children` $\rightarrow$ `0`).

2. **Data Leakage Elimination:** Global removal of post-booking outcome attributes (`ReservationStatus`, `ReservationStatusDate`) to prevent artificial label leakage.

3. **Feature Construction:** Engineering domain-specific composite variables:

   * `TotalGuests` = `Adults` + `Children` + `Babies`

   * `RoomUpgrade` / `RoomMismatch` = (`ReservedRoomType` $\ne$ `AssignedRoomType`)

4. **Data Sanitization & Export:** Validating and exporting `data/H2_cleaned.csv` with zero remaining nulls.

## 6. Cleaned Dataset Statistical Audit (`data/H2_cleaned.csv`)

### Classification Target Distribution (`IsCanceled`):

* **Class 0 (Not Canceled):** 57.78% (45,119 rows)

* **Class 1 (Canceled):** 42.22% (32,972 rows)

### Average Daily Rate (ADR) Summary:

* **Minimum ADR:** \$0.50

* **Maximum ADR:** \$510.00

* **Mean ADR:** \$106.88

### Overall Data Quality:

* **Total Missing Values:** 0

* **Dataset Shape:** (78,091, 31)