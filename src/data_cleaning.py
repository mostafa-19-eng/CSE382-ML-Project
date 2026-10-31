import os
import pandas as pd
import numpy as np

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Load Dataset and explicitly catch 'NULL', whitespace, and blanks as NaN
data_path = 'data/H2.csv'
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found. Please ensure H2.csv is placed in the data/ directory.")
    exit()

# na_values catches various forms of null/NULL strings automatically upon loading
missing_markers = ['NULL', 'null', '', ' ', '   ', 'NaN', 'nan']
df = pd.read_csv(data_path, na_values=missing_markers, skipinitialspace=True)
initial_shape = df.shape
print(f"Original Dataset Shape: {initial_shape}")

# ==========================================
# APPLIED CLEANING & REFINEMENT STEPS
# ==========================================

# Step A: Double-check regex replacement for any leftover whitespace or string 'NULL'
df.replace(r'^\s*NULL\s*$', np.nan, regex=True, inplace=True)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Step B: Drop Data Leakage Columns
leakage_cols = ['ReservationStatus', 'ReservationStatusDate']
cols_to_drop = [col for col in leakage_cols if col in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

# Step C: Clean ADR Anomalies (Remove <= 0 and extreme outliers > 1000 with precise tracking)
if 'ADR' in df.columns:
    invalid_adr = (df['ADR'] <= 0).sum()
    extreme_adr = (df['ADR'] > 1000).sum()
    df = df[(df['ADR'] > 0) & (df['ADR'] <= 1000)]
    print(f"ADR cleaned -> Removed: {invalid_adr} non-positive, {extreme_adr} extreme (>1000) records.")

# Step D: Impute Missing Values cleanly
if 'Country' in df.columns:
    df['Country'] = df['Country'].fillna('Unknown')

if 'Agent' in df.columns:
    df['Agent'] = df['Agent'].fillna(0)

if 'Company' in df.columns:
    df['Company'] = df['Company'].fillna(0)

if 'Children' in df.columns:
    df['Children'] = df['Children'].fillna(0)

# Step E: Remove rows with zero total guests (after imputing Children)
guest_cols = [col for col in ['Adults', 'Children', 'Babies'] if col in df.columns]
if guest_cols:
    df['TotalGuests'] = df[guest_cols].sum(axis=1)
    df = df[df['TotalGuests'] > 0]

# Step F: Feature Engineering & Naming Consistency (RoomMismatch)
if all(col in df.columns for col in ['ReservedRoomType', 'AssignedRoomType']):
    df['RoomMismatch'] = (df['ReservedRoomType'] != df['AssignedRoomType']).astype(int)

# ==========================================
# FINAL VALIDATION & SUMMARY REPORT
# ==========================================
final_shape = df.shape
total_rows_removed = initial_shape[0] - final_shape[0]

print("\n" + "="*50)
print("DATA CLEANING & AUDIT REPORT SUMMARY")
print("="*50)
print(f"• Initial Dataset Shape          : {initial_shape[0]} rows, {initial_shape[1]} columns")
print(f"• Final Cleaned Dataset Shape    : {final_shape[0]} rows, {final_shape[1]} columns")
print(f"• Total Rows Removed / Filtered  : {total_rows_removed} rows")
print(f"• Literal 'NULL' / Whitespace    : Handled via `na_values` and regex mapping.")
print(f"• Missing Value Imputation       : 'Country' -> 'Unknown'; 'Agent', 'Company', 'Children' -> 0.")
print(f"• Remaining Missing Values       : {df.isnull().sum().sum()} total nulls remaining.")
print("="*50)

# Save Cleaned Dataset
cleaned_output_path = 'data/H2_cleaned.csv'
df.to_csv(cleaned_output_path, index=False)
print(f"\nSuccessfully exported cleaned dataset to: {cleaned_output_path}")