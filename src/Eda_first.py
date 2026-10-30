import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)
sns.set_theme(style="whitegrid")

# ==========================================
# LOAD DATASET (H2.csv)
# ==========================================
data_path = 'data/H2.csv'
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found. Please ensure H2.csv is placed in the data/ directory.")
    exit()

df = pd.read_csv(data_path)
print(f"Successfully loaded H2.csv with shape: {df.shape}")  # Should be (79330, 31)

# ==========================================
# 1. TARGET INQUIRIES & INITIAL PROFILE
# ==========================================
print("\n--- [1] Target Inquiries & Initial Profile ---")

# Classification Base Rates (IsCanceled)
if 'IsCanceled' in df.columns:
    cancel_counts = df['IsCanceled'].value_counts(normalize=True) * 100
    print("IsCanceled Class Distribution (%):\n", cancel_counts)

# Regression Distribution Analysis (ADR)
if 'ADR' in df.columns:
    adr_clean = df['ADR'].dropna()
    adr_skew = skew(adr_clean)
    adr_kurt = kurtosis(adr_clean)
    anomalies_neg = (adr_clean <= 0).sum()
    print(
        f"ADR Statistics -> Skewness: {adr_skew:.2f}, Kurtosis: {adr_kurt:.2f}, Non-positive ADR count (<=0): {anomalies_neg}")

    plt.figure(figsize=(8, 4))
    sns.histplot(adr_clean[adr_clean < 500], kde=True, color='teal', bins=50)
    plt.title('Distribution of ADR (Filtered < 500 for visualization)')
    plt.savefig('outputs/adr_distribution.png')
    plt.close()

# Missingness Mapping
high_risk_cols = [col for col in ['Country', 'Agent', 'Company'] if col in df.columns]
if high_risk_cols:
    print("\nMissing values in high-risk columns:")
    print(df[high_risk_cols].isnull().sum())

# ==========================================
# 2. TEMPORAL & SEASONALITY FOOTPRINT
# ==========================================
print("\n--- [2] Temporal & Seasonality Footprint ---")

# Cancellation Horizons (LeadTime vs IsCanceled)
if 'LeadTime' in df.columns and 'IsCanceled' in df.columns:
    lead_corr = df['LeadTime'].corr(df['IsCanceled'])
    print(f"Correlation between LeadTime and IsCanceled: {lead_corr:.3f}")

# Rate Elasticity over Time
if 'ArrivalDateMonth' in df.columns and 'ADR' in df.columns:
    monthly_adr = df.groupby('ArrivalDateMonth')['ADR'].mean()
    print("\nMean ADR by Month:\n", monthly_adr)

# ==========================================
# 3. CUSTOMER & BEHAVIORAL GEOMETRY
# ==========================================
print("\n--- [3] Customer & Behavioral Geometry ---")

# Composition Metrics (TotalGuests)
guest_cols = [col for col in ['Adults', 'Children', 'Babies'] if col in df.columns]
if guest_cols:
    df['TotalGuests'] = df[guest_cols].sum(axis=1)
    print("Total Guests summary statistics:\n", df['TotalGuests'].describe())

# Loyalty & Risk Indicators
loyalty_cols = [col for col in ['IsRepeatedGuest', 'PreviousCancellations', 'PreviousBookingsNotCanceled'] if
                col in df.columns]
if loyalty_cols:
    print("\nLoyalty & Risk Summary:\n", df[loyalty_cols].mean())

# ==========================================
# 4. OPERATIONAL VARIANCE & MISMATCHES
# ==========================================
print("\n--- [4] Operational Variance & Mismatches ---")

# Room Upgrades Analysis
if all(col in df.columns for col in ['ReservedRoomType', 'AssignedRoomType']):
    df['RoomUpgrade'] = df['ReservedRoomType'] != df['AssignedRoomType']
    upgrade_pct = df['RoomUpgrade'].mean() * 100
    print(f"Percentage of bookings with Room Upgrade/Mismatch: {upgrade_pct:.2f}%")

# Policy Constraints (DepositType vs IsCanceled)
if all(col in df.columns for col in ['DepositType', 'IsCanceled']):
    print("\nCancellation Rate by Deposit Type (%):\n", df.groupby('DepositType')['IsCanceled'].mean() * 100)

# ==========================================
# 5. MULTI-GOAL FEATURE INTERACTIONS
# ==========================================
print("\n--- [5] Multi-Goal Feature Interactions ---")

# Stratified Density Analysis
if all(col in df.columns for col in ['LeadTime', 'ADR', 'IsCanceled']):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x='LeadTime', hue='IsCanceled', common_norm=False, palette='Set1')
    plt.title('Stratified Density: Lead Time by Cancellation')

    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df[df['ADR'] < 500], x='ADR', hue='IsCanceled', common_norm=False, palette='Set1')
    plt.title('Stratified Density: ADR by Cancellation')

    plt.tight_layout()
    plt.savefig('outputs/stratified_density_plots.png')
    plt.close()

# Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

# ==========================================
# 6. ENHANCED ADDITIONS (CARDINALITY, LEAKAGE & IMPUTATION)
# ==========================================
print("\n--- [6] Enhanced Additions: Leakage & Imputation Audit ---")

# 6.1 Low-Variance & Cardinality Filter
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
print(f"Constant/Zero-variance columns to drop: {constant_cols}")

# 6.2 Data Leakage Audit (Section 16 Requirement)
leakage_candidates = ['ReservationStatus', 'ReservationStatusDate', 'BookingChanges']
present_leakage = [col for col in leakage_candidates if col in df.columns]
print(f"Identified potential data leakage columns: {present_leakage}")

# 6.3 Missingness Imputation Plan for Person B
print("\nRecommended Imputation Strategy for Person B:")
print(" - Agent / Company: Impute missing values with 'None' string (indicates direct booking).")
print(" - Country: Impute missing values with Mode or 'Unknown'.")
print(" - Children: Impute missing numerical values with 0.")

print("\n--- Execution Complete! Check 'outputs/' for saved visualizations. ---")