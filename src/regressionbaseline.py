"""CSE382 Milestone 1: reproducible, untuned ADR regression baseline (Person C).

Run from the project root:
    python -m src.regression_baseline

The held-out final test portion is NEVER evaluated here. This experiment uses a
conservative booking-creation feature set and prevents identical feature
vectors from appearing across train/validation/test splits. Feature and cleaning
decisions should be confirmed by the team before reporting as final.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Use Person B's existing month transformer, keeping the rest of Person B's
# preprocessing approach while selecting a different (leakage-safe) ADR feature list.
from src.preprocessing import encode_month

TARGET = 'ADR'
RANDOM_SEED = 42
DATA_PATH = Path('data/H2_cleaned.csv')
OUTPUT_PATH = Path('outputs/regressionbaseline.json')

NUMERICAL = [
    'LeadTime', 'ArrivalDateYear', 'ArrivalDateWeekNumber',
    'ArrivalDateDayOfMonth', 'StaysInWeekendNights', 'StaysInWeekNights',
    'Adults', 'Children', 'Babies', 'PreviousCancellations',
    'PreviousBookingsNotCanceled',
]
CATEGORICAL = [
    'Meal', 'Country', 'MarketSegment', 'DistributionChannel',
    'ReservedRoomType', 'DepositType', 'CustomerType',
]
BINARY = ['IsRepeatedGuest']
CYCLICAL = ['ArrivalDateMonth']
FEATURES = NUMERICAL + CATEGORICAL + BINARY + CYCLICAL


def build_adr_preprocessor():
    """Task-specific equivalent of Person B's ColumnTransformer (exclude ADR)."""
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), NUMERICAL),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL),
        ('bin', SimpleImputer(strategy='most_frequent'), BINARY),
        ('month', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('cyclical', FunctionTransformer(encode_month, validate=False)),
        ]), CYCLICAL),
    ])


def run():
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required dataset columns: {missing}')
    if df[TARGET].isna().any():
        raise ValueError('ADR contains missing labels. Review cleaning first.')

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Group solely by the 20 input attributes, not the ADR target. This avoids
    # identical X appearing on both sides of a split without deleting bookings.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    holdout_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    dev_idx, test_idx = next(holdout_split.split(X, y, groups))

    # Within dev, reserve validation. 20% here applies to groups, not rows.
    valid_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_local, valid_local = next(valid_split.split(X.iloc[dev_idx], y.iloc[dev_idx], groups[dev_idx]))
    train_idx, valid_idx = dev_idx[train_local], dev_idx[valid_local]

    assert len(set(groups[train_idx]) & set(groups[valid_idx])) == 0
    assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0
    assert len(set(groups[valid_idx]) & set(groups[test_idx])) == 0
    assert len(set(train_idx) | set(valid_idx) | set(test_idx)) == len(df)

    model = Pipeline([
        ('preprocessor', build_adr_preprocessor()),
        ('regressor', LinearRegression()),  # untuned baseline
    ])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[valid_idx])
    mse = mean_squared_error(y.iloc[valid_idx], pred)
    results = {
        'status': 'provisional; cleaning and feature availability require team sign-off',
        'task': 'ADR regression, at booking creation',
        'dataset': str(DATA_PATH),
        'rows': {'total': len(df), 'train': len(train_idx), 'validation': len(valid_idx),
                 'reserved_final_test_not_evaluated': len(test_idx)},
        'split': {'method': 'GroupShuffleSplit on input-feature row hashes',
                  'seed': RANDOM_SEED,
                  'group_overlap_between_splits': 0,
                  'group_test_fraction': 0.20,
                  'group_validation_fraction_of_development': 0.20},
        'features': FEATURES,
        'model': 'LinearRegression (untuned)',
        'validation_metrics': {
            'MAE': round(float(mean_absolute_error(y.iloc[valid_idx], pred)), 4),
            'MSE': round(float(mse), 4),
            'RMSE': round(float(np.sqrt(mse)), 4),
            'R2': round(float(r2_score(y.iloc[valid_idx], pred)), 4),
        },
        'final_test_evaluated': False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == '__main__':
    run()
