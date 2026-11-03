# Data Preprocessing and Pipelines

## 1. Overview

This module is responsible for preparing the hotel booking dataset for machine learning models.

It provides a reusable preprocessing pipeline using Scikit-learn's `ColumnTransformer` and `Pipeline`.

The input dataset is `data/H2_cleaned.csv`, prepared by the data cleaning stage.

The target variable is `IsCanceled`.

## 2. Feature Selection

Features are divided into four groups:

### Numerical Features

The following numerical features are processed using median imputation and standardization:

- LeadTime
- ArrivalDateYear
- ArrivalDateWeekNumber
- ArrivalDateDayOfMonth
- StaysInWeekendNights
- StaysInWeekNights
- Adults
- Children
- Babies
- PreviousCancellations
- PreviousBookingsNotCanceled
- DaysInWaitingList
- ADR
- RequiredCarParkingSpaces
- TotalOfSpecialRequests
- TotalGuests

### Categorical Features

The following categorical features are processed using most-frequent imputation and One-Hot Encoding:

- Meal
- Country
- MarketSegment
- DistributionChannel
- ReservedRoomType
- DepositType
- CustomerType
- Agent
- Company

Agent and Company are treated as categorical features because they represent identifiers rather than continuous numerical measurements.

`OneHotEncoder(handle_unknown='ignore')` is used to handle categories that were not present during training.

### Binary Features

- IsRepeatedGuest

Missing values are handled using the most frequent value.

### Cyclical Features

`ArrivalDateMonth` is transformed using sine and cosine encoding.

This preserves the cyclical relationship between months, where December and January are close to each other.

The transformation uses a period of 12 months.

## 3. Data Leakage Prevention

The prediction task assumes that cancellation is predicted at the time of booking creation.

Based on this assumption, the following features are excluded:

- ReservationStatus
- ReservationStatusDate
- BookingChanges
- AssignedRoomType
- RoomMismatch

`ReservationStatus` and `ReservationStatusDate` are removed during data cleaning.

`BookingChanges` may contain information about actions taken after booking creation.

`AssignedRoomType` represents the room actually assigned, which may not be known when the booking is created.

`RoomMismatch` is derived from the assigned room and is therefore also excluded.

`ReservedRoomType` is retained because it is known when the booking is created.

These exclusions should be reviewed if the prediction time changes.

## 4. Preprocessing Techniques

| Feature Type | Technique |
|---|---|
| Numerical | Median Imputation |
| Numerical | Standard Scaling |
| Categorical | Most Frequent Imputation |
| Categorical | One-Hot Encoding |
| Binary | Most Frequent Imputation |
| Month | Cyclical Encoding |

All transformations are implemented using Scikit-learn pipelines.

## 5. Train-Test Splitting

The cleaned dataset is divided into:

- Training set: 80%
- Testing set: 20%

The split uses:

- `random_state=42`
- `stratify=y`

Stratification preserves the target class distribution in both sets.

The preprocessor is fitted only on the training data and then applied to the testing data.

## 6. Implementation

The main preprocessing function is:

`build_preprocessor()`

It returns a reusable Scikit-learn `ColumnTransformer`.

It can be integrated into a machine learning pipeline together with a classifier.

Example:

```python
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

model_pipeline = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', your_classifier)
])
```

The classifier is supplied by the model development stage.

Using a single pipeline ensures that preprocessing transformations are fitted only on the training data during model training and cross-validation.

## 7. Testing and Validation

The preprocessing module was tested using the cleaned dataset.

The resulting dimensions were:

| Dataset | Shape |
|---|---|
| Training | (62472, 606) |
| Testing | (15619, 606) |

Additional validation confirmed:

- No NaN values after preprocessing.
- No infinite values after preprocessing.
- Training and testing have the same number of features.
- Unknown categorical values are handled without errors.

## 8. Main Implementation File

`src/preprocessing.py`

The module contains:

- Feature group definitions.
- Month cyclical encoding.
- Numerical, categorical, and binary pipelines.
- The `build_preprocessor()` function.
- A standalone preprocessing test.
