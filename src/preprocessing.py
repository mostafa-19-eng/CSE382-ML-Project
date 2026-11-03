
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

def encode_month(X):
    month_numbers = pd.to_datetime(
        np.asarray(X).ravel(), format='%B'
    ).month

    return np.column_stack([
        np.sin(2 * np.pi * (month_numbers - 1) / 12),
        np.cos(2 * np.pi * (month_numbers - 1) / 12)
    ])

# Numerical Features
NUMERICAL_FEATURES = [
    'LeadTime',
    'ArrivalDateYear',
    'ArrivalDateWeekNumber',
    'ArrivalDateDayOfMonth',
    'StaysInWeekendNights',
    'StaysInWeekNights',
    'Adults',
    'Children',
    'Babies',
    'PreviousCancellations',
    'PreviousBookingsNotCanceled',
    
    'DaysInWaitingList',
    'ADR',
    'RequiredCarParkingSpaces',
    'TotalOfSpecialRequests',
    'TotalGuests'
]


# Categorical Features
CATEGORICAL_FEATURES = [
    'Meal',
    'Country',
    'MarketSegment',
    'DistributionChannel',
    'ReservedRoomType',
    
    'DepositType',
    'CustomerType',
    'Agent',
    'Company'
]


# Binary Features
BINARY_FEATURES = [
    'IsRepeatedGuest'
    
]


def build_preprocessor():

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    month_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cyclical', FunctionTransformer(encode_month, validate=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ('bin', binary_pipeline, BINARY_FEATURES),
        ('month', month_pipeline, ['ArrivalDateMonth'])
    ])

    return preprocessor



 
if __name__ == "__main__":

    df = pd.read_csv('data/H2_cleaned.csv')

    X = df.drop(columns=['IsCanceled'])
    y = df['IsCanceled']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing completed successfully!")
    print("Training features shape:", X_train_processed.shape)
    print("Testing features shape:", X_test_processed.shape)

    if issparse(X_train_processed):
        train_data = X_train_processed.data
        test_data = X_test_processed.data
    else:
        train_data = X_train_processed.ravel()
        test_data = X_test_processed.ravel()

    print("Training contains NaN:", np.isnan(train_data).any())
    print("Testing contains NaN:", np.isnan(test_data).any())

    print("Training contains infinity:", np.isinf(train_data).any())
    print("Testing contains infinity:", np.isinf(test_data).any())


    

  
