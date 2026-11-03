import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(data_path='../data/H2.csv'):
    df = pd.read_csv(data_path)
    drop_cols = ['ReservationStatus', 'ReservationStatusDate']
    X = df.drop(columns=['IsCanceled'] + drop_cols)
    y = df['IsCanceled']
    X = pd.get_dummies(X)
    X = X.fillna(X.median())
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
