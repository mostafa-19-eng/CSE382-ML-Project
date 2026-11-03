import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from preprocessing_personD import load_and_preprocess, split_data

def train_model(X_train, y_train):
    warnings.filterwarnings('ignore')
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def cross_validate(clf, X_train, y_train):
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual folds: {cv_scores.round(4)}")

if __name__ == '__main__':
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_model(X_train, y_train)
    cross_validate(clf, X_train, y_train)
