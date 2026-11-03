import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from preprocessing_personD import load_and_preprocess, split_data
from train_personD import train_model

def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=['Not Canceled', 'Canceled']))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Canceled', 'Canceled'])
    disp.plot(cmap='Blues')
    plt.title('Logistic Regression – Confusion Matrix (Baseline)')
    plt.tight_layout()
    plt.savefig('../outputs/phase1/confusion_matrix_baseline.png', dpi=150)
    plt.show()

    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Baseline Classifier')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../outputs/phase1/roc_curve_baseline.png', dpi=150)
    plt.show()
    print(f"ROC-AUC: {auc:.4f}")

if __name__ == '__main__':
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_model(X_train, y_train)
    evaluate(clf, X_test, y_test)
