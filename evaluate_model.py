# Script per la valutazione dei modelli addestrati su test.csv
# Calcola metriche e genera file di report

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# Caricamento test set
df = pd.read_csv('test.csv')
X = df.drop(columns=['Label'])
y = df['Label']

# Funzione per valutazione e salvataggio
def evaluate_model(model, name, is_dnn=False):
    if is_dnn:
        y_pred_proba = model.predict(X).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    report_file = f"{name}_results.txt"
    with open(report_file, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name}")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"roc_curve_{name}.png")
    plt.close()

# Valutazione RF
rf = joblib.load("random_forest_model.pkl")
evaluate_model(rf, "rf")

# Valutazione SVM
svm = joblib.load("svm_model.pkl")
evaluate_model(svm, "svm")

# Valutazione DNN
dnn = load_model("dnn_model.h5")
evaluate_model(dnn, "dnn", is_dnn=True)
