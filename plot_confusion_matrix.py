# Script per generaee e salvare le matrici di confusione per i modelli

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Caricamento del modello
model = joblib.load("random_forest_model.pkl")

# Caricamento dei dati di test
df_test = pd.read_csv("UNSW-NB15_test.csv")
X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

# Predizioni
y_pred = model.predict(X_test)

# Calcolo della matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# Visualizzazione con heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione - Random Forest")
plt.tight_layout()
# Salvataggio
plt.savefig("confusion_matrix_rf.png")
