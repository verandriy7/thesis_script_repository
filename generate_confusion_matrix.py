# Script per generare e salvare una matrice di confusione da CSV

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Lettura dei file CSV contenenti le etichette vere e predette
y_true = pd.read_csv("y_test.csv")["Label"]
y_pred = pd.read_csv("y_pred.csv")["Pred"]

# Calcolo della matrice di confusione
cm = confusion_matrix(y_true, y_pred)

# Visualizzazione della matrice di confusione
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice di Confusione - Modello Finale")
plt.savefig("confusion_matrix.png")
plt.show()
