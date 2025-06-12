# Script di esplorazione iniziale del dataset (dimensioni, etichette, distribuzione, valori nulli)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del dataset
df = pd.read_csv("UNSW-NB15_clean_merged.csv")

# Visualizzazione delle dimensioni del dataset
print("Dimensioni del dataset:", df.shape)

# Distribuzione delle classi (etichetta 'label' o 'Label')
label_col = 'label' if 'label' in df.columns else 'Label'
print("\nDistribuzione delle etichette:")
print(df[label_col].value_counts())

# Percentuale di valori nulli per colonna
print("\nValori nulli per colonna (in percentuale):")
print((df.isnull().sum() / len(df)) * 100)

# Istogramma delle etichette
sns.countplot(x=label_col, data=df)
plt.title("Distribuzione delle classi")
plt.xlabel("Etichetta")
plt.ylabel("Numero di campioni")
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()

# Heatmap delle correlazioni (solo per numeriche)
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap="coolwarm", annot=False)
plt.title("Matrice di correlazione")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()
