# Analisi della distribuzione delle etichette in un file CSV
# Generazione di un grafico

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento del file
df = pd.read_csv('train.csv')

# Calcolo della distribuzione delle etichette
label_counts = df['Label'].value_counts().sort_index()

# Visualizzazione
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette='viridis')
plt.title('Distribuzione delle etichette nel dataset')
plt.xlabel('Etichetta')
plt.ylabel('Numero di istanze')
plt.tight_layout()
plt.savefig('label_distribution.png')
plt.show()
