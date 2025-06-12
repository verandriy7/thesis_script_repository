# Script per applicare SMOTE a un file CSV del dataset CSE-CIC-IDS2018
# Utilizzato per i file con forte sbilanciamento delle classi

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# File di input e output
input_file = 'Tuesday-20-02-2018_rfe.csv'
output_file = 'Tuesday-20-02-2018_balanced.csv'

# Caricamento del file pulito
df = pd.read_csv(input_file)

# Separazione delle feature e della variabile target
X = df.drop(columns=['Label'])
y = df['Label']

# Normalizzazione delle feature
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Applicazione di SMOTE per bilanciare il dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Salvataggio del dataset bilanciato
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Label'] = y_resampled
df_resampled.to_csv(output_file, index=False)
