# Script per il preprocessing di UNSW-NB15
# Pulizia, normalizzazione, bilanciamento SMOTE, split train/val/test

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Caricamento del dataset unificato e pulito
df = pd.read_csv('UNSW-NB15_clean_merged.csv')

# Rimozione di colonne non informative, se presenti
colonne_da_rimuovere = ['id', 'attack_cat']  # 'attack_cat' non viene usata per la classificazione binaria
df = df.drop(columns=[col for col in colonne_da_rimuovere if col in df.columns])

# Separazione delle feature e dell’etichetta
X = df.drop(columns=['label'])
y = df['label']

# Normalizzazione delle feature numeriche con Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Bilanciamento delle classi con SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Suddivisione in train (70%), validation (15%) e test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Salvataggio dei file preprocessati
pd.DataFrame(X_train).assign(label=y_train).to_csv('UNSW-NB15_train.csv', index=False)
pd.DataFrame(X_val).assign(label=y_val).to_csv('UNSW-NB15_val.csv', index=False)
pd.DataFrame(X_test).assign(label=y_test).to_csv('UNSW-NB15_test.csv', index=False)
pd.DataFrame(X_resampled).assign(label=y_resampled).to_csv('UNSW-NB15_final_preprocessed.csv', index=False)
