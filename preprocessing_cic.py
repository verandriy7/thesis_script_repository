# Script di preprocessing per CSE-CIC-IDS2018
# Pulizia, normalizzazione, encoding, bilanciamento tramite SMOTE

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Lista dei file CSV selezionati dopo analisi esplorativa e cleaning
selected_files = [
    'Friday-02-03-2018_preprocessed.csv',
    'Friday-16-02-2018_preprocessed.csv',
    'Wednesday-14-02-2018_preprocessed.csv',
    'Tuesday-20-02-2018_preprocessed.csv',
    'Thursday-15-02-2018_preprocessed.csv',
    'Thursday-22-02-2018_preprocessed.csv'
]
# Caricamento e concatenazione dei file selezionati
dataframes = [pd.read_csv(f) for f in selected_files]
df = pd.concat(dataframes, ignore_index=True)

# Separazione feature e target
X = df.drop(columns=['Label'])
y = df['Label']

# Normalizzazione con Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Bilanciamento con SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
# Suddivisione in train/validation/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Salvataggio dei file finali
pd.DataFrame(X_train).assign(Label=y_train).to_csv('train.csv', index=False)
pd.DataFrame(X_val).assign(Label=y_val).to_csv('validation.csv', index=False)
pd.DataFrame(X_test).assign(Label=y_test).to_csv('test.csv', index=False)
pd.DataFrame(X_resampled).assign(Label=y_resampled).to_csv('merged.csv', index=False)
