# Script per unire più file bilanciati e suddivisione in train/validation/test

import pandas as pd
from sklearn.model_selection import train_test_split

# Lista dei file bilanciati da unire
balanced_files = [
    'Friday-02-03-2018_rfe.csv',
    'Friday-16-02-2018_rfe.csv',
    'Wednesday-14-02-2018_rfe.csv',
    'Tuesday-20-02-2018_balanced.csv',
    'Thursday-15-02-2018_balanced.csv',
    'Thursday-22-02-2018_balanced.csv'
]

# Caricamento e unione dei file
dfs = [pd.read_csv(file) for file in balanced_files]
df_merged = pd.concat(dfs, ignore_index=True)

# Salvataggio del dataset unificato
df_merged.to_csv('merged.csv', index=False)

# Suddivisione stratificata train (70%), validation (15%), test (15%)
X = df_merged.drop(columns=['Label'])
y = df_merged['Label']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# Salvataggio dei file suddivisi
pd.DataFrame(X_train).assign(Label=y_train).to_csv('train.csv', index=False)
pd.DataFrame(X_val).assign(Label=y_val).to_csv('validation.csv', index=False)
pd.DataFrame(X_test).assign(Label=y_test).to_csv('test.csv', index=False)
