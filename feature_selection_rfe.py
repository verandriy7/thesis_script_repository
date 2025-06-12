# Script per la selezione delle feature tramite RFE con Random Forest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Caricamento del file bilanciato
input_file = 'Friday-16-02-2018_preprocessed.csv'
output_file = 'Friday-16-02-2018_rfe.csv'

# Caricamento del dataset
df = pd.read_csv(input_file)

# Separazione delle feature e della label
X = df.drop(columns=['Label'])
y = df['Label']

# Inizializzazione del classificatore
estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Selezione ricorsiva delle migliori feature (es. 30)
selector = RFE(estimator=estimator, n_features_to_select=30, step=1)
X_selected = selector.fit_transform(X, y)

# Recupero dei nomi delle colonne selezionate
selected_features = X.columns[selector.support_]

# Creazione nuovo DataFrame
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected['Label'] = y.values
# Salvataggio del file ridotto
df_selected.to_csv(output_file, index=False)
