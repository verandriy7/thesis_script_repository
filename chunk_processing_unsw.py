# Elaborazione RAM-friendly a chunk di un file CSV

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Parametri
input_file = "UNSW-NB15_clean_merged.csv"
output_file = "UNSW-NB15_final_preprocessed.csv"
chunk_size = 100000

# Inizializza lo scaler
scaler = MinMaxScaler()

# Calcolo globale dei min e max (necessario per MinMaxScaler)
print("Calcolo statistico iniziale...")
full_df = pd.read_csv(input_file)
X = full_df.drop(columns=["label"])
scaler.fit(X)
del full_df  # Libera la RAM

# Elaborazione a chunk con normalizzazione e salvataggio incrementale
print("Elaborazione a chunk...")
reader = pd.read_csv(input_file, chunksize=chunk_size)
for i, chunk in enumerate(tqdm(reader)):
    X_chunk = chunk.drop(columns=["label"])
    y_chunk = chunk["label"]
    X_scaled = scaler.transform(X_chunk)
    df_scaled = pd.DataFrame(X_scaled, columns=X_chunk.columns)
    df_scaled["label"] = y_chunk.values
    mode = "w" if i == 0 else "a"
    header = i == 0
    df_scaled.to_csv(output_file, mode=mode, header=header, index=False)
print("Elaborazione completata. Output salvato in:", output_file)
