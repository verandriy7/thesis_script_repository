# Script per normalizzare le etichette del dataset CSE-CIC-IDS2018
# Trasforma etichette in versione binaria: 0=BENIGN, 1=ATTACK

import pandas as pd
import os

# Directory contenente i file CSV da normalizzare
input_dir = "./"
output_dir = "./FASE1_Uniformazione"

# Mappatura delle etichette personalizzata
def normalize_label(label):
    return 0 if label.strip().upper() == "BENIGN" else 1

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Elenco dei file da processare
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# Elaborazione di ciascun file
for file in csv_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    if "Label" in df.columns:
        df["Label"] = df["Label"].apply(normalize_label)
        output_path = os.path.join(output_dir, file.replace(".csv", "_clean.csv"))
        df.to_csv(output_path, index=False)
        print(f"File normalizzato salvato in: {output_path}")
    else:
        print(f"Colonna 'Label' non trovata in {file}, file ignorato.")
