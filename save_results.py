# Script per salvare le metriche di valutazione (Accuracy, Precision, Recall, F1-score) in un file CSV per confronto tra modelli

import pandas as pd

def save_results(algorithm_name, dataset_name, accuracy, precision, recall, f1, output_file):
    # Crea un dizionario con i risultati
    results = {
        'Algorithm': [algorithm_name],
        'Dataset': [dataset_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    }
    df = pd.DataFrame(results)

    # Salva o aggiunge i risultati al CSV
    try:
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    except FileNotFoundError:
        df_combined = df

    df_combined.to_csv(output_file, index=False)
