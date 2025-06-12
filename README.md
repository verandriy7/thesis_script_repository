# Script Python per il preprocessing e l'addestramento dei Modelli

Questa repository contiene una raccolta di esempi di script Python sviluppati e utilizzati nell’ambito della tesi di laurea triennale:
**"IDENTIFICAZIONE DI ATTACCHI WEB CON INTELLIGENZA ARTIFICIALE"** di Veronica Tavazzi (12247A).

Gli script sono stati scritti per implementare le principali fasi di una pipeline di rilevamento delle intrusioni basata su algoritmi supervisionati di apprendimento automatico. I file inclusi coprono:
- **Preprocessing dei dataset** (normalizzazione, bilanciamento con SMOTE, suddivisione stratificata).
- **Selezione delle feature** (con tecnica RFE).
- **Addestramento dei modelli** (Random Forest, SVM, Deep Neural Network).
- **Valutazione dei modelli** (metriche, curve ROC, matrici di confusione).
- **Analisi esplorativa e gestione dei file** (ad esempio, chunking e unione dei CSV).

Tutti gli script rappresentano esempi **semplificati e documentati** dei processi reali impiegati, con l’obiettivo di fornire una panoramica chiara e replicabile delle operazioni descritte nel documento di tesi.

## Struttura della repository

| File                          | Descrizione                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `preprocessing_unsw.py`      | Preprocessing del dataset UNSW-NB15: pulizia, normalizzazione, SMOTE, split |
| `preprocessing_cic.py`       | Preprocessing del dataset CSE-CIC-IDS2018                                  |
| `feature_selection_rfe.py`   | Selezione automatica delle feature tramite RFE                             |
| `smote_balancing.py`         | Bilanciamento delle classi con SMOTE per CSE-CIC-IDS2018                   |
| `merge_split.py`             | Merge dei file bilanciati di CSE-CIC-IDS2018 e suddivisione stratificata   |
| `train_model.py`             | Addestramento dei modelli: Random Forest, SVM, Deep Neural Network         |
| `evaluate_model.py`          | Valutazione dei modelli e calcolo delle metriche                           |
| `chunk_processing_unsw.py`   | Preprocessing del dataset UNSW-NB15 a chunk per ottimizzare l’uso della RAM|
| `analyze_label_distribution.py` | Analisi della distribuzione delle etichette nei dataset                |
| `explore_dataset.py`         | Esplorazione iniziale dei dataset: nulli, distribuzioni, riepilogo         |
| `generate_confusion_matrix.py` | Generazione della matrice di confusione da `y_true` e `y_pred`          |
| `plot_confusion_matrix.py`   | Visualizzazione grafica delle matrici di confusione con Seaborn            |
| `normalize_labels.py`        | Normalizzazione delle etichette nei file del dataset CSE-CIC-IDS2018       |
| `save_results.py`            | Salvataggio incrementale dei risultati in formato CSV                      |

## Note
Gli script presenti in questa repository non corrispondono esattamente al codice utilizzato durante la sperimentazione, ma rappresentano versioni **pulite, semplificate e pienamente funzionanti** che riflettono fedelmente la logica e la sequenza delle operazioni effettivamente svolte.
Durante il lavoro di tesi, le attività di sviluppo si sono basate su un approccio **iterativo e sperimentale (trial-and-error)**, che ha prodotto numerose varianti dei codici. Per garantire chiarezza, leggibilità e facilità di consultazione, gli script sono stati successivamente **ricostruiti in forma ordinata e documentata**, mantenendo intatti i principi metodologici e le scelte tecniche descritte nella tesi.
Tutti i file inclusi hanno quindi lo scopo di **illustrare in modo chiaro e replicabile** i principali processi implementativi adottati, offrendo una panoramica coerente con quanto riportato nei capitoli metodologici e sperimentali del documento di tesi.

