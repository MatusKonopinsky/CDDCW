import pandas as pd
import os
import numpy as np

from utils.data_preprocesing import read_jigsaw_tfidf_data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.arff2pandas import load as load_arff # Priamy import z vášho súboru

# -----------------------------------------------------------------------------
# KONFIGURÁCIA
# -----------------------------------------------------------------------------


# Adresár, kde sa nachádzajú pôvodné datasety
SOURCE_DATA_DIR = 'DDCW/data/real/'

# Adresár, kam sa uložia spracované CSV súbory
OUTPUT_DATA_DIR = 'DDCW/data/real/'

# --- Konfigurácia pre Jigsaw ---
JIGSAW_CONFIG = {
    "source_file": os.path.join(SOURCE_DATA_DIR, 'toxic_train.csv'),
    "output_file": os.path.join(OUTPUT_DATA_DIR, 'jigsaw_tfidf.csv'),
    "sample_size": 100000,
    "max_features": 2000,
    "target_col": 'toxic'
}

FILES_TO_PREPROCESS = {
    "shuttle_numeric.csv": "shuttle.arff",
    "kdd_numeric.csv": "kddcup.csv",
    # Sem môžete pridať ďalšie súbory, ktoré potrebujú konverziu
}


# -----------------------------------------------------------------------------
# LOGIKA SPRACOVANIA
# -----------------------------------------------------------------------------

def process_jigsaw():
    """
    Spracuje Jigsaw dataset pomocou TF-IDF a uloží ho ako CSV bez hlavičky.
    """
    print(f"\n--- Spracovanie Jigsaw datasetu ---")
    config = JIGSAW_CONFIG
    
    if not os.path.exists(config["source_file"]):
        print(f"Chyba: Pôvodný súbor Jigsaw nebol nájdený: {config['source_file']}")
        return

    # Vaša funkcia vracia X a y ako pandas DataFrames
    _, X, y = read_jigsaw_tfidf_data(
        filename=config["source_file"],
        sample_size=config["sample_size"],
        max_features=config["max_features"],
        target_col=config["target_col"]
    )
    
    # Spojíme X a y do jedného numpy poľa
    # X.values a y.values prevedú DataFrame na numpy pole
    processed_data_np = np.concatenate((X.values, y.values.reshape(-1, 1)), axis=1)
    
    try:
        # Uloženie numpy poľa do CSV súboru
        np.savetxt(config["output_file"], processed_data_np, delimiter=',', fmt='%.8f')
        
        print(f"Úspech! Spracovaný Jigsaw dataset uložený do: {config['output_file']}")
        print(f"   Počet vzoriek: {processed_data_np.shape[0]}")
        print(f"   Počet príznakov: {processed_data_np.shape[1] - 1}")
    except Exception as e:
        print(f"Chyba pri ukladaní Jigsaw CSV: {e}")


def preprocess_and_save(source_path, output_path):
    """
    Univerzálna funkcia na načítanie, predspracovanie a uloženie datasetu.
    Zvláda .arff aj .csv a ukladá ako čisté CSV bez hlavičky.
    """
    # Načítanie na základe koncovky súboru
    if source_path.endswith('.arff'):
        df = load_arff(source_path)
    else: # predpokladáme .csv
        df = pd.read_csv(source_path)
    
    df.dropna(inplace=True)
    if df.empty:
        print(f"  Varovanie: Súbor {os.path.basename(source_path)} je po odstránení NaN prázdny.")
        return

    X_df = df.iloc[:, :-1]
    y_series = df.iloc[:, -1] # Ako séria pre jednoduchšiu manipuláciu

    # Konverzia kategorických príznakov
    le = LabelEncoder()
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = le.fit_transform(X_df[col])

    # Normalizácia
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Spracovanie cieľovej premennej
    if y_series.dtype == 'object':
        y = le.fit_transform(y_series)
    else:
        y = y_series.values
        
    y = y.astype(int)

    # Spojenie do finálneho numpy poľa
    final_data = np.concatenate((X_scaled, y.reshape(-1, 1)), axis=1)

    # Uloženie ako CSV bez hlavičky
    np.savetxt(output_path, final_data, delimiter=',', fmt='%.8f')
    print(f"  Úspech! Uložené do: {output_path} ({final_data.shape[0]} riadkov)")

if __name__ == "__main__":
    # Vytvorenie adresárov, ak neexistujú
    if not os.path.exists(SOURCE_DATA_DIR):
        os.makedirs(SOURCE_DATA_DIR)
        print(f"Vytvorený adresár pre zdrojové dáta: {SOURCE_DATA_DIR}")
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
        print(f"Vytvorený adresár pre spracované dáta: {OUTPUT_DATA_DIR}")

    # Spustenie jednotlivých úloh
    process_jigsaw()
     # Spustenie spracovania pre ostatné súbory
    print("\n--- Konverzia a predspracovanie ostatných datasetov ---")
    for output_file, source_file in FILES_TO_PREPROCESS.items():
        source_path = os.path.join(SOURCE_DATA_DIR, source_file)
        output_path = os.path.join(OUTPUT_DATA_DIR, output_file)
        print(f"Spracúva sa: {source_file}")
        if os.path.exists(source_path):
            preprocess_and_save(source_path, output_path)
        else:
            print(f"  Chyba: Zdrojový súbor nebol nájdený: {source_path}")

    print("\nSpracovanie datasetov dokončené.")