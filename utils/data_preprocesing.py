import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from . import arff2pandas

def read_elec_norm_data(filename):
    """
    Function read elec_norm dataset and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    dataset = pd.read_csv(filename)

    X = pd.DataFrame(dataset.iloc[:, 1:-1])
    y = pd.DataFrame(dataset.iloc[:, -1])

    scaler = MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X))

    label = np.unique(y)
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)
    y = pd.DataFrame(y)
    y.astype('int32')

    data = pd.concat([X, y], axis=1)
    data = data.values

    return data, X, y

def read_kdd_data_multilable(filename):
    """
    Function read kdd kdd multilable data and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                 "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
                 "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                 "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "label"]

    data_10percent = pd.read_csv(filename, names=col_names, header=None)

    X = pd.DataFrame(data_10percent.iloc[:, :-1])
    y = pd.DataFrame(data_10percent.iloc[:, -1])

    le = LabelEncoder()
    for col in X.columns.values:
        if X[ col ].dtypes == 'object':
            le.fit(X[ col ])
            X[ col ] = le.transform(X[ col ])

    y = pd.DataFrame(le.fit_transform(y))
    y.astype('int32')

    scaler = MinMaxScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X))
    data = pd.concat([ X_sc, y ], axis=1)
    data = data.values

    return data, X_sc, y


def read_data_arff(filename):
    """
    Function read arff data and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    with open(filename) as f:
        df = arff2pandas.load(f)

    X = pd.DataFrame(df.iloc[:, :-1])
    y = pd.DataFrame(df.iloc[:, -1])

    le = LabelEncoder()
    for col in X.columns.values:
        if X[ col ].dtypes == 'object':
            le.fit(X[ col ])
            X[ col ] = le.transform(X[ col ])

    y = pd.DataFrame(le.fit_transform(y))
    y.astype('int32')

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    data = pd.concat([ X, y ], axis=1)
    data = data.values

    return data, X, y

def read_data_csv(filename):
    """
    Načíta CSV súbor (s hlavičkou alebo bez) a vráti dáta pre scikit-multiflow.
    Automaticky preskočí hlavičku, ak existuje.
    Očakáva, že posledný stĺpec je cieľová premenná.
    """
    try:
        # Skúsi načítať dáta a použiť prvý riadok ako hlavičku.
        # Ak prvý riadok neobsahuje číselné dáta, toto je správny prístup.
        df = pd.read_csv(filename, header=0)
    except Exception:
        # Ak to zlyhá (napr. prvý riadok má zlý počet stĺpcov),
        # skúsime to načítať bez hlavičky.
        df = pd.read_csv(filename, header=None)

    # Odstránenie riadkov s akoukoľvek chýbajúcou hodnotou
    df.dropna(inplace=True)
    
    # Prevod na numpy pole
    data_np = df.values
    
    if data_np.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Rozdelenie na X a y
    X_np = data_np[:, :-1]
    y_np = data_np[:, -1]
    
    # Prevod na správne dátové typy, s ošetrením chýb
    try:
        X_np = X_np.astype(float)
        y_np = y_np.astype(int)
    except ValueError:
        # Ak konverzia zlyhá, znamená to, že v dátach sú stále nečíselné hodnoty
        # Pravdepodobne problém s kategorickými premennými
        print(f"Varovanie: V súbore {filename} boli nájdené nečíselné hodnoty aj po načítaní. Skúste ho predspracovať na čisto číselný formát.")
        # Vrátime prázdne polia, aby experiment preskočil tento dataset
        return np.array([]), np.array([]), np.array([])
    
    # data_np pre kompatibilitu
    all_data_np = np.concatenate((X_np, y_np.reshape(-1, 1)), axis=1)

    return all_data_np, X_np, y_np

def read_jigsaw_tfidf_data(filename, sample_size=50000, max_features=1000, target_col='toxic'):
    """
    Načíta Jigsaw dataset, predspracuje text pomocou TF-IDF a vráti dáta pre scikit-multiflow.

    :param filename: Cesta k Jigsaw train.csv súboru.
    :param sample_size: Počet náhodne vybraných vzoriek na spracovanie.
    :param max_features: Maximálny počet "slov" (príznakov) pre TF-IDF.
    :param target_col: Názov stĺpca, ktorý sa použije ako cieľová premenná (napr. 'toxic', 'severe_toxic', atď.).
    :return: data, X, y
    """
    print(f"Spracúva sa Jigsaw dataset z {filename}...")
    
    # Načítanie náhodnej vzorky pre rýchlejšie spracovanie
    df = pd.read_csv(filename)

    df.dropna(subset=['comment_text', target_col], inplace=True)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    texts = df['comment_text'].fillna('missing') # Vyplnenie chýbajúcich komentárov
    y_labels = df[target_col].values

    # Vektorizácia textu pomocou TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_vectors = vectorizer.fit_transform(texts).toarray()
    
    # Prevod na DataFrame pre jednoduchšiu manipuláciu
    X = pd.DataFrame(X_vectors)
    y = pd.DataFrame(y_labels)
    y.astype('int32')

    # Spojenie do 'data' matice, ako v ostatných funkciách
    data = np.concatenate([X.values, y.values.reshape(-1, 1)], axis=1)

    print("Jigsaw dataset spracovaný.")
    return data, X, y

def read_clean_csv(filename):
    """
    Načíta predspracovaný CSV súbor bez hlavičky.
    """
    try:
        data_np = np.loadtxt(filename, delimiter=',')
    except ValueError:
        # záložný plán, ak má súbor hlavičku
        data_np = np.loadtxt(filename, delimiter=',', skiprows=1)

    if data_np.ndim == 1: data_np = data_np.reshape(1, -1)

    X_np = data_np[:, :-1]
    y_np = data_np[:, -1].astype(int)
    
    return data_np, X_np, y_np