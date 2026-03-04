"""
Generátor syntetických dát pre streamové experimenty.

Cieľ:
- vygenerovať dáta zo skmultiflow generátorov
- zabezpečiť presný pomer tried s presnými počtami
- uložiť CSV do ./data/synthetic_imbalanced/

- Aktuálne generuje BINÁRNE datasety (trieda 0 a 1).
  Pre multi-class treba upraviť:
  - n_classes generátorov
  - sampling_strategy v make_imbalance na viac tried
"""

import os
import numpy as np
import pandas as pd

from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import RandomRBFGeneratorDrift

from imblearn.datasets import make_imbalance  # undersampling helper


# ============================================================
# 1) CESTY + KONFIGURÁCIA
# ============================================================

DATA_DIR = "./data/synthetic_imbalanced/"
os.makedirs(DATA_DIR, exist_ok=True)

# koľko vzoriek v cieľovom datasete
N_SAMPLES = 50_000

# pomer majority (0) / minority (1)
MAJ_RATIO = 0.90

# Presné počty tried
imbalance_config = {
    0: int(N_SAMPLES * MAJ_RATIO),
    1: N_SAMPLES - int(N_SAMPLES * MAJ_RATIO),
}


# ============================================================
# 2) HELPER: VYGENERUJ "POOL" A POTOM VYREŽ PRESNÝ POMER
# ============================================================

def generate_with_target_counts(stream_like, target_counts, batch_size=50_000, max_batches=100):
    """
    Vygeneruje dáta zo streamu (stream_like.next_sample),
    kým nebude mať aspoň target_counts[0] a target_counts[1] vzoriek.

    Potom použije make_imbalance (undersampling) na presné vyrezanie počtov.

    stream_like: objekt s metódou next_sample(n) -> (X, y)
    target_counts: dict {class_label: desired_count}
    """
    need0 = target_counts[0]
    need1 = target_counts[1]

    X_chunks = []
    y_chunks = []

    total_generated = 0

    for _ in range(max_batches):
        # vygeneruj ďalší batch z generátora
        Xb, yb = stream_like.next_sample(batch_size)
        X_chunks.append(Xb)
        y_chunks.append(yb)
        total_generated += len(yb)

        # skontroluj, či už máme dosť tried 0 aj 1
        y_all = np.concatenate(y_chunks)
        counts = np.bincount(y_all.astype(int), minlength=2)

        if counts[0] >= need0 and counts[1] >= need1:
            # už máme dosť dát, sprav "pool"
            X_all = np.vstack(X_chunks)
            y_all = y_all.astype(int)

            # make_imbalance spraví undersampling tak, aby počty presne sedeli
            X_imb, y_imb = make_imbalance(
                X_all,
                y_all,
                sampling_strategy=target_counts,
                random_state=42,
            )

            # zamiešanie (aby triedy neboli v blokoch)
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(y_imb))
            return X_imb[idx], y_imb[idx], total_generated, counts

    raise RuntimeError(
        f"Nepodarilo sa nazbierať dostatok vzoriek pre target_counts={target_counts} "
        f"ani po {max_batches} batchoch (batch_size={batch_size})."
    )


# ============================================================
# 3) DATASETY
# ============================================================

# --- 1) SEA (abrupt) ---
print("Generujem silne nevyvážený SEA_Abrupt...")
stream_sea = SEAGenerator(classification_function=0, random_state=42)

X_imb, y_imb, total_generated, counts = generate_with_target_counts(
    stream_sea, target_counts=imbalance_config, batch_size=50_000, max_batches=100
)

df_sea = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
df_sea["class"] = y_imb
df_sea.to_csv(os.path.join(DATA_DIR, "sea_abrupt_imb9010.csv"), index=False)

print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
print(f"Pomer tried:\n{df_sea['class'].value_counts(normalize=True)}\n")


# --- 2) Agrawal s driftom ---
print("Generujem silne nevyvážený Agrawal s driftom...")
stream1 = AGRAWALGenerator(classification_function=0, random_state=42)
stream2 = AGRAWALGenerator(classification_function=1, random_state=42)

# Drift v strede poolu (position) so šírkou prechodu width
drift_stream = ConceptDriftStream(
    stream1, stream2,
    position=N_SAMPLES // 2,
    width=100
)

X_imb, y_imb, total_generated, counts = generate_with_target_counts(
    drift_stream, target_counts=imbalance_config, batch_size=50_000, max_batches=100
)

df_agr = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
df_agr["class"] = y_imb
df_agr.to_csv(os.path.join(DATA_DIR, "agrawal_drift_imb9010.csv"), index=False)

print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
print(f"Pomer tried:\n{df_agr['class'].value_counts(normalize=True)}\n")


# --- 3) Rotating Hyperplane (gradual drift) ---
print("Generujem silne nevyvážený Rotating Hyperplane...")
hp_stream = HyperplaneGenerator(
    random_state=42,
    n_features=10,
    n_drift_features=5,
    mag_change=0.001,
    noise_percentage=0.05
)

X_imb, y_imb, total_generated, counts = generate_with_target_counts(
    hp_stream, target_counts=imbalance_config, batch_size=50_000, max_batches=100
)

df_hp = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
df_hp["class"] = y_imb
df_hp.to_csv(os.path.join(DATA_DIR, "hyperplane_gradual_imb9010.csv"), index=False)

print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
print(f"Pomer tried:\n{df_hp['class'].value_counts(normalize=True)}\n")


# --- 4) Random RBF drift ---
print("Generujem silne nevyvážený Random RBF drift...")
rbf_stream = RandomRBFGeneratorDrift(
    model_random_state=42,
    sample_random_state=42,
    n_classes=2,
    n_features=10,
    n_centroids=50,
    change_speed=0.001
)

X_imb, y_imb, total_generated, counts = generate_with_target_counts(
    rbf_stream, target_counts=imbalance_config, batch_size=50_000, max_batches=100
)

df_rbf = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
df_rbf["class"] = y_imb
df_rbf.to_csv(os.path.join(DATA_DIR, "rbf_drift_imb9010.csv"), index=False)

print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
print(f"Pomer tried:\n{df_rbf['class'].value_counts(normalize=True)}\n")