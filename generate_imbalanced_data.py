import os
import numpy as np
import pandas as pd

from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.agrawal_generator import AGRAWALGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import RandomRBFGeneratorDrift
from imblearn.datasets import make_imbalance

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# OUTPUT PRIEČINKY
# ============================================================

DATA_DIR_BINARY = "./data/synthetic_imbalanced/"
DATA_DIR_MULTI = "./data/synthetic_multiclass/"
os.makedirs(DATA_DIR_BINARY, exist_ok=True)
os.makedirs(DATA_DIR_MULTI, exist_ok=True)

N_SAMPLES = 50_000
MAJ_RATIO = 0.90

imbalance_config_binary = {
    0: int(N_SAMPLES * MAJ_RATIO),
    1: N_SAMPLES - int(N_SAMPLES * MAJ_RATIO)
}


# ============================================================
# BINARY DATASETY
# ============================================================

def generate_with_target_counts(stream_like, target_counts, batch_size=50_000, max_batches=100):
    need_counts = dict(target_counts)

    X_chunks = []
    y_chunks = []

    total_generated = 0
    n_classes = max(need_counts.keys()) + 1

    for _ in range(max_batches):
        Xb, yb = stream_like.next_sample(batch_size)
        X_chunks.append(Xb)
        y_chunks.append(yb)
        total_generated += len(yb)

        y_all = np.concatenate(y_chunks).astype(int)
        counts = np.bincount(y_all, minlength=n_classes)

        enough = True
        for cls, need in need_counts.items():
            if counts[cls] < need:
                enough = False
                break

        if enough:
            X_all = np.vstack(X_chunks)
            y_all = y_all.astype(int)

            X_imb, y_imb = make_imbalance(
                X_all,
                y_all,
                sampling_strategy=need_counts,
                random_state=42
            )

            rng = np.random.RandomState(42)
            idx = rng.permutation(len(y_imb))
            return X_imb[idx], y_imb[idx], total_generated, counts

    raise RuntimeError(
        f"Nepodarilo sa nazbierať dostatok vzoriek pre target_counts={target_counts} "
        f"ani po {max_batches} batchoch."
    )


def generate_binary_datasets():
    print("Generujem binary datasety...")

    # 1) SEA abrupt
    print("Generujem silne nevyvážený SEA_Abrupt...")
    stream_sea = SEAGenerator(classification_function=0, random_state=42)

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        stream_sea,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_sea = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_sea["class"] = y_imb
    df_sea.to_csv(os.path.join(DATA_DIR_BINARY, "sea_abrupt_imb9010.csv"), index=False)

    print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
    print(df_sea["class"].value_counts(normalize=True).sort_index())
    print()

    # 2) Agrawal drift
    print("Generujem silne nevyvážený Agrawal s driftom...")
    stream1 = AGRAWALGenerator(classification_function=0, random_state=42)
    stream2 = AGRAWALGenerator(classification_function=1, random_state=42)

    drift_stream = ConceptDriftStream(
        stream=stream1,
        drift_stream=stream2,
        position=N_SAMPLES // 2,
        width=100
    )

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        drift_stream,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_agr = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_agr["class"] = y_imb
    df_agr.to_csv(os.path.join(DATA_DIR_BINARY, "agrawal_drift_imb9010.csv"), index=False)

    print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
    print(df_agr["class"].value_counts(normalize=True).sort_index())
    print()

    # 3) Hyperplane gradual drift
    print("Generujem silne nevyvážený Rotating Hyperplane...")
    hp_stream = HyperplaneGenerator(
        random_state=42,
        n_features=10,
        n_drift_features=5,
        mag_change=0.001,
        noise_percentage=0.05
    )

    X_imb, y_imb, total_generated, counts = generate_with_target_counts(
        hp_stream,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_hp = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_hp["class"] = y_imb
    df_hp.to_csv(os.path.join(DATA_DIR_BINARY, "hyperplane_gradual_imb9010.csv"), index=False)

    print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
    print(df_hp["class"].value_counts(normalize=True).sort_index())
    print()

    # 4) RBF drift
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
        rbf_stream,
        target_counts=imbalance_config_binary,
        batch_size=50_000,
        max_batches=100
    )

    df_rbf = pd.DataFrame(X_imb, columns=[f"attr_{i}" for i in range(X_imb.shape[1])])
    df_rbf["class"] = y_imb
    df_rbf.to_csv(os.path.join(DATA_DIR_BINARY, "rbf_drift_imb9010.csv"), index=False)

    print(f"Hotovo. (vygenerované spolu: {total_generated}, pool counts: {counts})")
    print(df_rbf["class"].value_counts(normalize=True).sort_index())
    print()


# ============================================================
# MULTICLASS DATASETY
# ============================================================

def normalize_weights(class_weights):
    w = np.array(class_weights, dtype=float)
    w = w / w.sum()
    return w.tolist()


def make_multiclass_chunk(
    n_samples,
    n_features,
    n_classes,
    class_weights,
    class_sep=1.0,
    flip_y=0.01,
    n_informative=None,
    shift=None,
    scale=None,
    random_state=42,
):
    if n_informative is None:
        n_informative = max(3, min(n_features, n_classes * 2))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=normalize_weights(class_weights),
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    if shift is not None:
        shift = np.asarray(shift, dtype=float)
        X = X + shift

    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        X = X * scale

    return X.astype(float), y.astype(int)


def blend_chunks(X_a, y_a, X_b, y_b, width):
    assert len(X_a) == len(X_b) == len(y_a) == len(y_b)
    n = len(y_a)

    if width <= 0:
        return X_b, y_b

    start = max(0, (n // 2) - (width // 2))
    end = min(n, start + width)

    X_out = []
    y_out = []

    for i in range(n):
        if i < start:
            X_out.append(X_a[i])
            y_out.append(y_a[i])
        elif i >= end:
            X_out.append(X_b[i])
            y_out.append(y_b[i])
        else:
            p_b = (i - start) / max(1, (end - start))
            if np.random.rand() < p_b:
                X_out.append(X_b[i])
                y_out.append(y_b[i])
            else:
                X_out.append(X_a[i])
                y_out.append(y_a[i])

    return np.asarray(X_out, dtype=float), np.asarray(y_out, dtype=int)


def scale_features_0_1(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def save_multiclass_dataset(X, y, filename):
    cols = [f"attr_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["class"] = y.astype(int)
    df.to_csv(os.path.join(DATA_DIR_MULTI, filename), index=False)

    print(f"\nUložené: {filename}")
    print("Shape:", df.shape)
    print("Class distribution:")
    print(df["class"].value_counts().sort_index())
    print("Class ratios:")
    print(df["class"].value_counts(normalize=True).sort_index())


def generate_mc_abrupt_3c_70155():
    n1 = N_SAMPLES // 2
    n2 = N_SAMPLES - n1

    X1, y1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=1.2,
        flip_y=0.01,
        shift=np.zeros(12),
        scale=np.ones(12),
        random_state=101,
    )

    X2, y2 = make_multiclass_chunk(
        n_samples=n2,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=0.9,
        flip_y=0.02,
        shift=np.array([0.8, -0.7, 0.5, -0.3, 0.6, -0.5, 0.2, -0.1, 0.4, -0.6, 0.7, -0.2]),
        scale=np.array([1.1, 0.9, 1.0, 1.2, 1.0, 0.8, 1.1, 1.0, 0.9, 1.2, 1.0, 1.1]),
        random_state=102,
    )

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_abrupt_3c_70155.csv")


def generate_mc_gradual_3c_70155():
    n = N_SAMPLES

    X_a, y_a = make_multiclass_chunk(
        n_samples=n,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=1.1,
        flip_y=0.01,
        shift=np.zeros(12),
        scale=np.ones(12),
        random_state=201,
    )

    X_b, y_b = make_multiclass_chunk(
        n_samples=n,
        n_features=12,
        n_classes=3,
        class_weights=[0.70, 0.15, 0.15],
        class_sep=0.85,
        flip_y=0.03,
        shift=np.array([-0.9, 0.7, -0.4, 0.5, -0.6, 0.3, -0.7, 0.2, -0.5, 0.6, -0.2, 0.4]),
        scale=np.array([0.9, 1.2, 1.0, 0.8, 1.1, 1.0, 1.2, 0.9, 1.1, 0.8, 1.0, 1.2]),
        random_state=202,
    )

    X, y = blend_chunks(X_a, y_a, X_b, y_b, width=8000)
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_gradual_3c_70155.csv")


def generate_mc_abrupt_4c_601555():
    n1 = N_SAMPLES // 2
    n2 = N_SAMPLES - n1

    X1, y1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=14,
        n_classes=4,
        class_weights=[0.60, 0.15, 0.15, 0.10],
        class_sep=1.15,
        flip_y=0.01,
        shift=np.zeros(14),
        scale=np.ones(14),
        random_state=301,
    )

    X2, y2 = make_multiclass_chunk(
        n_samples=n2,
        n_features=14,
        n_classes=4,
        class_weights=[0.60, 0.15, 0.15, 0.10],
        class_sep=0.80,
        flip_y=0.03,
        shift=np.array([0.7, -0.4, 0.5, -0.6, 0.4, -0.3, 0.6, -0.7, 0.2, -0.5, 0.3, -0.2, 0.8, -0.1]),
        scale=np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 1.0, 0.9]),
        random_state=302,
    )

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_abrupt_4c_601555.csv")


def generate_mc_reoccurring_3c_80155():
    n1 = N_SAMPLES // 3
    n2 = N_SAMPLES // 3
    n3 = N_SAMPLES - n1 - n2

    Xa1, ya1 = make_multiclass_chunk(
        n_samples=n1,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=1.25,
        flip_y=0.01,
        shift=np.zeros(10),
        scale=np.ones(10),
        random_state=401,
    )

    Xb, yb = make_multiclass_chunk(
        n_samples=n2,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=0.85,
        flip_y=0.03,
        shift=np.array([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.6, -0.5]),
        scale=np.array([1.2, 0.8, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 1.0, 0.9]),
        random_state=402,
    )

    Xa2, ya2 = make_multiclass_chunk(
        n_samples=n3,
        n_features=10,
        n_classes=3,
        class_weights=[0.80, 0.15, 0.05],
        class_sep=1.20,
        flip_y=0.015,
        shift=np.array([0.15, -0.10, 0.05, -0.08, 0.03, -0.02, 0.01, -0.04, 0.07, -0.03]),
        scale=np.ones(10),
        random_state=403,
    )

    X = np.vstack([Xa1, Xb, Xa2])
    y = np.concatenate([ya1, yb, ya2])
    X = scale_features_0_1(X)

    save_multiclass_dataset(X, y, "mc_reoccurring_3c_80155.csv")


def generate_multiclass_datasets():
    print("Generujem multiclass datasety...")
    np.random.seed(42)

    generate_mc_abrupt_3c_70155()
    generate_mc_gradual_3c_70155()
    generate_mc_abrupt_4c_601555()
    generate_mc_reoccurring_3c_80155()


if __name__ == "__main__":
    generate_binary_datasets()
    generate_multiclass_datasets()

    print("\nHotovo.")
    print("Binary datasety sú v:", DATA_DIR_BINARY)
    print("Multiclass datasety sú v:", DATA_DIR_MULTI)