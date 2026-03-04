"""
Runner skript pre streamové experimenty.

Nová verzia:
- binary aj multiclass datasety
- DDCW ablation:
    1) base
    2) replay
    3) augment
    4) augment + drift-aware
- baseline KUE
- voliteľne baseline ARF, ak je dostupný
- ukladá raw výsledky
- ukladá predikcie + proba
- ukladá blokové/prequential metriky
- pre multiclass sleduje:
    - Macro_F1
    - Mean_Minority_F1
    - Worst_Minority_F1
    - Mean_Minority_Recall
    - Worst_Minority_Recall
"""

import os
import sys
import time
import pickle
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

from skmultiflow.data.data_stream import DataStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import (
    HoeffdingTreeClassifier,
    ExtremelyFastDecisionTreeClassifier,
    HoeffdingAdaptiveTreeClassifier,
)
from skmultiflow.neural_networks import PerceptronMask

try:
    from skmultiflow.meta import AdaptiveRandomForestClassifier
    HAS_ARF = True
except Exception:
    HAS_ARF = False

from utils.data_preprocesing import read_clean_csv
from utils.rwa_metric import calculate_rwa

from model.configurable_ddcw_new import Configurable_DDCW
from model.kue import KUE


# ============================================================
# 1) KONFIGURÁCIA
# ============================================================

NUMBER_OF_RUNS = 1

DATA_DIR = "./data/"
RESULTS_DIR = "./results/"

# None = spracuj celý dataset po pretrain
MAX_SAMPLES = None

PRETRAIN_SIZE = 500
BLOCK_SIZE = 500

SYNTHETIC_DATASETS = {
    # binary
    "SEA_Imb9010": "synthetic_imbalanced/sea_abrupt_imb9010.csv",
    "Agrawal_Imb9010": "synthetic_imbalanced/agrawal_drift_imb9010.csv",
    "hyperplane_gradual_imb9010": "synthetic_imbalanced/hyperplane_gradual_imb9010.csv",
    "rbf_drift_imb9010": "synthetic_imbalanced/rbf_drift_imb9010.csv",

    # multiclass
    "MC_Abrupt_3C_70155": "synthetic_multiclass/mc_abrupt_3c_70155.csv",
    "MC_Gradual_3C_70155": "synthetic_multiclass/mc_gradual_3c_70155.csv",
    "MC_Abrupt_4C_601555": "synthetic_multiclass/mc_abrupt_4c_601555.csv",
    "MC_Reoccurring_3C_80155": "synthetic_multiclass/mc_reoccurring_3c_80155.csv",
}


# ============================================================
# 2) PROGRESS BAR
# ============================================================

_LAST_PROGRESS = {"current": 0, "total": 1, "prefix": ""}


def print_progress_bar(current, total, prefix="", bar_length=30):
    total = max(1, int(total))
    current = int(current)

    progress = min(1.0, max(0.0, current / float(total)))
    filled_len = int(bar_length * progress)

    bar = "#" * filled_len + "-" * (bar_length - filled_len)
    percent = int(progress * 100)

    if prefix:
        print(f"\r{prefix} |{bar}| {percent:3d}% ({current}/{total})", end="", flush=True)
    else:
        print(f"\r|{bar}| {percent:3d}% ({current}/{total})", end="", flush=True)

    if current >= total:
        print("")


def _warning_to_clean_line(message, category, filename, lineno, file=None, line=None):
    print("", flush=True)
    print(f"WARNING: {category.__name__}: {message} ({os.path.basename(filename)}:{lineno})")
    print_progress_bar(
        _LAST_PROGRESS["current"],
        _LAST_PROGRESS["total"],
        prefix=_LAST_PROGRESS["prefix"]
    )


warnings.showwarning = _warning_to_clean_line


# ============================================================
# 3) METRIKY HELPERS
# ============================================================

def safe_auc(y_true, y_proba, n_total_classes):
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.5

    try:
        if n_total_classes > 2:
            return roc_auc_score(y_true, y_proba, multi_class="ovr")
        return roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        return 0.5


def compute_main_metrics(y_true, y_pred, y_proba, n_total_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    labels_all = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    counts = pd.Series(y_true).value_counts().sort_index()

    majority_class = int(counts.idxmax())
    minority_classes = [int(c) for c in counts.index if int(c) != majority_class]

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    rwa = float(calculate_rwa(y_true, y_pred, np.arange(n_total_classes)))
    auc_score = float(safe_auc(y_true, y_proba, n_total_classes))

    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        average=None,
        zero_division=0
    )

    class_rows = []
    for i, cls in enumerate(labels_all):
        class_rows.append({
            "class": int(cls),
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "f1": float(f1_arr[i]),
            "support": int(supp_arr[i]),
        })

    df_cls = pd.DataFrame(class_rows)
    df_min = df_cls[df_cls["class"] != majority_class].copy()

    if len(df_min) > 0:
        mean_minority_f1 = float(df_min["f1"].mean())
        worst_minority_f1 = float(df_min["f1"].min())
        mean_minority_recall = float(df_min["recall"].mean())
        worst_minority_recall = float(df_min["recall"].min())
    else:
        mean_minority_f1 = 0.0
        worst_minority_f1 = 0.0
        mean_minority_recall = 0.0
        worst_minority_recall = 0.0

    return {
        "Accuracy": accuracy,
        "Weighted_F1": weighted_f1,
        "Macro_F1": macro_f1,
        "Kappa": kappa,
        "AUC": auc_score,
        "RWA_Score": rwa,
        "Majority_Class": majority_class,
        "Minority_Classes": ",".join(map(str, minority_classes)),
        "Mean_Minority_F1": mean_minority_f1,
        "Worst_Minority_F1": worst_minority_f1,
        "Mean_Minority_Recall": mean_minority_recall,
        "Worst_Minority_Recall": worst_minority_recall,
    }


def get_model_name(model):
    if isinstance(model, Configurable_DDCW):
        params = model.get_params()

        m_name = "DDCW"
        m_name += f"_mode-{params['replay_mode']}"
        if params["augmentation_mode"] != "none":
            m_name += f"_aug-{params['augmentation_mode']}{params['augmentation_strength']}"
        if params["enable_drift_detector"]:
            m_name += "_drift"
        m_name += "_noDiv" if not params["enable_diversity"] else "_Div"
        m_name += f"_p{params['period']}"
        m_name += f"_hb{params['history_buffer_size']}"
        m_name += f"_cb{params['class_buffer_size']}"
        m_name += f"_rk{params['replay_k']}"
        return m_name, params

    return model.__class__.__name__, {}


# ============================================================
# 4) MODELY
# ============================================================

def get_model_configs(run_id=1):
    estimators_hetero = [
        NaiveBayes(),
        HoeffdingTreeClassifier(),
        HoeffdingAdaptiveTreeClassifier(),
        ExtremelyFastDecisionTreeClassifier(),
        PerceptronMask(),
    ]

    models = []

    # DDCW base
    models.append(
        Configurable_DDCW(
            base_estimators=estimators_hetero,
            period=600,
            beta=1.5,
            theta=0.02,
            enable_diversity=False,
            use_lifetime_trend=True,
            history_buffer_size=600,
            class_buffer_size=300,
            replay_mode="off",
            replay_k=0,
            augmentation_mode="none",
            augmentation_strength=0.0,
            enable_drift_detector=False,
            random_state=100 + run_id,
        )
    )

    # DDCW replay
    models.append(
        Configurable_DDCW(
            base_estimators=estimators_hetero,
            period=600,
            beta=1.5,
            theta=0.02,
            enable_diversity=False,
            use_lifetime_trend=True,
            history_buffer_size=600,
            class_buffer_size=300,
            replay_mode="replay",
            replay_k=3,
            augmentation_mode="none",
            augmentation_strength=0.0,
            enable_drift_detector=False,
            random_state=200 + run_id,
        )
    )

    # DDCW augment
    models.append(
        Configurable_DDCW(
            base_estimators=estimators_hetero,
            period=600,
            beta=1.5,
            theta=0.02,
            enable_diversity=False,
            use_lifetime_trend=True,
            history_buffer_size=600,
            class_buffer_size=300,
            replay_mode="augment",
            replay_k=3,
            augmentation_mode="noise",
            augmentation_strength=0.02,
            imbalance_aware_augmentation=True,
            enable_drift_detector=False,
            random_state=300 + run_id,
        )
    )

    # DDCW augment + drift-aware
    models.append(
        Configurable_DDCW(
            base_estimators=estimators_hetero,
            period=600,
            beta=1.5,
            theta=0.02,
            enable_diversity=False,
            use_lifetime_trend=True,
            history_buffer_size=600,
            class_buffer_size=300,
            replay_mode="augment",
            replay_k=3,
            augmentation_mode="noise",
            augmentation_strength=0.02,
            imbalance_aware_augmentation=True,
            enable_drift_detector=True,
            drift_delta=0.005,
            drift_threshold=20.0,
            post_drift_cooldown=300,
            post_drift_replay_boost=1,
            post_drift_aug_reduction=0.5,
            reset_majority_history_on_drift=True,
            keep_class_buffers_on_drift=True,
            random_state=400 + run_id,
        )
    )

    # baseline KUE
    models.append(
        KUE(
            base_estimator=HoeffdingTreeClassifier(),
            n_estimators=10,
        )
    )

    # baseline ARF - ak je dostupný
    if HAS_ARF:
        models.append(
            AdaptiveRandomForestClassifier(
                n_estimators=10,
                random_state=600 + run_id,
            )
        )

    return models


# ============================================================
# 5) HLAVNÝ LOOP
# ============================================================

def run_experiments():
    all_results = []
    all_block_rows = []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    preds_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    for run_id in range(1, NUMBER_OF_RUNS + 1):
        print(f"\n{'='*50}\nZAČÍNA BEH EXPERIMENTU č. {run_id}/{NUMBER_OF_RUNS}\n{'='*50}")

        model_configs = get_model_configs(run_id=run_id)

        for d_name, d_filename in SYNTHETIC_DATASETS.items():
            print(f"\n{'='*25}\nDataset: {d_name} | Beh {run_id}\n{'='*25}")

            file_path = os.path.join(DATA_DIR, d_filename)
            if not os.path.exists(file_path):
                print(f"Chyba: Súbor {file_path} neexistuje. Preskočené.")
                continue

            try:
                _, X, y = read_clean_csv(file_path)
                stream = DataStream(X, y)
            except Exception as e:
                print(f"Chyba pri načítaní datasetu {d_name}: {e}")
                continue

            for model in model_configs:
                m_name, params = get_model_name(model)
                print(f"--- Spúšťa sa model: {m_name} ---")

                stream.restart()

                y_true_list = []
                y_pred_list = []
                y_proba_full_list = []
                model_sizes = []

                block_y_true = []
                block_y_pred = []
                block_y_proba = []

                start_time = time.time()
                n_samples = 0

                try:
                    if hasattr(model, "reset"):
                        model.reset()

                    if stream.n_remaining_samples() >= PRETRAIN_SIZE:
                        X_pre, y_pre = stream.next_sample(PRETRAIN_SIZE)
                        model.partial_fit(X_pre, y_pre, classes=stream.target_values)

                    n_total_classes = stream.n_classes

                    remaining_after_pretrain = stream.n_remaining_samples()
                    total_to_process = remaining_after_pretrain if MAX_SAMPLES is None else min(MAX_SAMPLES, remaining_after_pretrain)
                    progress_prefix = f"{d_name} | {m_name}"

                    while stream.has_more_samples() and (MAX_SAMPLES is None or n_samples < MAX_SAMPLES):
                        X_sample, y_sample = stream.next_sample()

                        try:
                            proba_array = model.predict_proba(X_sample)
                        except Exception:
                            proba_array = None

                        if proba_array is None:
                            try:
                                pred_tmp = int(model.predict(X_sample)[0])
                            except Exception:
                                pred_tmp = 0
                            proba_array = np.zeros((1, n_total_classes), dtype=float)
                            if 0 <= pred_tmp < n_total_classes:
                                proba_array[0, pred_tmp] = 1.0

                        full_proba = np.zeros(n_total_classes, dtype=float)
                        current_probas = np.asarray(proba_array[0], dtype=float)
                        full_proba[:len(current_probas)] = current_probas[:n_total_classes]

                        try:
                            pred = int(model.predict(X_sample)[0])
                        except Exception:
                            pred = 0

                        pred = int(np.clip(pred, 0, n_total_classes - 1))
                        true_label = int(y_sample[0])

                        y_pred_list.append(pred)
                        y_true_list.append(true_label)
                        y_proba_full_list.append(full_proba)

                        block_y_true.append(true_label)
                        block_y_pred.append(pred)
                        block_y_proba.append(full_proba)

                        model.partial_fit(X_sample, y_sample)

                        n_samples += 1

                        if (n_samples % 200 == 0 or n_samples == 1) and n_samples < total_to_process:
                            _LAST_PROGRESS["current"] = n_samples
                            _LAST_PROGRESS["total"] = total_to_process
                            _LAST_PROGRESS["prefix"] = progress_prefix
                            print_progress_bar(n_samples, total_to_process, prefix=progress_prefix)

                        if n_samples % 200 == 0:
                            try:
                                model_sizes.append(sys.getsizeof(pickle.dumps(model)))
                            except Exception:
                                pass

                        if len(block_y_true) >= BLOCK_SIZE:
                            block_metrics = compute_main_metrics(
                                y_true=np.array(block_y_true, dtype=int),
                                y_pred=np.array(block_y_pred, dtype=int),
                                y_proba=np.array(block_y_proba, dtype=float),
                                n_total_classes=n_total_classes,
                            )

                            block_row = {
                                "Run_ID": run_id,
                                "Dataset": d_name,
                                "Model": m_name,
                                "Block_End": n_samples,
                                **block_metrics,
                            }
                            all_block_rows.append(block_row)

                            block_y_true = []
                            block_y_pred = []
                            block_y_proba = []

                    if len(block_y_true) > 0:
                        block_metrics = compute_main_metrics(
                            y_true=np.array(block_y_true, dtype=int),
                            y_pred=np.array(block_y_pred, dtype=int),
                            y_proba=np.array(block_y_proba, dtype=float),
                            n_total_classes=n_total_classes,
                        )
                        block_row = {
                            "Run_ID": run_id,
                            "Dataset": d_name,
                            "Model": m_name,
                            "Block_End": n_samples,
                            **block_metrics,
                        }
                        all_block_rows.append(block_row)

                    _LAST_PROGRESS["current"] = total_to_process
                    _LAST_PROGRESS["total"] = total_to_process
                    _LAST_PROGRESS["prefix"] = progress_prefix
                    print_progress_bar(total_to_process, total_to_process, prefix=progress_prefix)

                    total_time = time.time() - start_time
                    avg_update_time = np.mean(model.update_times) if hasattr(model, "update_times") and model.update_times else 0.0
                    avg_mem_kb = (np.mean(model_sizes) / 1024.0) if model_sizes else 0.0

                    y_true_arr = np.array(y_true_list, dtype=int)
                    y_pred_arr = np.array(y_pred_list, dtype=int)
                    y_proba_full_arr = np.array(y_proba_full_list, dtype=float)

                    metrics = compute_main_metrics(
                        y_true=y_true_arr,
                        y_pred=y_pred_arr,
                        y_proba=y_proba_full_arr,
                        n_total_classes=n_total_classes,
                    )

                    preds_filename = f"{d_name}_{m_name}_run{run_id}.npz"
                    save_dict = {
                        "y_true": y_true_arr,
                        "y_pred": y_pred_arr,
                        "y_proba": y_proba_full_arr,
                    }

                    if hasattr(model, "_drift_points"):
                        save_dict["drift_points"] = np.array(model._drift_points, dtype=int)

                    np.savez_compressed(
                        os.path.join(preds_dir, preds_filename),
                        **save_dict
                    )

                    result_row = {
                        "Run_ID": run_id,
                        "Dataset": d_name,
                        "Model": m_name,
                        "Total_Samples_Evaluated": int(len(y_true_arr)),
                        **metrics,
                        "Total_Time_s": total_time,
                        "Avg_Update_Time_s": avg_update_time,
                        "Memory_kB": avg_mem_kb,
                    }
                    result_row.update(params)
                    all_results.append(result_row)

                    print(
                        f"Dokončené | {m_name} | "
                        f"RWA={metrics['RWA_Score']:.4f} | "
                        f"MacroF1={metrics['Macro_F1']:.4f} | "
                        f"MeanMinF1={metrics['Mean_Minority_F1']:.4f} | "
                        f"WorstMinF1={metrics['Worst_Minority_F1']:.4f} | "
                        f"Time={total_time:.2f}s"
                    )

                except Exception as e:
                    print(f"CHYBA počas evaluácie modelu {m_name}: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "Run_ID": run_id,
                        "Dataset": d_name,
                        "Model": m_name,
                        "RWA_Score": -1.0,
                        "Macro_F1": -1.0,
                        "Mean_Minority_F1": -1.0,
                        "Worst_Minority_F1": -1.0,
                    })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("grid_search_results_raw.csv", index=False, float_format="%.6f")
    print("\nRaw výsledky uložené v: grid_search_results_raw.csv")

    blocks_df = pd.DataFrame(all_block_rows)
    blocks_df.to_csv(os.path.join(RESULTS_DIR, "prequential_block_metrics.csv"), index=False, float_format="%.6f")
    print("Blokové metriky uložené v: ./results/prequential_block_metrics.csv")

    summary = results_df.groupby(["Dataset", "Model"]).agg(
        Avg_RWA=("RWA_Score", "mean"),
        Std_RWA=("RWA_Score", "std"),
        Avg_Macro_F1=("Macro_F1", "mean"),
        Std_Macro_F1=("Macro_F1", "std"),
        Avg_Weighted_F1=("Weighted_F1", "mean"),
        Std_Weighted_F1=("Weighted_F1", "std"),
        Avg_Mean_Minority_F1=("Mean_Minority_F1", "mean"),
        Std_Mean_Minority_F1=("Mean_Minority_F1", "std"),
        Avg_Worst_Minority_F1=("Worst_Minority_F1", "mean"),
        Std_Worst_Minority_F1=("Worst_Minority_F1", "std"),
        Avg_Mean_Minority_Recall=("Mean_Minority_Recall", "mean"),
        Avg_Worst_Minority_Recall=("Worst_Minority_Recall", "mean"),
        Avg_Kappa=("Kappa", "mean"),
        Avg_AUC=("AUC", "mean"),
        Avg_Time=("Total_Time_s", "mean"),
        Avg_Memory=("Memory_kB", "mean"),
    ).reset_index()

    summary.to_csv("grid_search_summary.csv", index=False, float_format="%.6f")
    print("\nSúhrnné výsledky uložené v: grid_search_summary.csv")
    print(summary.sort_values(["Dataset", "Avg_RWA"], ascending=[True, False]))


if __name__ == "__main__":
    run_experiments()