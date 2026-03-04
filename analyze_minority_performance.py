import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


RESULTS_DIR = "./results/"
RAW_RESULTS_FILE = "grid_search_results_raw.csv"
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
OUT_DIR = os.path.join(RESULTS_DIR, "minority_analysis")
CM_DIR = os.path.join(OUT_DIR, "confusion_matrices")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)


def sanitize_filename(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))


def save_confusion_matrix_plot(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_results():
    ensure_dirs()

    if not os.path.exists(RAW_RESULTS_FILE):
        print(f"Chyba: Súbor '{RAW_RESULTS_FILE}' neexistuje.")
        return

    df_raw = pd.read_csv(RAW_RESULTS_FILE)

    rows_overall = []
    rows_per_class = []
    rows_conf = []

    for _, row in df_raw.iterrows():
        run_id = int(row["Run_ID"])
        d_name = row["Dataset"]
        m_name = row["Model"]

        preds_filename = f"{d_name}_{m_name}_run{run_id}.npz"
        preds_path = os.path.join(PREDICTIONS_DIR, preds_filename)

        if not os.path.exists(preds_path):
            print(f"Varovanie: chýbajú predikcie pre {preds_filename}")
            continue

        with np.load(preds_path, allow_pickle=True) as data:
            y_true = data["y_true"].astype(int)
            y_pred = data["y_pred"].astype(int)

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        counts = pd.Series(y_true).value_counts().sort_index()
        majority_class = int(counts.idxmax())
        minority_classes = [int(c) for c in counts.index if int(c) != majority_class]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        minority_f1_values = []
        minority_recall_values = []

        for cls in labels:
            cls_key = str(cls)
            cls_metrics = report.get(cls_key, {})
            cls_f1 = float(cls_metrics.get("f1-score", 0.0))
            cls_rec = float(cls_metrics.get("recall", 0.0))
            cls_prec = float(cls_metrics.get("precision", 0.0))
            cls_sup = int(cls_metrics.get("support", 0))

            rows_per_class.append({
                "Run_ID": run_id,
                "Dataset": d_name,
                "Model": m_name,
                "Class": int(cls),
                "Is_Majority": int(int(cls) == majority_class),
                "Precision": cls_prec,
                "Recall": cls_rec,
                "F1": cls_f1,
                "Support": cls_sup,
            })

            if int(cls) != majority_class:
                minority_f1_values.append(cls_f1)
                minority_recall_values.append(cls_rec)

        rows_overall.append({
            "Run_ID": run_id,
            "Dataset": d_name,
            "Model": m_name,
            "Majority_Class": majority_class,
            "Minority_Classes": ",".join(map(str, minority_classes)),
            "Mean_Minority_F1": float(np.mean(minority_f1_values)) if minority_f1_values else 0.0,
            "Worst_Minority_F1": float(np.min(minority_f1_values)) if minority_f1_values else 0.0,
            "Mean_Minority_Recall": float(np.mean(minority_recall_values)) if minority_recall_values else 0.0,
            "Worst_Minority_Recall": float(np.min(minority_recall_values)) if minority_recall_values else 0.0,
            "Macro_F1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
            "Weighted_F1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        })

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        for i, tl in enumerate(labels):
            for j, pl in enumerate(labels):
                rows_conf.append({
                    "Run_ID": run_id,
                    "Dataset": d_name,
                    "Model": m_name,
                    "True_Label": int(tl),
                    "Pred_Label": int(pl),
                    "Count": int(cm[i, j]),
                })

        plot_name = sanitize_filename(f"{d_name}__{m_name}__run{run_id}.png")
        save_confusion_matrix_plot(
            cm=cm,
            labels=labels,
            title=f"{d_name} | {m_name} | run {run_id}",
            out_path=os.path.join(CM_DIR, plot_name),
        )

        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print(
            f"{d_name} | {m_name} | run {run_id} | "
            f"y_true={dict(zip(unique_true, counts_true))} | "
            f"y_pred={dict(zip(unique_pred, counts_pred))}"
        )

    df_overall = pd.DataFrame(rows_overall)
    df_per_class = pd.DataFrame(rows_per_class)
    df_conf = pd.DataFrame(rows_conf)

    df_overall.to_csv(os.path.join(OUT_DIR, "multiclass_minority_overall_raw.csv"), index=False, float_format="%.6f")
    df_per_class.to_csv(os.path.join(OUT_DIR, "multiclass_per_class_raw.csv"), index=False, float_format="%.6f")
    df_conf.to_csv(os.path.join(OUT_DIR, "multiclass_confusion_matrix_raw.csv"), index=False)

    summary = df_overall.groupby(["Dataset", "Model"]).agg(
        Avg_Mean_Minority_F1=("Mean_Minority_F1", "mean"),
        Avg_Worst_Minority_F1=("Worst_Minority_F1", "mean"),
        Avg_Mean_Minority_Recall=("Mean_Minority_Recall", "mean"),
        Avg_Worst_Minority_Recall=("Worst_Minority_Recall", "mean"),
        Avg_Macro_F1=("Macro_F1", "mean"),
        Avg_Weighted_F1=("Weighted_F1", "mean"),
    ).reset_index()

    summary.to_csv(os.path.join(OUT_DIR, "multiclass_minority_summary.csv"), index=False, float_format="%.6f")

    worst_cases = df_overall.sort_values("Worst_Minority_F1", ascending=True).head(20)
    worst_cases.to_csv(os.path.join(OUT_DIR, "worst_cases_by_worst_minority_f1.csv"), index=False, float_format="%.6f")

    print("\n--- MULTICLASS minority summary ---")
    print(summary.sort_values(["Dataset", "Avg_Worst_Minority_F1"], ascending=[True, False]))

    print("\n--- Worst cases ---")
    print(worst_cases[["Dataset", "Model", "Run_ID", "Worst_Minority_F1", "Mean_Minority_F1"]])


if __name__ == "__main__":
    analyze_results()