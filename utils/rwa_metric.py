import numpy as np
from collections import Counter


def calculate_rwa(y_true, y_pred, y_classes=None):
    """
    Vypočíta Rarity-based Weighted Accuracy (RWA).

    Táto verzia je multi-class safe.
    Väčšiu váhu dáva správnej klasifikácii vzácnejších tried.

    Parametre:
    - y_true: numpy pole true tried
    - y_pred: numpy pole predikovaných tried
    - y_classes: zoznam / pole všetkých tried, ktoré chceme zohľadniť

    Návrat:
    - float RWA skóre
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return 0.0

    if y_classes is None:
        y_classes = np.unique(y_true)
    else:
        y_classes = np.asarray(y_classes, dtype=int)

    class_counts = Counter(y_true)

    # 1) per-class recall / accuracy
    per_class_accuracy = {}
    for c in y_classes:
        class_indices = np.where(y_true == c)[0]

        if len(class_indices) == 0:
            per_class_accuracy[int(c)] = 0.0
            continue

        correct_predictions = np.sum(y_pred[class_indices] == y_true[class_indices])
        per_class_accuracy[int(c)] = correct_predictions / len(class_indices)

    # 2) rarity weights
    total_samples = len(y_true)
    class_rarity_weights = {}

    sum_of_inverse_freq = 0.0
    for c in y_classes:
        count = class_counts.get(int(c), 0)
        if count > 0:
            sum_of_inverse_freq += total_samples / count

    if sum_of_inverse_freq == 0:
        return 0.0

    for c in y_classes:
        count = class_counts.get(int(c), 0)
        if count == 0:
            class_rarity_weights[int(c)] = 0.0
        else:
            frequency = count / total_samples
            class_rarity_weights[int(c)] = (1.0 / frequency) / sum_of_inverse_freq

    # 3) final RWA
    rwa_score = sum([
        class_rarity_weights[int(c)] * per_class_accuracy[int(c)]
        for c in y_classes
    ])

    return float(rwa_score)


if __name__ == '__main__':
    print("--- Testovanie calculate_rwa ---")

    # binary test
    y_true_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_pred_lazy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred_specialist = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    print("Binary lazy:", calculate_rwa(y_true_test, y_pred_lazy, [0, 1]))
    print("Binary specialist:", calculate_rwa(y_true_test, y_pred_specialist, [0, 1]))

    # multiclass test
    y_true_mc = np.array([0, 0, 0, 0, 1, 1, 2])
    y_pred_mc = np.array([0, 0, 0, 1, 1, 0, 2])

    print("Multiclass:", calculate_rwa(y_true_mc, y_pred_mc, [0, 1, 2]))