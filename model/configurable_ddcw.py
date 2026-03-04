import copy as cp
import numpy as np
from collections import deque

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.neural_networks import PerceptronMask
from utils import diversity
from utils.rwa_metric import calculate_rwa
import time

from collections import Counter

import pandas as pd

from sklearn.exceptions import NotFittedError


class Configurable_DDCW(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """
    Konfigurovateľná verzia DDCW, ktorá zahŕňa:
    - RWA-based úpravu váh s nastaviteľnou "silou".
    - Možnosť vypnúť pairwise diverzitu.
    - Možnosť použiť inteligentný lifetime založený na trende RWA.
    """

    class WeightedExpert:
        def __init__(self, estimator, weight, num_classes):
            self.estimator = estimator
            self.model_type = type(estimator).__name__   # <-- NOVÉ
            self.weight_class = np.full(num_classes, weight, dtype=float)
            self.lifetime = 0
            self.rwa_history = deque(maxlen=5)
            self.warmup_remaining = 0  # <-- NOVÉ

    def __init__(
        self,
        min_estimators=5,
        max_estimators=20,
        base_estimators=[NaiveBayes(), HoeffdingTreeClassifier()],
        period=1000,
        alpha=0.002,
        beta=1.5,
        theta=0.05,
        enable_diversity=True,
        rwa_strength=0.0,
        use_lifetime_trend=False,
        warmup_windows=2,
    ):
        super().__init__()
        self.min_estimators = min_estimators
        self.max_estimators = max_estimators
        self.base_estimators = base_estimators
        self.period = period
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.enable_diversity = enable_diversity
        self.rwa_strength = rwa_strength
        self.use_lifetime_trend = use_lifetime_trend
        self.warmup_windows = int(max(0, warmup_windows))
        self._warmup_samples = None  # nastaví sa až keď poznáme period

        # interný stav
        self.p = -1
        self.epochs = None
        self.num_classes = None
        self.experts = None
        self.div = []
        self.window_size = None
        self.X_batch = None
        self.y_batch = None
        self.y_batch_experts = None
        self._classes = None

        self.update_times = []

        self.reset()

    # ----------------------------
    # safe wrappers
    # ----------------------------
    def _safe_predict(self, estimator, X):
        try:
            return estimator.predict(X)
        except (NotFittedError, AttributeError, ValueError):
            return None
        except Exception:
            return None

    def _safe_predict_proba(self, estimator, X):
        try:
            if hasattr(estimator, "predict_proba"):
                return estimator.predict_proba(X)
            return None
        except (NotFittedError, AttributeError, ValueError):
            return None
        except Exception:
            return None

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self._classes is None and classes is not None:
            self._classes = list(classes)
        for i in range(len(X)):
            self.fit_single_sample(X[i : i + 1, :], y[i : i + 1], self._classes, sample_weight)
        return self

    def predict(self, X):
        """
        Správne vážené hlasovanie:
        expert prispieva váhou len do triedy, ktorú predikoval.
        """
        if self._classes is None or not self.experts:
            return np.zeros(X.shape[0], dtype=int)

        predictions_class = np.zeros((X.shape[0], self.num_classes), dtype=float)

        for exp in self.experts:
            y_hat = self._safe_predict(exp.estimator, X)
            if y_hat is None:
                continue

            y_hat = y_hat.astype(int)
            y_hat = np.clip(y_hat, 0, self.num_classes - 1)
            for i, c in enumerate(y_hat):
                predictions_class[i, c] += float(exp.weight_class[c])

        return np.argmax(predictions_class, axis=1)

    def predict_proba(self, X):
        """
        Pravdepodobnosti ako vážený priemer expertov.
        """
        N = X.shape[0]
        if self._classes is None:
            num_classes = max(2, getattr(self, "num_classes", 2))
            out = np.ones((N, num_classes), dtype=float)
            out /= out.sum(axis=1, keepdims=True)
            return out

        num_classes = max(len(self._classes), getattr(self, "num_classes", len(self._classes)))
        proba_sum = np.zeros((N, num_classes), dtype=float)

        for exp in self.experts:
            wc = exp.weight_class
            if len(wc) < num_classes:
                wc = np.pad(wc, (0, num_classes - len(wc)), "constant", constant_values=1.0)

            p = self._safe_predict_proba(exp.estimator, X)
            if p is None:
                pred = self._safe_predict(exp.estimator, X)
                if pred is None:
                    continue
                pred = pred.astype(int)
                p = np.zeros((N, num_classes), dtype=float)
                pred = np.clip(pred, 0, num_classes - 1)
                p[np.arange(N), pred] = 1.0

            if p.shape[1] < num_classes:
                p = np.pad(p, ((0, 0), (0, num_classes - p.shape[1])), "constant", constant_values=0.0)

            # wc je "per-class" váha – dáva zmysel aplikovať na stĺpce
            proba_sum += p * wc[:num_classes]

        row_sums = proba_sum.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba_sum /= row_sums
        return proba_sum

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        # --- triedy ---
        if self._classes is None and classes is not None:
            self._classes = list(classes)
        elif classes is None and self._classes is not None:
            classes = self._classes

        N, D = X.shape
        self.window_size = self.period

        if self._warmup_samples is None:
            self._warmup_samples = self.warmup_windows * self.window_size

        # init batchov (y_batch_experts musí byť -1 pre invalid)
        if (
            self.p <= 0
            or self.X_batch is None
            or self.y_batch is None
            or self.y_batch_experts is None
            or self.y_batch_experts.shape[0] != len(self.experts)
            or self.y_batch_experts.shape[1] != self.window_size
        ):
            self.X_batch = np.zeros((self.window_size, D), dtype=float)
            self.y_batch = np.zeros(self.window_size, dtype=int)
            self.y_batch_experts = np.full((len(self.experts), self.window_size), -1, dtype=int)
            if self.p < 0:
                self.p = 0

        self.epochs += 1

        known_classes_count = len(self._classes) if self._classes is not None else 0
        current_max_class = int(np.max(y)) + 1 if len(y) > 0 else 0
        self.num_classes = max(known_classes_count, current_max_class)
        if self._classes is not None and len(self._classes) < self.num_classes:
            self._classes = list(range(self.num_classes))

        # lifetime +1 raz na vzorku
        for exp in self.experts:
            exp.lifetime += 1
            if getattr(exp, "warmup_remaining", 0) > 0:
                exp.warmup_remaining -= 1

        # ochrana pred pretečením váh
        if self.epochs % 100 == 0:
            total_weight_sum = sum(float(np.sum(exp.weight_class)) for exp in self.experts)
            if total_weight_sum > 0:
                for exp in self.experts:
                    exp.weight_class = exp.weight_class / total_weight_sum

        # ====== SPRÁVNE HLASOVANIE (FIX) ======
        predictions_class = np.zeros((self.num_classes,), dtype=float)

        for i, exp in enumerate(self.experts):
            if len(exp.weight_class) < self.num_classes:
                exp.weight_class = np.pad(
                    exp.weight_class,
                    (0, self.num_classes - len(exp.weight_class)),
                    "constant",
                    constant_values=(1.0 / max(1, len(self.experts))),
                )

            y_hat = self._safe_predict(exp.estimator, X)
            if y_hat is None or len(y_hat) == 0:
                self.y_batch_experts[i, self.p] = -1
                continue

            pred_c = int(y_hat[0])
            pred_c = int(np.clip(pred_c, 0, self.num_classes - 1))
            self.y_batch_experts[i, self.p] = pred_c

            # minorita z doterajšieho okna (bez current sample)
            counts = np.bincount(self.y_batch[: self.p].astype(int), minlength=self.num_classes)
            present = np.where(counts > 0)[0]
            if len(present) == 0:
                minority_cls = 1
            else:
                minority_cls = int(present[np.argmin(counts[present])])

            # update len pre triedu, ktorú expert predikoval
            if pred_c == int(y[0]):
                gain = 1.0 + self.beta * (2.0 if int(y[0]) == minority_cls else 1.0) * 0.05
                exp.weight_class[pred_c] = np.clip(exp.weight_class[pred_c] * gain, 1e-4, 1e4)
            else:
                loss = 1.0 - self.beta * 0.02
                exp.weight_class[pred_c] = np.clip(exp.weight_class[pred_c] * loss, 1e-4, 1e4)

            # hlasovanie: expert prispieva len do triedy, ktorú predikoval
            predictions_class[pred_c] += float(exp.weight_class[pred_c])

        # uloženie vzorky do okna
        self.X_batch[self.p] = X[0]
        self.y_batch[self.p] = int(y[0])

        self.p += 1
        y_hat_final = np.array([int(np.argmax(predictions_class))])

        # =========================
        # UPDATE PO OKNE
        # =========================
        if self.p >= self.window_size:
            self.p = 0
            update_start_time = time.time()

            # ---- adaptívna RWA sila (bez hard-cap 0.3) ----
            final_rwa_strength = 0.0
            imbalance_ratio = 0.0

            if self.rwa_strength > 0:
                class_counts = pd.Series(self.y_batch).value_counts()
                if len(class_counts) > 1 and class_counts.min() > 0:
                    imbalance_ratio = float(class_counts.max() / class_counts.min())
                    # smooth scaling: 0..1, saturuje postupne (zachová rozdiel 0.5/1.0/1.5)
                    scale = np.tanh((imbalance_ratio - 1.0) / 4.0)
                    final_rwa_strength = float(self.rwa_strength * scale)
                else:
                    final_rwa_strength = 0.0

                print(
                    f"DEBUG: Imbalance Ratio: {imbalance_ratio:.2f}, Adaptívna RWA Sila: {final_rwa_strength:.2f}"
                )

            expert_rwa_scores = []
            if self.rwa_strength > 0 or self.use_lifetime_trend:
                for i in range(len(self.experts)):
                    preds = self.y_batch_experts[i, :]
                    valid_idx = np.where(preds != -1)[0]
                    rwa = (
                        calculate_rwa(self.y_batch[valid_idx], preds[valid_idx], self._classes)
                        if len(valid_idx) > 0
                        else 0.0
                    )
                    expert_rwa_scores.append(rwa)
                    self.experts[i].rwa_history.append(rwa)

            if self.enable_diversity and len(self.experts) > 1:
                self._calculate_diversity(self.y_batch_experts, self.y_batch)
            else:
                self.div = []

            # úprava váh expertov
            for i, exp in enumerate(self.experts):
                # WARM-START: prvé okná experta iba jemne decayuj, bez RWA/div penalties
                if getattr(exp, "warmup_remaining", 0) > 0:
                    exp.weight_class *= (1.0 - self.alpha)  # mierny decay
                    exp.weight_class[exp.weight_class <= 0] = 1e-4
                    continue

                if self.use_lifetime_trend and len(exp.rwa_history) >= 5:
                    recent_avg = np.mean(list(exp.rwa_history)[-2:])
                    past_avg = np.mean(list(exp.rwa_history)[:-2])
                    trend = recent_avg - past_avg
                    lifetime_modifier = 1 - trend
                    exp.weight_class *= lifetime_modifier
                else:
                    exp.weight_class *= (1.0 - self.alpha * 5)

                if len(self.div) > 0:
                    exp.weight_class *= (1 - self.div[i])

                if final_rwa_strength > 0 and i < len(expert_rwa_scores):
                    # center okolo 0.5 → lepší expert dostane boost
                    rwa_bonus = 1.0 + (final_rwa_strength * (expert_rwa_scores[i] - 0.5))
                    exp.weight_class *= rwa_bonus

                #if not self.enable_diversity:
                #    mean_w = np.mean(exp.weight_class)
                #    exp.weight_class = 0.7 * exp.weight_class + 0.3 * mean_w

                exp.weight_class[exp.weight_class <= 0] = 1e-4

            # =========================
            # IMPLICIT DIVERZITA: TYPE BALANCING (1x per window)
            # (nahrádza pairwise diversity, ale drží heterogénny pool "živý")
            # =========================
            type_counts = {}
            for e in self.experts:
                type_counts[e.model_type] = type_counts.get(e.model_type, 0) + 1

            # mäkký penalty je stabilnejší než 1/count
            for e in self.experts:
                penalty = 1.0 / np.sqrt(float(type_counts[e.model_type]))
                e.weight_class *= penalty
                e.weight_class = np.clip(e.weight_class, 1e-4, 1e4)

            # normalizácia + nájdi najslabšieho experta
            sum_weight_class = np.zeros((self.num_classes,), dtype=float)
            weakest_expert_index, weakest_expert_weight_class = None, float("inf")

            for i, exp in enumerate(self.experts):
                sum_weight_class[: len(exp.weight_class)] += exp.weight_class[: self.num_classes]

                # WARM-START: počas warmupu experta nepovažuj za kandidáta na vyhodenie
                if getattr(exp, "warmup_remaining", 0) > 0:
                    continue

                s = float(np.sum(exp.weight_class))
                if s <= weakest_expert_weight_class:
                    weakest_expert_weight_class = s
                    weakest_expert_index = i

            self._normalize_weights_class(sum_weight_class)

            # =========================
            # WEIGHTED ENSEMBLE ERROR (FIX)
            # hodnotí sa rovnaké hlasovanie ako používa predict()
            # =========================
            ensemble_preds = []

            for k in range(self.window_size):

                vote = np.zeros(self.num_classes, dtype=float)

                for i, exp in enumerate(self.experts):
                    pred = self.y_batch_experts[i, k]

                    if pred == -1:
                        continue

                    pred = int(np.clip(pred, 0, self.num_classes - 1))

                    if pred < len(exp.weight_class):
                        vote[pred] += float(exp.weight_class[pred])

                if vote.sum() == 0:
                    ensemble_preds.append(0)
                else:
                    ensemble_preds.append(int(np.argmax(vote)))

            ensemble_preds = np.array(ensemble_preds, dtype=int)

            # vážený error (inverse-frequency)
            y_true = self.y_batch.astype(int)
            counts = np.bincount(y_true, minlength=self.num_classes).astype(float)

            w = np.zeros(self.num_classes, dtype=float)
            present = counts > 0
            w[present] = 1.0 / counts[present]
            wsum = float(np.sum(w))
            if wsum > 0:
                w /= wsum

            adaptive_threshold = 0.35 + 0.1 * np.tanh(imbalance_ratio / 5.0)

            window_error = np.mean((ensemble_preds != y_true) * w[y_true])

            # trigger: pridaj/odober experta
            if window_error > adaptive_threshold:
                if len(self.experts) >= self.max_estimators and weakest_expert_index is not None:
                    self.experts.pop(weakest_expert_index)

                if len(self.experts) < self.max_estimators:
                    new_exp = self._construct_new_expert()
                    new_exp.lifetime = 0
                    new_exp.warmup_remaining = int(self._warmup_samples or 0)
                    self.experts.append(new_exp)

                self.y_batch_experts = np.full((len(self.experts), self.window_size), -1, dtype=int)

            self._remove_experts_class()

            if len(self.experts) < self.min_estimators:
                new_exp = self._construct_new_expert()
                new_exp.lifetime = 0
                new_exp.warmup_remaining = int(self._warmup_samples or 0)
                self.experts.append(new_exp)
                self.y_batch_experts = np.full((len(self.experts), self.window_size), -1, dtype=int)

            update_end_time = time.time()
            self.update_times.append(update_end_time - update_start_time)

        # ONLINE tréning: raz na vzorku
        for exp in self.experts:
            exp.estimator = self.train_model(exp.estimator, X, y, self._classes, sample_weight)

        return y_hat_final

    def reset(self):
        self.epochs = 0
        self.p = -1
        self.window_size = None
        self.X_batch = None
        self.y_batch = None
        self.y_batch_experts = None
        self.div = []
        self.update_times = []

        self.num_classes = 2
        self.experts = []
        self._classes = None

        for _ in range(self.min_estimators):
            self.experts.append(self._construct_new_expert())

    def _normalize_weights_class(self, sum_weight_class):
        for exp in self.experts:
            for i in range(len(exp.weight_class)):
                if sum_weight_class[i] > 0:
                    exp.weight_class[i] /= sum_weight_class[i]

    def _calculate_diversity(self, y_experts, y):
        self.div = diversity.compute_pairwise_diversity(y, y_experts, diversity.Q_statistic)

    def _remove_experts_class(self):
        kept = []
        for ex in self.experts:
            # WARM-START: počas warmupu nikdy nevyhadzuj
            if getattr(ex, "warmup_remaining", 0) > 0:
                kept.append(ex)
                continue

            if sum(ex.weight_class) >= self.theta * self.num_classes:
                kept.append(ex)
        self.experts = kept

    def _construct_new_expert(self):
        """
        Balanced sampling z poolu base_estimators:
        - spočíta aktuálne zastúpenie typov v ensemble (self.experts)
        - preferuje typy, ktoré sú najmenej zastúpené
        - v rámci vybraného typu vyberie náhodne jeden konkrétny estimator z base_estimators
        """

        # aktuálne počty typov v ensemble
        current_counts = Counter()
        for e in self.experts:
            # bezpečne, keby starí experti nemali model_type
            mt = getattr(e, "model_type", type(e.estimator).__name__)
            current_counts[mt] += 1

        # typy dostupné v poole
        pool_types = [type(est).__name__ for est in self.base_estimators]
        pool_type_counts = Counter(pool_types)

        # ak je pool prázdny (nemalo by nastať)
        if len(pool_types) == 0:
            x = np.random.randint(0, len(self.base_estimators))
            weight = 1.0 / (len(self.experts) + 1)
            return self.WeightedExpert(cp.deepcopy(self.base_estimators[x]), weight, self.num_classes)

        # nájdi minimálnu obsadenosť v ensemble len pre typy, ktoré existujú v poole
        min_count = None
        for t in pool_type_counts.keys():
            c = current_counts.get(t, 0)
            if min_count is None or c < min_count:
                min_count = c

        # kandidáti = typy s minimálnym zastúpením
        candidate_types = [t for t in pool_type_counts.keys() if current_counts.get(t, 0) == min_count]

        # vyber náhodne jeden z kandidátnych typov
        chosen_type = np.random.choice(candidate_types)

        # vyber náhodný estimator z base_estimators, ktorý má zvolený typ
        candidate_idx = [i for i, est in enumerate(self.base_estimators) if type(est).__name__ == chosen_type]
        x = int(np.random.choice(candidate_idx))

        # konštrukcia experta
        weight = 1.0 / (len(self.experts) + 1)
        return self.WeightedExpert(cp.deepcopy(self.base_estimators[x]), weight, self.num_classes)

    def train_model(self, model, X, y, classes, sample_weight=None):
        if y is None or len(y) == 0:
            return model

        try:
            try:
                model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
                return model
            except TypeError:
                model.partial_fit(X, y, classes=classes)
                return model
        except Exception:
            try:
                if len(np.unique(y)) < 2:
                    return model
                model.fit(X, y)
            except Exception:
                return model
            return model

    def get_params(self, deep=True):
        return {
            "min_estimators": self.min_estimators,
            "max_estimators": self.max_estimators,
            "period": self.period,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "enable_diversity": self.enable_diversity,
            "rwa_strength": self.rwa_strength,
            "use_lifetime_trend": self.use_lifetime_trend,
        }