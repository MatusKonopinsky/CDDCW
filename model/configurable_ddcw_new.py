"""
Configurable_DDCW (upravená verzia pre imbalance + drift + multiclass).

Hlavné rozšírenia oproti pôvodnej verzii:
- history buffer ako hyperparameter
- multi-class safe replay logika (neukladá len jednu minoritu, ale všetky non-majority triedy)
- režimy:
    - off
    - replay
    - augment
- lokálna augmentácia numerických čŕt pomocou malého gaussovského šumu
- augmentácia sa vie mierne prispôsobovať aktuálnemu imbalance v okne
- jednoduchý drift detector (Page-Hinkley nad 0/1 error streamom)
- drift-aware reset histórie
"""

import copy as cp
import time
from collections import deque, Counter, defaultdict

import numpy as np
from sklearn.exceptions import NotFittedError

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import (
    HoeffdingTreeClassifier,
    HoeffdingAdaptiveTreeClassifier,
    ExtremelyFastDecisionTreeClassifier,
)
from skmultiflow.neural_networks import PerceptronMask


class SimplePageHinkley:
    """
    Jednoduchý Page-Hinkley drift detector nad error streamom.
    value = 1.0 -> chyba
    value = 0.0 -> správna predikcia
    """
    def __init__(self, delta=0.005, threshold=20.0, alpha=0.999):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.t = 0

    def update(self, value):
        self.t += 1
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value
        self.cum_sum += value - self.mean - self.delta
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)
        return (self.cum_sum - self.min_cum_sum) > self.threshold


class Configurable_DDCW(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    class WeightedExpert:
        def __init__(self, estimator, weight, num_classes):
            self.estimator = estimator
            self.model_type = type(estimator).__name__
            self.weight_class = np.full(num_classes, weight, dtype=float)
            self.lifetime = 0
            self.warmup_remaining = 0

    def __init__(
        self,
        min_estimators=5,
        max_estimators=20,
        base_estimators=None,
        period=600,
        alpha=0.002,
        beta=1.5,
        theta=0.02,
        enable_diversity=False,
        rwa_strength=1.0,
        use_lifetime_trend=True,
        warmup_windows=2,

        # nové parametre histórie
        history_buffer_size=600,
        class_buffer_size=300,

        # replay / augment
        replay_mode="replay",          # "off" | "replay" | "augment"
        replay_k=3,
        augmentation_mode="none",      # "none" | "noise"
        augmentation_strength=0.02,
        imbalance_aware_augmentation=True,

        # drift-aware logika
        enable_drift_detector=False,
        drift_delta=0.005,
        drift_threshold=20.0,
        post_drift_cooldown=300,
        post_drift_replay_boost=1,
        post_drift_aug_reduction=0.5,
        reset_majority_history_on_drift=True,
        keep_class_buffers_on_drift=True,

        random_state=None,
    ):
        super().__init__()

        if base_estimators is None:
            base_estimators = [
                NaiveBayes(),
                HoeffdingTreeClassifier(),
                HoeffdingAdaptiveTreeClassifier(),
                ExtremelyFastDecisionTreeClassifier(),
                PerceptronMask(),
            ]

        self.min_estimators = int(min_estimators)
        self.max_estimators = int(max_estimators)
        self.base_estimators = base_estimators

        self.period = int(period)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)
        self.enable_diversity = bool(enable_diversity)
        self.rwa_strength = float(rwa_strength)
        self.use_lifetime_trend = bool(use_lifetime_trend)
        self.warmup_windows = int(warmup_windows)

        self.history_buffer_size = int(history_buffer_size)
        self.class_buffer_size = int(class_buffer_size)

        self.replay_mode = str(replay_mode)
        self.replay_k = int(replay_k)
        self.augmentation_mode = str(augmentation_mode)
        self.augmentation_strength = float(augmentation_strength)
        self.imbalance_aware_augmentation = bool(imbalance_aware_augmentation)

        self.enable_drift_detector = bool(enable_drift_detector)
        self.drift_delta = float(drift_delta)
        self.drift_threshold = float(drift_threshold)
        self.post_drift_cooldown = int(post_drift_cooldown)
        self.post_drift_replay_boost = int(post_drift_replay_boost)
        self.post_drift_aug_reduction = float(post_drift_aug_reduction)
        self.reset_majority_history_on_drift = bool(reset_majority_history_on_drift)
        self.keep_class_buffers_on_drift = bool(keep_class_buffers_on_drift)

        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

        self.num_classes = 2
        self._classes = None
        self.experts = []
        self.update_times = []

        self._y_window = deque(maxlen=self.period)
        self._history_buffer = deque(maxlen=self.history_buffer_size)

        self._warmup_samples = None
        self._seen_samples = 0

        # pre multiclass: buffer pre každú triedu zvlášť
        self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))

        self._post_drift_remaining = 0
        self._drift_points = []
        self._drift_detector = None

        self.reset()

    # ============================================================
    # SAFE WRAPPERS
    # ============================================================

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
        except Exception:
            pass
        return None

    # ============================================================
    # PARTIAL FIT
    # ============================================================

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if classes is not None and self._classes is None:
            self._classes = list(classes)
            self.num_classes = max(self.num_classes, len(self._classes))

        if X is None or y is None or len(y) == 0:
            return self

        for i in range(len(y)):
            Xi = X[i:i + 1]
            yi = y[i:i + 1]
            self.fit_single_sample(Xi, yi, classes=self._classes, sample_weight=sample_weight)

        return self

    # ============================================================
    # PREDIKCIA
    # ============================================================

    def predict(self, X):
        if not self.experts:
            return np.zeros(X.shape[0], dtype=int)

        pred_votes = np.zeros((X.shape[0], self.num_classes), dtype=float)

        for exp in self.experts:
            y_hat = self._safe_predict(exp.estimator, X)
            if y_hat is None:
                continue

            y_hat = np.clip(np.asarray(y_hat, dtype=int), 0, self.num_classes - 1)
            for i, c in enumerate(y_hat):
                pred_votes[i, c] += float(exp.weight_class[c])

        return np.argmax(pred_votes, axis=1)

    def predict_proba(self, X):
        N = X.shape[0]
        out = np.zeros((N, self.num_classes), dtype=float)

        for exp in self.experts:
            wc = exp.weight_class
            if len(wc) < self.num_classes:
                wc = np.pad(
                    wc,
                    (0, self.num_classes - len(wc)),
                    "constant",
                    constant_values=1.0
                )

            p = self._safe_predict_proba(exp.estimator, X)
            if p is None:
                pred = self._safe_predict(exp.estimator, X)
                if pred is None:
                    continue
                pred = np.clip(np.asarray(pred, dtype=int), 0, self.num_classes - 1)
                p = np.zeros((N, self.num_classes), dtype=float)
                p[np.arange(N), pred] = 1.0

            if p.shape[1] < self.num_classes:
                p = np.pad(
                    p,
                    ((0, 0), (0, self.num_classes - p.shape[1])),
                    "constant",
                    constant_values=0.0
                )

            out += p * wc[:self.num_classes]

        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        out /= row_sums
        return out

    # ============================================================
    # HELPERS
    # ============================================================

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

    def _window_counts(self):
        if len(self._y_window) == 0:
            return np.zeros(self.num_classes, dtype=int)
        return np.bincount(np.asarray(self._y_window, dtype=int), minlength=self.num_classes)

    def _get_majority_and_minorities(self):
        counts = self._window_counts()
        present = np.where(counts > 0)[0]
        if len(present) == 0:
            return None, []

        majority = int(present[np.argmax(counts[present])])
        minorities = [int(c) for c in present if int(c) != majority]
        return majority, minorities

    def _current_imbalance_ratio(self):
        counts = self._window_counts()
        present = counts[counts > 0]
        if len(present) < 2:
            return 1.0
        return float(np.max(present) / max(1, np.min(present)))

    def _local_feature_std(self):
        if len(self._history_buffer) < 2:
            return None

        X_hist = np.vstack([item[0] for item in self._history_buffer])
        std = np.std(X_hist, axis=0)
        std = np.where(std < 1e-8, 1e-8, std)
        return std

    def _effective_replay_k(self):
        k = self.replay_k
        if self._post_drift_remaining > 0:
            k += self.post_drift_replay_boost
        return max(0, int(k))

    def _effective_aug_strength(self):
        strength = self.augmentation_strength

        if self.imbalance_aware_augmentation:
            ratio = self._current_imbalance_ratio()
            strength *= min(3.0, 1.0 + 0.25 * max(0.0, ratio - 1.0))

        if self._post_drift_remaining > 0:
            strength *= self.post_drift_aug_reduction

        return max(0.0, float(strength))

    def _augment_sample(self, X, y):
        if self.augmentation_mode == "none":
            return X.copy(), y.copy()

        if self.augmentation_mode == "noise":
            std = self._local_feature_std()
            if std is None:
                return X.copy(), y.copy()

            sigma = self._effective_aug_strength()
            noise = self._rng.normal(loc=0.0, scale=sigma * std, size=X.shape)
            return (X + noise).astype(float), y.copy()

        return X.copy(), y.copy()

    def _construct_new_expert(self):
        idx = self._rng.randint(0, len(self.base_estimators))
        est = cp.deepcopy(self.base_estimators[idx])

        if isinstance(est, (HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier)):
            try:
                est.grace_period = int(self._rng.randint(20, 200))
            except Exception:
                pass

        if isinstance(est, PerceptronMask):
            try:
                est.learning_ratio = float(self._rng.uniform(0.001, 0.05))
            except Exception:
                pass

        weight = 1.0 / max(1, len(self.experts) + 1)
        ex = self.WeightedExpert(est, weight, self.num_classes)

        if self._warmup_samples is not None:
            ex.warmup_remaining = int(self._warmup_samples)

        return ex

    def _update_drift_detector(self, was_correct):
        if not self.enable_drift_detector or self._drift_detector is None:
            return False

        error_value = 0.0 if was_correct else 1.0
        drift = self._drift_detector.update(error_value)
        if drift:
            self._handle_drift()
        return drift

    def _handle_drift(self):
        self._drift_points.append(self._seen_samples)
        self._post_drift_remaining = self.post_drift_cooldown

        majority_class, minority_classes = self._get_majority_and_minorities()

        if self.reset_majority_history_on_drift:
            new_history = deque(maxlen=self.history_buffer_size)
            for Xh, yh in self._history_buffer:
                cls = int(yh[0])
                if cls in minority_classes:
                    new_history.append((Xh, yh))
            self._history_buffer = new_history
            self._y_window = deque(maxlen=self.period)

        if not self.keep_class_buffers_on_drift:
            self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))
        else:
            new_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))
            for cls, buf in self.class_buffers.items():
                keep_n = max(1, self.class_buffer_size // 2)
                tail = list(buf)[-keep_n:]
                new_buffers[int(cls)] = deque(tail, maxlen=self.class_buffer_size)
            self.class_buffers = new_buffers

        if self._drift_detector is not None:
            self._drift_detector.reset()

    def _get_replay_pool(self):
        majority_class, minority_classes = self._get_majority_and_minorities()
        replay_pool = []

        for cls in minority_classes:
            if cls in self.class_buffers and len(self.class_buffers[cls]) > 0:
                replay_pool.extend(list(self.class_buffers[cls]))

        return replay_pool, majority_class, minority_classes

    # ============================================================
    # SINGLE SAMPLE UPDATE
    # ============================================================

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        t0 = time.time()

        if self._classes is None and classes is not None:
            self._classes = list(classes)

        if self._classes is not None:
            self.num_classes = max(self.num_classes, len(self._classes))
        if y is not None and len(y) > 0:
            self.num_classes = max(self.num_classes, int(np.max(y)) + 1)

        if self._warmup_samples is None:
            self._warmup_samples = self.warmup_windows * self.period

        for exp in self.experts:
            if len(exp.weight_class) < self.num_classes:
                fill_value = float(np.mean(exp.weight_class) if len(exp.weight_class) else 1.0)
                exp.weight_class = np.pad(
                    exp.weight_class,
                    (0, self.num_classes - len(exp.weight_class)),
                    "constant",
                    constant_values=fill_value,
                )

        true_c = int(y[0])

        self._history_buffer.append((X.copy(), y.copy()))
        self._y_window.append(true_c)

        majority_class, minority_classes = self._get_majority_and_minorities()

        if majority_class is not None and true_c != majority_class:
            self.class_buffers[true_c].append((X.copy(), y.copy()))

        ensemble_pred = int(self.predict(X)[0]) if len(self.experts) > 0 else true_c
        was_correct = (ensemble_pred == true_c)
        self._update_drift_detector(was_correct)

        # update váh expertov
        for exp in self.experts:
            exp.lifetime += 1
            if exp.warmup_remaining > 0:
                exp.warmup_remaining -= 1

            y_hat = self._safe_predict(exp.estimator, X)
            if y_hat is None or len(y_hat) == 0:
                continue

            pred_c = int(np.clip(int(y_hat[0]), 0, self.num_classes - 1))

            is_minority_sample = (majority_class is not None and true_c != majority_class)

            if pred_c == true_c:
                if is_minority_sample:
                    mult = 1.0 + self.beta * 0.40
                else:
                    mult = 1.0 + self.beta * 0.06
            else:
                if is_minority_sample:
                    mult = 1.0 - self.beta * 0.06
                else:
                    mult = 1.0 - self.beta * 0.02

            exp.weight_class[pred_c] = np.clip(exp.weight_class[pred_c] * mult, 1e-4, 1e4)

        if not self.enable_diversity and len(self.experts) > 1:
            type_counts = Counter(e.model_type for e in self.experts)
            for e in self.experts:
                penalty = 1.0 / np.sqrt(type_counts[e.model_type])
                e.weight_class *= penalty

        for exp in self.experts:
            exp.estimator = self.train_model(exp.estimator, X, y, self._classes, sample_weight)

        # replay / augment pre všetky non-majority triedy
        if self.replay_mode != "off":
            replay_pool, _, _ = self._get_replay_pool()
            if len(replay_pool) > 0:
                effective_k = self._effective_replay_k()

                for _ in range(effective_k):
                    idx = self._rng.randint(0, len(replay_pool))
                    Xr, yr = replay_pool[idx]

                    if self.replay_mode == "augment":
                        X_train, y_train = self._augment_sample(Xr, yr)
                    else:
                        X_train, y_train = Xr, yr

                    for exp in self.experts:
                        exp.estimator = self.train_model(exp.estimator, X_train, y_train, self._classes, sample_weight)

        self.experts = [
            e for e in self.experts
            if float(np.sum(e.weight_class)) >= self.theta * self.num_classes
        ]

        while len(self.experts) < self.min_estimators:
            self.experts.append(self._construct_new_expert())

        if len(self.experts) > self.max_estimators:
            sums = [float(np.sum(e.weight_class)) for e in self.experts]
            drop = int(np.argmin(sums))
            self.experts.pop(drop)

        self._seen_samples += 1

        if self._post_drift_remaining > 0:
            self._post_drift_remaining -= 1

        self.update_times.append(time.time() - t0)
        return self.predict(X)

    # ============================================================
    # RESET + PARAMS
    # ============================================================

    def reset(self):
        self.num_classes = 2
        self._classes = None
        self.experts = []
        self.update_times = []

        self._y_window = deque(maxlen=self.period)
        self._history_buffer = deque(maxlen=self.history_buffer_size)

        self._warmup_samples = None
        self._seen_samples = 0

        self.class_buffers = defaultdict(lambda: deque(maxlen=self.class_buffer_size))

        self._post_drift_remaining = 0
        self._drift_points = []

        if self.enable_drift_detector:
            self._drift_detector = SimplePageHinkley(
                delta=self.drift_delta,
                threshold=self.drift_threshold,
                alpha=0.999,
            )
        else:
            self._drift_detector = None

        for _ in range(self.min_estimators):
            self.experts.append(self._construct_new_expert())

        return self

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
            "warmup_windows": self.warmup_windows,
            "history_buffer_size": self.history_buffer_size,
            "class_buffer_size": self.class_buffer_size,
            "replay_mode": self.replay_mode,
            "replay_k": self.replay_k,
            "augmentation_mode": self.augmentation_mode,
            "augmentation_strength": self.augmentation_strength,
            "imbalance_aware_augmentation": self.imbalance_aware_augmentation,
            "enable_drift_detector": self.enable_drift_detector,
            "drift_delta": self.drift_delta,
            "drift_threshold": self.drift_threshold,
            "post_drift_cooldown": self.post_drift_cooldown,
            "post_drift_replay_boost": self.post_drift_replay_boost,
            "post_drift_aug_reduction": self.post_drift_aug_reduction,
            "reset_majority_history_on_drift": self.reset_majority_history_on_drift,
            "keep_class_buffers_on_drift": self.keep_class_buffers_on_drift,
            "random_state": self.random_state,
        }