import numpy as np
import copy as cp

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import cohen_kappa_score


class KUE(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """
    Faithful adaptation of Kappa Updated Ensemble (KUE) classifier
    for data streams.

    Key properties:
    - Chunk-based update
    - Evaluate-then-train (prequential)
    - Kappa-based weighting
    - Worst expert replacement
    - Online bagging (Poisson)
    - Random feature subspaces
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        chunk_size=1000,
        subspace_size=0.6,
        lam=1.0,
        random_state=None,
    ):
        super().__init__()

        self.base_estimator = base_estimator if base_estimator else HoeffdingTreeClassifier()
        self.n_estimators = n_estimators
        self.chunk_size = chunk_size
        self.subspace_size = subspace_size  # percent of features
        self.lam = lam  # poisson lambda for online bagging
        self.random_state = random_state

        self._rng = np.random.RandomState(random_state)

        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):
        self.ensemble = []
        self.weights = []
        self.subspaces = []

        self._X_chunk = []
        self._y_chunk = []

        self._classes = None
        self._class_to_idx = {}

    # ============================================================
    # PARTIAL FIT
    # ============================================================

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        if self._classes is None:
            if classes is None:
                raise ValueError("classes must be provided at first partial_fit.")
            self._classes = list(classes)
            self._class_to_idx = {c: i for i, c in enumerate(self._classes)}

        for i in range(X.shape[0]):
            self._X_chunk.append(X[i])
            self._y_chunk.append(y[i])

            if len(self._X_chunk) >= self.chunk_size:
                self._update_ensemble()

        return self

    # ============================================================
    # CREATE RANDOM SUBSPACE
    # ============================================================

    def _sample_subspace(self, n_features):
        k = max(1, int(self.subspace_size * n_features))
        return self._rng.choice(np.arange(n_features), size=k, replace=False)

    # ============================================================
    # UPDATE ENSEMBLE (CORE OF KUE)
    # ============================================================

    def _update_ensemble(self):

        Xc = np.array(self._X_chunk)
        yc = np.array(self._y_chunk)

        n_features = Xc.shape[1]

        # --------------------------------------------------------
        # INITIAL FILL
        # --------------------------------------------------------
        if not self.ensemble:

            for _ in range(self.n_estimators):
                clf = cp.deepcopy(self.base_estimator)

                subspace = self._sample_subspace(n_features)
                self.subspaces.append(subspace)

                X_sub = Xc[:, subspace]

                # Online bagging
                k = self._rng.poisson(self.lam, size=len(X_sub))
                sample_weight = np.maximum(k, 0)

                clf.partial_fit(
                    X_sub,
                    yc,
                    classes=self._classes,
                    sample_weight=sample_weight,
                )

                self.ensemble.append(clf)

            self.weights = np.ones(self.n_estimators) / self.n_estimators
            self._clear_chunk()
            return

        # --------------------------------------------------------
        # 1️⃣ EVALUATE CURRENT EXPERTS (BEFORE TRAINING)
        # --------------------------------------------------------
        kappas = []

        for clf, subspace in zip(self.ensemble, self.subspaces):
            X_sub = Xc[:, subspace]
            y_pred = clf.predict(X_sub)

            kappa = cohen_kappa_score(yc, y_pred)
            kappas.append(max(0.0, kappa))

        # --------------------------------------------------------
        # 2️⃣ TRAIN EXISTING EXPERTS (ONLINE BAGGING)
        # --------------------------------------------------------
        for clf, subspace in zip(self.ensemble, self.subspaces):

            X_sub = Xc[:, subspace]

            k = self._rng.poisson(self.lam, size=len(X_sub))
            sample_weight = np.maximum(k, 0)

            clf.partial_fit(
                X_sub,
                yc,
                classes=self._classes,
                sample_weight=sample_weight,
            )

        # --------------------------------------------------------
        # 3️⃣ TRAIN NEW CANDIDATE
        # --------------------------------------------------------
        new_clf = cp.deepcopy(self.base_estimator)
        new_subspace = self._sample_subspace(n_features)

        X_new = Xc[:, new_subspace]

        k = self._rng.poisson(self.lam, size=len(X_new))
        sample_weight = np.maximum(k, 0)

        new_clf.partial_fit(
            X_new,
            yc,
            classes=self._classes,
            sample_weight=sample_weight,
        )

        y_pred_new = new_clf.predict(X_new)
        kappa_new = max(0.0, cohen_kappa_score(yc, y_pred_new))

        # --------------------------------------------------------
        # 4️⃣ REPLACE WORST IF BETTER
        # --------------------------------------------------------
        worst_idx = int(np.argmin(kappas))

        if kappa_new > kappas[worst_idx]:
            self.ensemble[worst_idx] = new_clf
            self.subspaces[worst_idx] = new_subspace
            kappas[worst_idx] = kappa_new

        # --------------------------------------------------------
        # 5️⃣ UPDATE WEIGHTS
        # --------------------------------------------------------
        sum_kappa = np.sum(kappas)

        if sum_kappa > 0:
            self.weights = np.array(kappas) / sum_kappa
        else:
            self.weights = np.ones(len(self.ensemble)) / len(self.ensemble)

        self._clear_chunk()

    # ============================================================
    # CLEAR CHUNK
    # ============================================================

    def _clear_chunk(self):
        self._X_chunk = []
        self._y_chunk = []

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(self, X):

        if not self.ensemble:
            return np.array([self._classes[0]] * X.shape[0])

        votes = np.zeros((X.shape[0], len(self._classes)))

        for w, clf, subspace in zip(self.weights, self.ensemble, self.subspaces):

            preds = clf.predict(X[:, subspace])

            for i, p in enumerate(preds):
                votes[i, self._class_to_idx[p]] += w

        idx = np.argmax(votes, axis=1)
        return np.array([self._classes[i] for i in idx])

    # ============================================================
    # PREDICT PROBA
    # ============================================================

    def predict_proba(self, X):

        if not self.ensemble:
            return np.ones((X.shape[0], len(self._classes))) / len(self._classes)

        probas = np.zeros((X.shape[0], len(self._classes)))

        for w, clf, subspace in zip(self.weights, self.ensemble, self.subspaces):
            p = clf.predict_proba(X[:, subspace])
            probas += p * w

        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probas / row_sums