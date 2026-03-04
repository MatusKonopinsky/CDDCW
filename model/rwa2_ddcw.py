import copy as cp
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from utils import diversity
from utils.rwa_metric import calculate_rwa
import time

class RWA_DDCW_Classifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Diversified Dynamic Class Weighted ensemble classifier.

    Parameters
    ----------
    min_estimators: int (default=5)
        Minimum number of estimators to hold.
    max_estimators: int (default=20)
        Maximum number of estimatorst to hold.
    base_estimators: List of StreamModel or sklearn.BaseEstimator (default=[NaiveBayes(), HoeffdingTreeClassifier())
        Each member of the ensemble is an instance of the base estimator.
    period: int (default=100)
        Period between expert removal, creation, and weight update.
    alpha: float (default=0.02)
        Factor for which to decrease weights on experts lifetime
    beta: float (default=3)
        Factor for which to increase weights by.
    theta: float (default=0.02)
        Minimum fraction of weight per model.
    enable_diversity: bool (default=True)
        If true, calculate diversity of experts and weights update.

    Notes
    -----
    The diversified dynamic class weighted (DDCW) [1]_, uses five mechanisms to
    cope with concept drift: It trains online learners of the ensemble, it uses weights per class,
    it update class weights for those learners based on their performance diversity and time spend in ensemble,
    it removes them, also based on their performance, and it adds new experts based on the
    global performance of the ensemble.

    """

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its class weights.

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: float
            The estimator's weight.
        num_classes: int
            The number of actual target classes
        """
        def __init__(self, estimator, weight, num_classes):
            self.estimator = estimator
            self.weight_class = np.full(num_classes, weight, dtype=float)
            self.lifetime = 0


    def __init__(self, min_estimators=5, max_estimators=20, base_estimators=[NaiveBayes(), HoeffdingTreeClassifier()],
                 period=1000, alpha=0.002, beta=1.5, theta=0.05, enable_diversity=True):
        """
        Creates a new instance of DiversifiedDynamicClassWeightedClassifier.
        """
        super().__init__()

        self.enable_diversity = enable_diversity
        self.min_estimators = min_estimators
        self.max_estimators = max_estimators
        self.base_estimators = base_estimators

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.period = period

        self.p = -1

        self.n_estimators = max_estimators
        self.epochs = None
        self.num_classes = None
        self.experts = None
        self.div = []

        self.window_size = None
        self.X_batch = None
        self.y_batch = None
        self.y_batch_experts = None

        # custom measurements atributes
        self.custom_measurements = []
        self.custom_time = []

        self._classes = None # Sem si uložíme zoznam tried

        self.reset()


    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Returns
        -------
        DiversifiedDynamicClassWeightedClassifier
            self
        """
        # Uložíme si zoznam tried pri prvom volaní, aby bol dostupný pre nových expertov
        if self._classes is None and classes is not None:
            self._classes = list(classes)

        # Iterujeme cez každú vzorku a voláme fit_single_sample
        for i in range(len(X)):
            self.fit_single_sample(
                X[i:i+1, :], y[i:i+1], self._classes, sample_weight
            )
 
        return self

    def predict(self, X):
        """ predict

        The predict function will take an average of the predictions of its
        learners, weighted by their respective class weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        predictions_class = np.zeros((len(X), self.num_classes))
        for exp in self.experts:
            if exp.lifetime > 0:
                Y_hat = exp.estimator.predict(X)
                for i, y_hat in enumerate(Y_hat):
                    predictions_class[i][y_hat] += exp.weight_class[y_hat]
        y_hat_final = np.argmax(predictions_class, axis=1)
        return y_hat_final


    def predict_proba(self, X):
        raise NotImplementedError

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Fits a single sample of shape `X.shape=(1, n_attributes)` and `y.shape=(1)`

        Aggregates all experts' predictions, diminishes weight of experts whose
        predictions were wrong, and may create or remove experts every _period_
        samples.

        Finally, trains each individual expert on the provided data.
        """
        # Uistenie sa, že _classes je nastavené
        if self._classes is None and classes is not None:
            self._classes = list(classes)
        elif classes is None and self._classes is not None:
            classes = self._classes

        # Prvotný tréning expertov pri inicializácii
        if self.p <= 0:
            for exp in self.experts:
                exp.estimator = self.train_model(cp.copy(exp.estimator), X, y, self._classes, sample_weight)

        N, D = X.shape
        self.window_size = self.period

        # Dynamická kontrola a reinicializácia bufferov
        if self.p <= 0 or self.y_batch_experts is None or self.y_batch_experts.shape[0] != len(self.experts):
            self.X_batch = np.zeros((self.window_size, D), dtype=float)
            self.y_batch = np.zeros(self.window_size, dtype=int)
            self.y_batch_experts = np.zeros((len(self.experts), self.window_size), dtype=int)
            if self.p < 0:
                self.p = 0

        self.epochs += 1
        
        # Aktualizácia počtu tried
        known_classes_count = len(self._classes) if self._classes is not None else 0
        current_max_class = int(np.max(y)) + 1 if len(y) > 0 else 0
        self.num_classes = max(known_classes_count, current_max_class)
        if self._classes is not None and len(self._classes) < self.num_classes:
            self._classes = list(range(self.num_classes))

        predictions_class = np.zeros((self.num_classes,))

        # Zber predikcií od expertov
        for i, exp in enumerate(self.experts):
            y_hat = exp.estimator.predict(X)
            if y_hat is None or len(y_hat) == 0:
                # Ak expert nevie predikovať, preskočíme ho a uložíme neplatnú hodnotu
                self.y_batch_experts[i, self.p] = -1
                continue

            self.y_batch_experts[i, self.p] = y_hat[0]

            if len(exp.weight_class) < self.num_classes:
                exp.weight_class = np.pad(exp.weight_class, (0, self.num_classes - len(exp.weight_class)), 'constant', constant_values=(1.0 / len(self.experts)))

            # Odmena experta, ak uhádol správne (len ak sa má meniť váha v tomto bloku)
            if np.any(y_hat == y) and (self.epochs % self.period == 0):
                exp.weight_class[y_hat[0]] = exp.weight_class[y_hat[0]] * self.beta
                
            predictions_class[y_hat[0]] += exp.weight_class[y_hat[0]]

        self.X_batch[self.p] = X[0]
        self.y_batch[self.p] = y[0]
        self.p += 1
        
        y_hat_final = np.array([np.argmax(predictions_class)])

        # Hlavná logika sa spúšťa po naplnení okna
        if self.p >= self.window_size:
            self.p = 0  # Reset počítadla pre ďalšie okno
            
            # 1. Štandardná úprava váh (životnosť, diverzita)
            if len(self.experts) > 1:
                self._calculate_diversity(self.y_batch_experts, self.y_batch)
            else:
                self.div = []

            for i, exp in enumerate(self.experts):
                exp.lifetime += 1
                exp.weight_class = exp.weight_class - (np.exp(self.alpha * exp.lifetime) - 1) / 10
                if len(self.div) > 0 and self.enable_diversity:
                    exp.weight_class = exp.weight_class * (1 - self.div[i])
                exp.weight_class[exp.weight_class <= 0] = 0.0001
            
            # --- ZAČIATOK NOVÉHO PRAVIDLA: RWA AKO BONUS ---
            expert_rwa_scores = []
            for i in range(len(self.experts)):
                expert_predictions = self.y_batch_experts[i, :]
                # Odfiltrujeme neplatné predikcie, ak nejaké sú
                valid_indices = np.where(expert_predictions != -1)[0]
                if len(valid_indices) > 0:
                    rwa = calculate_rwa(self.y_batch[valid_indices], expert_predictions[valid_indices], self._classes)
                else:
                    rwa = 0.0
                expert_rwa_scores.append(rwa)

            if len(expert_rwa_scores) > 0:
                print(f"DEBUG (vzorka {self.epochs}): Aplikujem RWA bonusy. RWA skóre expertov: {[f'{s:.2f}' for s in expert_rwa_scores]}")
                for i, exp in enumerate(self.experts):
                    # Váhy experta sú vynásobené jeho RWA skóre
                    exp.weight_class = exp.weight_class * (expert_rwa_scores[i] + 0.01) # Smoothing
            # --- KONIEC NOVÉHO PRAVIDLA ---

            # 2. Nájdeme najslabšieho experta na základe NOVÝCH, upravených váh
            sum_weight_class = np.zeros((self.num_classes,))
            weakest_expert_index = None
            weakest_expert_weight_class = float('inf')
            for i, exp in enumerate(self.experts):
                for j in range(len(exp.weight_class)):
                    sum_weight_class[j] += exp.weight_class[j]
                current_score = sum(exp.weight_class)
                if current_score <= weakest_expert_weight_class:
                    weakest_expert_index = i
                    weakest_expert_weight_class = current_score
            
            # 3. Normalizácia a pôvodná logika DDCW na pridávanie/odstraňovanie
            self._normalize_weights_class(sum_weight_class)

            if np.any(y_hat_final != y):
                if len(self.experts) >= self.max_estimators and weakest_expert_index is not None:
                    if weakest_expert_index < len(self.experts):
                        self.experts.pop(weakest_expert_index)
                
                if len(self.experts) < self.max_estimators:
                    new_exp = self._construct_new_expert()
                    self.experts.append(new_exp)
                    self.y_batch_experts = np.zeros((len(self.experts), self.window_size), dtype=int)

            self._remove_experts_class()

            if len(self.experts) < self.min_estimators:
                new_exp = self._construct_new_expert()
                self.experts.append(new_exp)
                self.y_batch_experts = np.zeros((len(self.experts), self.window_size), dtype=int)
                
        # 4. Finálny tréning všetkých expertov na aktuálnej vzorke
        for exp in self.experts:
            random_weights = []
            random_weight = np.random.randint(len(self.experts)) if len(self.experts) > 0 else 0
            random_weights.append(random_weight)
            if random_weight == 0: random_weights = None
            
            exp.estimator = self.train_model(exp.estimator, X, y, self._classes, random_weights)

        # Ukladanie custom meraní
        if self.p == 0:
            diversity_value = np.mean(self.div) if len(self.div) > 0 else 0
            data = {'id_period': self.epochs / self.period, 'n_experts': len(self.experts), 'diversity': diversity_value}
            self.custom_measurements.append(data)

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts, n_samples)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2    # Minimum of 2 classes
        self.experts = []
        self._classes = None    # Tiež resetujeme zoznam tried
        
        # Opravené volanie - už bez argumentu
        for i in range(self.min_estimators):
            self.experts.append(self._construct_new_expert())

    def _normalize_weights_class(self, sum_weight_class):
        """
        Normalize the experts' weights such that the sum per class is 1.
        """

        for exp in self.experts:
            for i in range(len(exp.weight_class)):
                exp.weight_class[i] /= sum_weight_class[i]

    def _calculate_diversity(self, y_experts, y):
        """
        Calculate Q stat pairwise diversity in actual model
        """
        self.div = diversity.compute_pairwise_diversity(y, y_experts, diversity.Q_statistic)

    def _remove_experts_class(self):
        """
        Removes all experts whose score (sum weights per class) is lower than self.theta.
        """
        self.experts = [ex for ex in self.experts if sum(ex.weight_class) >= self.theta * self.num_classes]

    def _construct_new_expert(self):
        """
        Constructs a new WeightedExpert randomly from list of provided base_estimators.
        """
        x = np.random.randint(0, len(self.base_estimators))
        
        # Zistí si počet expertov samo, pridá 1 pre výpočet váhy nového
        num_current_experts = len(self.experts)
        # Váha je 1 / (počet existujúcich + 1 (ten nový))
        weight = 1.0 / (num_current_experts + 1)
        
        return self.WeightedExpert(cp.deepcopy(self.base_estimators[x]), weight, self.num_classes)

    @staticmethod
    def train_model(model, X, y, classes=None, sample_weight=None):
        """ Trains a model, taking care of the fact that either fit or partial_fit is implemented
        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The model to train
        X: numpy.ndarray of shape (n_samples, n_features)
            The data chunk
        y: numpy.array of shape (n_samples)
            The labels in the chunk
        classes: list or numpy.array
            The unique classes in the data chunk
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        Returns
        -------
        StreamModel or sklearn.BaseEstimator
            The trained model
        """
        try:
            model.partial_fit(X, y, classes, sample_weight)
        except NotImplementedError:
            model.fit(X, y)
        return model
