from .base import UnsupervisedDriftDetector

import numpy as np
from scipy.stats import ttest_rel
from sklearn.svm import SVC


def svm_independence_test(X, y, n_itr=10, p_val=0.00001, n_sel=50):
    Z = np.concatenate((X, np.linspace(0, 1, X.shape[0]).reshape(-1, 1)),
                       axis=1)
    svm = SVC(gamma=2, C=1, kernel="rbf")
    s1, s2 = [], []
    for _ in range(n_itr):
        sel = np.random.choice(range(Z.shape[0]),
                               size=min(n_sel, int(2 * Z.shape[0] / 3)),
                               replace=False)
        if len(np.unique(
                y[sel])) == 1:  # Number classes has to be greater than one!
            continue
        y = np.rint(y).astype(int)  # Ensure y contains discrete labels
        svm.fit(Z[sel], y[sel].ravel())
        s1.append(svm.score(Z, y.ravel()))
        s2.append(svm.score(
            np.concatenate((X, np.random.random(X.shape[0]).reshape(-1, 1)),
                           axis=1), y.ravel()))

    if len(s1) == 0 or len(s2) == 0:
        return True
    elif (np.array(s1) - np.array(s2)).var() == 0:
        return abs(np.mean(s1) - np.mean(s2)) < 0.000001
    else:
        return ttest_rel(s1, s2)[1] > p_val


def test_independence(X, Y, Z=None):
    return svm_independence_test(X, Y)


class DAWIDD(UnsupervisedDriftDetector):
    """
    Implementation of the dynamic-adapting-window-independence-drift-detector (DAWIDD)
    
    Parameters
    ----------
    max_window_size : int, optional
        The maximal size of the window. When reaching the maximal size, the oldest sample is removed.

        The default is 90
    min_window_size : int, optional
        The minimal number of samples that is needed for computing the hypothesis test.

        The default is 70
    min_p_value : int, optional
        The threshold of the p-value - not every test outputs a p-value (sometimes only 1.0 <=> independent and 0.0 <=> not independent are returned)

        The default is 0.001
    """

    def __init__(self, max_window_size=90, min_window_size=70,
                 min_p_value=0.001, seed=None, recent_samples_size: int = 500):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_p_value = min_p_value

        self.X = []
        self.n_items = 0
        self.min_n_items = self.min_window_size / 4.

        self.drift_detected = False

    """
    You have to overwrite this function if you want to use a 
    different test for independence
    """

    def _test_for_independence(self):

        t = np.array(range(self.n_items)) / (1. * self.n_items)
        t /= np.std(t)
        t = t.reshape(-1, 1)

        return 1.0 if test_independence(np.array(self.X), t) == True else 0.0

    def add_record(self, x):
        self.drift_detected = False

        # Add item
        self.X.append(x.flatten())
        self.n_items += 1

        if self.n_items == self.max_window_size:
            # Test for drift
            p = self._test_for_independence()

            if p <= self.min_p_value:
                self.drift_detected = True

            # Remove samples
            while self.n_items > self.min_window_size:
                """
                Remove old samples after min window size 
                (baseline data never removed)
                """
                self.X.pop(self.min_window_size)
                self.n_items -= 1

    def detected_change(self):
        return self.drift_detected

    def update(self, x) -> bool:
        x_array = np.array(list(x.values()))
        self.add_record(x_array)
        if self.detected_change():
            return True
        return False

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_stream(stream, n_training_samples, classifier_path)
