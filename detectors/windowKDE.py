from .base import UnsupervisedDriftDetector


import numpy as np
from sklearn.neighbors import KernelDensity
import math


def calDistance(v1, v2):  # L1-Distance
    return sum(map(lambda i, j: abs(i - j), v1, v2))


class WindowKDE(UnsupervisedDriftDetector):

    def __init__(
        self,
        feature_id = 0,
        p = 16,
        big_windowSize = 100,
        small_windowSize = 20,
        seed : int = None,
        recent_samples_size: int = 500
    ):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         feature_id=feature_id)
        self.big_windowData = []
        # big data not needed
        #self.big_Data = []
        self.big_windowTime = []
        self.big_minVal = None
        self.big_maxVal = None
        self.h = None

        self.big_windowSize = big_windowSize
        self.small_windowSize = small_windowSize

        self.vectors = []
        self.distances = []
        self.p = p
        self.labels_wKDE = np.zeros(self.big_windowSize - 1)
        self.flag = 0
        self.label = 0

        self.k = int(math.sqrt(self.small_windowSize))
        self.feature_id = feature_id
        self.feature_key = ""

    def update(self, data) -> bool:
        # init
        if self.feature_key == "":
            self.retrieve_key_from_idx(list(data.keys()))

        if self.big_maxVal is None or self.big_minVal is None or not len(
                self.big_windowData):
            for data_ in self.recent_samples:
                feature_value = data_[0][self.feature_key]
                if self.big_maxVal is None or feature_value > self.big_maxVal:
                    self.big_maxVal = feature_value
                if self.big_minVal is None or feature_value < self.big_minVal:
                    self.big_minVal = feature_value
                self.big_windowData.append(feature_value)

        #####

        if data[self.feature_key] > self.big_maxVal:
            self.big_maxVal = data[self.feature_key]
        elif data[self.feature_key] < self.big_minVal:
            self.big_minVal = data[self.feature_key]

        finalScore = self._detect()
        self.reset()

        return bool(finalScore)
    def _getH(self):
        # main window:
        std = np.std(self.big_windowData)
        if std == 0.0:
            std = 0.000001
        self.h = (4 / (3 * self.big_windowSize)) ** (1 / 5) * std

    def _getVectors(self):
        m = int((self.big_windowSize - self.small_windowSize) / self.small_windowSize + 1)
        for i in range(1, m + 1):
            sub_window = list(
                self.big_windowData[
                self.big_windowSize - (i - 1) * self.small_windowSize -
                self.small_windowSize:self.big_windowSize - (
                        i - 1) * self.small_windowSize])
            v = self._calVector(sub_window)
            self.vectors.append(v)

    def _calVector(self, set):
        # get the target set T:
        targets = []
        interval = (self.big_maxVal - self.big_minVal) / self.p
        for i in range(0, self.p):
            targets.append(self.big_minVal + (2 * i + 1) / 2 * interval)
        targets = np.array(targets)
        targets = targets.reshape(-1, 1)

        # calculate the descriptor:
        set = np.array(set)
        set = set.reshape(-1, 1)
        kde = KernelDensity(bandwidth=self.h, kernel='gaussian').fit(set)
        v = kde.score_samples(targets)
        vector = np.exp(v)
        return vector

    def _getDistances(self):
        for i in range(0, len(self.vectors) - 1):
            d = calDistance(np.array(self.vectors[i]), np.array(self.vectors[i + 1]))
            self.distances.append(d)

    def _getDmax(self, vor_distances):
        max_d = max(vor_distances)
        deltas = []
        for i in range(0, len(vor_distances) - 1):
            t_delta = abs(vor_distances[i] - vor_distances[i + 1])
            deltas.append(t_delta)
        max_d = max_d + min(deltas)
        return max_d

    def _getLabel(self):
        finalScore = 0.0
        self.label = 0
        curr_d = self.distances[0]
        ex_d = self.distances[1]
        vor_distances = self.distances[1:]
        max_d = self._getDmax(vor_distances)
        avg_d = np.mean(vor_distances)
        # get label:
        if self.flag == 0:
            if curr_d > max_d and ex_d <= avg_d:
                finalScore = 1.0
                self.label = 1
        # get s:
        if curr_d > max_d and ex_d <= avg_d:
            self.flag = 1
        else:
            self.flag = 0
        self.labels_wKDE = np.append(self.labels_wKDE, self.label)
        return finalScore

    def _detect(self):
        self._getH()
        self._getVectors()
        self._getDistances()
        finalScore = self._getLabel()
        return finalScore

    def reset(self):
        self.vectors = []
        self.distances = []

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_stream(stream, n_training_samples, classifier_path)
