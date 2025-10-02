import pickle
from itertools import islice
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch
import random
from metrics.computational_metrics import computational_metrics


class DriftDetector(ABC):
    def __init__(self, seed: Optional[int] = None,
                 recent_samples_size: int = 500,
                 feature_id: int = None):
        if seed is None:
            seed = int(1337)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.recent_samples = []
        self.recent_samples_size = recent_samples_size
        self.drifts = []
        self.labels = []
        self.predictions = []
        self.used_labels_set = set()
        self.single_variate = False
        self.feature_id = feature_id
        if not self.feature_id is None:
            self.single_variate = True
        self.feature_key = None

    @abstractmethod
    def update(
            self,
            data,
    ) -> bool:
        raise NotImplementedError(
            "This abstract base class does not implement update.")

    """
    Trains a classifier from scratch. Should be avoided for benchmark setup.
    Instead, its better to load a pre-trained one for all the pipelines.
    """

    def train_main_clf(self, stream, n_training_samples, classifier):
        for x, y in islice(stream, n_training_samples):
            classifier.fit(x, y)
            self.recent_samples.append((x, y))

        training_data = self.recent_samples
        # take only the last samples according to recent_samples_size
        self.recent_samples = self.recent_samples[
                              n_training_samples - self.recent_samples_size:]
        return training_data

    """
    Loads a classifier. n_training_samples should contain the n samples that
    were involved in the training process/Leave-2-out cross validation.
    """

    def load_main_clf(self, stream, n_training_samples, classifier_path):
        for x, y in islice(stream, n_training_samples):
            self.recent_samples.append((x, y))

        training_data = self.recent_samples
        # take only the last samples according to recent_samples_size
        self.recent_samples = self.recent_samples[
                              n_training_samples - self.recent_samples_size:]

        return training_data, pickle.load(open(classifier_path, "rb"))

    def retrieve_key_from_idx(self, keys):
        self.feature_key = keys[self.feature_id]

    @abstractmethod
    def run_stream(self, stream, n_training_samples: int, classifier_path):
        raise NotImplementedError(
            "This abstract base class does not implement update.")


class UnsupervisedDriftDetector(DriftDetector):
    """
    This abstract base class provides a consistent interface for all
    unsupervised concept drift detectors.
    """

    @abstractmethod
    def update(self, data: dict) -> bool:
        raise NotImplementedError(
            "This abstract base class does not implement update.")

    def __init__(self, seed: Optional[int] = None,
                 recent_samples_size: int = 500,
                 feature_id: int = None,
                 **kw):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         feature_id=feature_id)

    @computational_metrics
    def process_main_stream(self, stream, n_training_samples: int, classifier):
        # Processing the rest of the stream for detecting drifts
        for i, (x, y) in enumerate(islice(stream, n_training_samples, None),
                                   start=n_training_samples):

            if self.single_variate:
                if not self.feature_key:
                    self.retrieve_key_from_idx(list(x.keys()))
                x = {self.feature_key: x[self.feature_key]}

            self.recent_samples.pop(0)
            self.recent_samples.append((x, y))

            self.predictions.append(classifier.predict(x))
            self.labels.append(y)

            if self.update(x):  # Use the drift detector's update method
                self.drifts.append(i)
                for new_x, new_y in self.recent_samples:
                    classifier.fit(new_x, new_y)
                self.used_labels_set.update(range(
                    max(n_training_samples, i - self.recent_samples_size + 1),
                    i + 1))

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        _, classifier = self.load_main_clf(stream, n_training_samples,
                                           classifier_path)
        return self.process_main_stream(stream, n_training_samples, classifier)


class SemisupervisedDriftDetector(DriftDetector):
    """
    This abstract base class provides a consistent interface for all
    semisupervised concept drift detectors. run_stream() has to be defined
    per detector.
    """

    @abstractmethod
    def update(self, data: dict) -> bool:
        raise NotImplementedError(
            "This abstract base class does not implement update.")

    def __init__(self, seed: Optional[int] = None,
                 recent_samples_size: int = 500,
                 feature_id: int = None,
                 **kw):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         feature_id=feature_id)

    @abstractmethod
    def run_stream(self, stream, n_training_samples: int, classifier_path):
        raise NotImplementedError(
            "This abstract base class does not implement update.")

    def handle_update(self, i, n_training_samples):
        self.drifts.append(i)
        """
        For semi supervised, maintain a set of used labels since there might
        be some overlap
        """
        self.used_labels_set.update(range(
            max(n_training_samples, i - self.recent_samples_size + 1), i + 1))


class BatchDetector(DriftDetector):
    """
    Abstract base class for all batch data-based detectors.
    Minimally implements abstract methods common to all batch
    based detection algorithms.
    """

    @abstractmethod
    def update(self, data) -> bool:
        raise NotImplementedError(
            "This abstract base class does not implement update.")

    def __init__(self, batch_size: int = 500,
                 seed: Optional[int] = None,
                 recent_samples_size: int = 500,
                 feature_id: int = None,
                 **kw):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         feature_id=feature_id)
        self.batch_size = batch_size
        self.drifting_batches = []

    def batch_stream(self, stream, start):
        """
        Generator function to yield batches from a data stream.
        """
        batch = []
        for i, (x, y) in enumerate(islice(stream, start, None),
                                   start=start):
            if self.single_variate:
                if not self.feature_key:
                    self.retrieve_key_from_idx(list(x.keys()))
                x = {self.feature_key: x[self.feature_key]}
            batch.append((x, y))  # Add the current instance to the batch
            if len(batch) == self.batch_size:  # Check if batch size is reached
                yield batch  # Yield the current batch
                batch = []  # Reset the batch for the next iteration

        if batch:  # Yield any remaining instances as a final batch
            yield batch

    def handle_batch_update(self, batch_id, n_training_samples):

        self.drifting_batches.append(batch_id)
        self.drifts.append(batch_id * self.batch_size)
        self.used_labels_set.update(range(
            max(n_training_samples,
                batch_id * self.batch_size - self.batch_size + 1),
            batch_id * self.batch_size + 1))

    @computational_metrics
    def process_main_batch_stream(self, stream, n_training_samples: int,
                                  classifier):
        # Processing the rest of the stream for detecting drifts
        for batch_id, batch in enumerate(self.batch_stream(
                stream, n_training_samples)):

            for i, (x, y) in enumerate(batch):
                self.predictions.append(classifier.predict(x))
                self.labels.append(y)

            # pass only the features to the detector
            # batchx = [tuple(sample[:-1]) for sample in batch]
            # we are passing x and y to the detector here instead of x only
            if self.update(batch):  # Use the drift detector's update method
                self.handle_batch_update(batch_id, n_training_samples)
                for i, (x, y) in enumerate(batch):
                    classifier.fit(x, y)

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_batch_stream(self, stream, n_training_samples: int,
                         classifier_path):
        _, classifier = self.load_main_clf(stream, n_training_samples,
                                           classifier_path)
        return self.process_main_batch_stream(stream, n_training_samples,
                                              classifier)
