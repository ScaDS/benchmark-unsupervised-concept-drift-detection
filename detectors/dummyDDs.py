from .base import UnsupervisedDriftDetector, SemisupervisedDriftDetector
from metrics.computational_metrics import computational_metrics
from itertools import islice


class DummyDDBL1(UnsupervisedDriftDetector):

    def __init__(
            self
    ):
        super().__init__()
        self.name = 'DummyDDBL1'

    def update(self, data: dict) -> bool:
        """
        The Dummy returns always False.
        Therefore, no retraining is done.
        """
        return False


class DummyDDBL2(SemisupervisedDriftDetector):

    def __init__(
            self,
            recent_samples_size: int = 500,
            retraining_after_n: int = 500,
            # Couldn't find a configuration where the reset before re-training
            # was helpful
            reset_on_update: bool = False,
            seed=None
    ):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)
        self.seen_samples = 0
        self.recent_samples = []
        self.recent_samples_size = recent_samples_size
        self.retraining_after_n = retraining_after_n
        self.reset_on_update = reset_on_update
        self.name = ('DummyDDBL2_' + str(self.recent_samples_size) + '_' +
                     str(self.retraining_after_n) + '_' + str(
                    self.reset_on_update))

    def update(self, data: dict) -> bool:
        """
        The Dummy returns True after n samples.
        Therefore, retraining is done periodically.
        """
        if self.seen_samples >= self.retraining_after_n - 1:
            self.seen_samples = 0
            return True
        self.seen_samples += 1
        return False

    @computational_metrics
    def process_main_stream(self, stream, n_training_samples: int, classifier):
        # Processing the rest of the stream for detecting drifts
        for i, (x, y) in enumerate(islice(stream, n_training_samples, None),
                                   start=n_training_samples):

            self.recent_samples.pop(0)
            self.recent_samples.append((x, y))

            self.predictions.append(classifier.predict(x))
            self.labels.append(y)

            if self.update(x):
                self.handle_update(i, n_training_samples)
                for new_x, new_y in self.recent_samples:
                    if self.reset_on_update:
                        classifier.reset()
                    classifier.fit(new_x, new_y)
        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        # Initial training data
        _, classifier = self.load_main_clf(stream, n_training_samples,
                                           classifier_path)
        return self.process_main_stream(stream, n_training_samples, classifier)
