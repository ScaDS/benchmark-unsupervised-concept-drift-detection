from itertools import islice
from copy import deepcopy

import river.tree
import river.naive_bayes

from .base import SemisupervisedDriftDetector
from river.drift import PageHinkley
from metrics.computational_metrics import computational_metrics
from optimization.classifiers import Classifiers

class STUDD(SemisupervisedDriftDetector):
    """

    """

    def __init__(
            self,
            delta: float = 0.5,
            seed: int = None,
            recent_samples_size: int = 500
    ):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)
        self.detect_tool = PageHinkley(delta=delta)

    def update(self, error) -> bool:
        self.detect_tool.update(error)
        return self.detect_tool.drift_detected
        #return self.detect_tool.update(error)[0]

    @computational_metrics
    def process_main_stream(self, stream, n_training_samples, teacher_clf,
                            student_clf):
        # Processing the rest of the stream for detecting drifts
        for i, (x, y) in enumerate(islice(stream, n_training_samples, None),
                                   start=n_training_samples):

            self.recent_samples.pop(0)
            self.recent_samples.append((x, y))

            self.predictions.append(teacher_clf.predict(x))
            self.labels.append(y)

            # Get predictions from both classifiers
            teacher_predict = teacher_clf.predict(x)
            student_predict = student_clf.predict(x)
            error = int(teacher_predict != student_predict)

            # Update the drift detector and retrain if drift is detected
            if self.update(error):
                self.handle_update(i, n_training_samples)
                # Resetting the classifiers
                teacher_clf.reset()
                student_clf.reset()

                # Retrain classifiers with the last n_training_samples
                for new_x, new_y in self.recent_samples:
                    teacher_clf.fit(new_x, new_y)
                    new_y_hat = teacher_clf.predict(new_x)
                    student_clf.fit(new_x, new_y_hat)

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_stream(self, stream, n_training_samples: int, classifier_path):

        # Initial training data
        _, teacher_clf = self.load_main_clf(stream, n_training_samples,
                                            classifier_path)
        student_clf = None
        if isinstance(teacher_clf.clf, river.tree.HoeffdingTreeClassifier):
            student_clf = Classifiers("HT")
        elif isinstance(teacher_clf.clf, river.naive_bayes.GaussianNB):
            student_clf = Classifiers("NB")
        else:
            print("Could not determine teacher classifier type.")

        for x, y in islice(stream, n_training_samples):
            y_hat = teacher_clf.predict(x)
            student_clf.fit(x, y_hat)
        return self.process_main_stream(stream, n_training_samples,
                                        teacher_clf, student_clf)
