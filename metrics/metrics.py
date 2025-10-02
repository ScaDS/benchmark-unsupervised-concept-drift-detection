from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .drift import calculate_drift_metrics
from .lift_per_drift import lift_per_drift
from .requested_labels import get_portion_requested_labels


@dataclass
class ExperimentResult:
    """
    This data class stores the following metrics recorded during experiments:
    - accuracy of Hoeffding tree and naive Bayes classifiers with and without the use of a concept drift detector
    - f1 scores of Hoeffding tree and naive Bayes classifiers with and without the use of a concept drift detector
    - lpd of a Hoeffding tree and a naive Bayes classifier
    - portion of requested labels
    - mtfa
    - mtr
    - mtd
    - mdr
    """
    accuracy: float
    f1_score: float
    lpd: float
    portion_req_labels: float
    mtfa: float = None
    mtr: float = None
    mtd: float = None
    mdr: float = None

    def to_dict(self, include_drift_metrics: bool) -> dict:
        """
        Convert the stored data to a dictionary.

        :param include_drift_metrics: True if drift metrics
        (mtr, mtfa, mtd and mdr) shall be included in the dict
        :return: the dict
        """
        results = {
            "lpd": self.lpd,
            "acc": self.accuracy,
            "f1": self.f1_score,
            "portion_req_labels": self.portion_req_labels,
        }
        if include_drift_metrics:
            results["mtfa"] = self.mtfa
            results["mtr"] = self.mtr
            results["mtd"] = self.mtd
            results["mdr"] = self.mdr
        return results


def get_metrics(stream, predicted_drifts, true_labels, predicted_labels,
                n_req_labels, n_training_samples) -> ExperimentResult:
    """
    Calculate performance metrics based on the predicted drifts, the predicted
    labels and the true labels to calculate accuracies, f1 scores and
    lift-per-drift.
    If stream contains ground truth concept drift, mtr, mtfa, mtd and mdr are
    calculated as well.

    :param stream: the data stream the experiment was conducted on
    :param predicted_drifts: the positions of detected drifts
    :param true_labels: the true class labels
    :param predicted_labels: the predicted class labels
    :param n_req_labels: number of requested labels
    :param unsupervised: bool if detector is unsupervised
    :return: an ExperimentResult data class storing the corresponding metrics
    """
    if hasattr(stream, "drifts"):
        drift_metrics = calculate_drift_metrics(stream.drifts, predicted_drifts, stream.n_samples)
    else:
        drift_metrics = {"mtfa": None, "mdr": None, "mtr": None, "mtd": None}
    predicted_labels = np.array(predicted_labels).transpose()
    """
    for the lift per drift (lpd), load the accuracy of the baseline approach
    and calculate: (acc_clf - acc_base) / n_drifts
    """
    # lpd_clf =
    metrics = ExperimentResult(
        accuracy=accuracy_score(y_true=true_labels, y_pred=predicted_labels),
        f1_score=f1_score(y_true=true_labels, y_pred=predicted_labels,
                          average="macro"),
        lpd=-1,
        portion_req_labels=get_portion_requested_labels(stream.n_samples,
                                                        n_req_labels,
                                                        n_training_samples),
        **drift_metrics
    )
    return metrics
