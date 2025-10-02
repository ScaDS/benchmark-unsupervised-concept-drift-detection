import os
import sys

from metrics.metrics import get_metrics
from detectors import DummyDDBL1, DummyDDBL2

from datasets import (
    Electricity,
    #InsectsAbruptBalanced,
    #InsectsGradualBalanced,
    #InsectsIncrementalAbruptBalanced,
    #InsectsIncrementalBalanced,
    #InsectsIncrementalReoccurringBalanced,
    NOAAWeather,
    OutdoorObjects,
    PokerHand,
    #Powersupply,
    RialtoBridgeTimelapse,
    #SineClusters,
    #RAIN,
    #Keystroke,
    #TMDBalanced5s,
    #WaveformDrift2,
    #Chess,
    SensorStream,
    Ozone,
    #Luxembourg,
    ForestCovertype,
    GasSensor
)

n_training_samples = 2000

dataset_string = sys.argv[1]

classifiers = ["HoeffdingTreeClassifier"]
baselines = [
             "DummyDDBL1()",
             "DummyDDBL2(recent_samples_size=1, retraining_after_n=1)",
             "DummyDDBL2(recent_samples_size=10, retraining_after_n=10)",
             "DummyDDBL2(recent_samples_size=100, retraining_after_n=100)",
             "DummyDDBL2(recent_samples_size=1000, retraining_after_n=1000)",
             "DummyDDBL2(recent_samples_size=100, retraining_after_n=1000)",
             "DummyDDBL2(recent_samples_size=300, retraining_after_n=1000)",
             "DummyDDBL2(recent_samples_size=600, retraining_after_n=1000)",
             "DummyDDBL2(recent_samples_size=10, retraining_after_n=100)",
             "DummyDDBL2(recent_samples_size=30, retraining_after_n=100)",
             "DummyDDBL2(recent_samples_size=60, retraining_after_n=100)",
             "DummyDDBL2(recent_samples_size=1, retraining_after_n=10)",
             "DummyDDBL2(recent_samples_size=3, retraining_after_n=10)",
             "DummyDDBL2(recent_samples_size=6, retraining_after_n=10)",
             #"DummyDDBL2(recent_samples_size=600, retraining_after_n=1000, "
             #"reset_on_update=True)",
             #"DummyDDBL2(recent_samples_size=1000, retraining_after_n=1000, "
             #"reset_on_update=True)",
             #"DummyDDBL2(recent_samples_size=2000, retraining_after_n=2000, "
             #"reset_on_update=True)",
             #"DummyDDBL2(recent_samples_size=3000, retraining_after_n=5000, "
             #"reset_on_update=True)"
             ]

for clf_str in classifiers:
    for baseline_str in baselines:
        stream = eval(dataset_string+"()")
        clf_path = (
                    os.getcwd() + "/model/" + clf_str + "/" +
                    clf_str + "_" + dataset_string + ".pkl")
        baseline = eval(baseline_str)
        if isinstance(baseline, DummyDDBL2):
            unsupervised = False
        else:
            unsupervised = True
        (drifts, labels, predictions, n_req_labels, runtime, peak_memory,
         mean_memory) = baseline.run_stream(stream, n_training_samples,
                                            clf_path)

        metrics = get_metrics(stream, drifts, labels, predictions,
                              n_req_labels, n_training_samples)

        print(f"\nGENERAL INFO: Drift Detector: {str(baseline.name)}"
              f"\n\tDataset: {str(stream.filename)}"
              f"\n\tn_training_samples: {n_training_samples}"
              f"\n\tClassifier: {str(clf_str)}")

        print(f"Accuracy: {metrics.accuracy}")
        print(f"Runtime: {runtime}")
        print(f"Peak_memory: {peak_memory}")
        print(f"Mean_memory: {mean_memory}")
        print(
            f"Portion_req_label: {metrics.portion_req_labels}\n")
