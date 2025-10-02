import itertools
from itertools import islice
from statistics import mean

from sklearn.metrics import accuracy_score

from optimization.classifiers import Classifiers
import sys
import numpy as np
import torch
import random

import pickle

from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier

import uuid

myId = uuid.uuid4()
print("classifierId: " + str(myId))

from datasets import (
    #Electricity,
    #InsectsAbruptBalanced,
    #InsectsGradualBalanced,
    #InsectsIncrementalAbruptBalanced,
    #InsectsIncrementalBalanced,
    #InsectsIncrementalReoccurringBalanced,
    #NOAAWeather,
    #OutdoorObjects,
    #PokerHand,
    #Powersupply,
    #RialtoBridgeTimelapse,
    #SineClusters,
    #RAIN,
    #Keystroke,
    #TMDBalanced5s,
    #WaveformDrift2,
    #Chess,
    #SensorStream,
    #Ozone,
    #Luxembourg,
    #ForestCovertype,
    #GasSensor
    SineClustersPre,
    WaveformPre
)

seed = int(1337)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dataset_string = sys.argv[1]
n_training_samples = int(sys.argv[2])
n_verification_samples = int(sys.argv[3])
classifier_string = sys.argv[4]
classifier_args = sys.argv[5:]

mystr = classifier_string + "("
for i in range(0, len(classifier_args), 2):
    mystr += classifier_args[i] + "=" + classifier_args[i + 1] + ", "
mystr += ")"

stream = eval(dataset_string + "()")
# Initialize the classifier

clf_core = eval(mystr)

clf = Classifiers("")
clf.clf = clf_core

# implementation of leave-two-out cross validation
data = []
predictions = []
true_labels = []
for x, y in islice(stream, n_training_samples + n_verification_samples):
    data.append((x, y))

random.shuffle(data)

# Create folds
folds = [[] for _ in range(10)]
for i, sample in enumerate(data):
    folds[i % 10].append(sample)

accuracies = []

k = 10
# Perform Leave-2-Out cross validation with 10 folds
for val_sets in itertools.combinations(range(k), 2):
    val_data = []
    training_data = []

    # Separate training and validation data
    for i in range(k):
        if i in val_sets:
            val_data.extend(folds[i])
        else:
            training_data.extend(folds[i])

    # Train the classifier
    for x, y in training_data:
        clf.fit(x, y)

    # Validate the classifier
    for x, y in val_data:
        predictions.append(clf.predict(x))
        true_labels.append(y)

    # Calculate accuracy
    accuracies.append(accuracy_score(true_labels, predictions))

with open(classifier_string + str(myId) + "_" + dataset_string + ".pickle",
          "wb") as f:
    pickle.dump(clf, f)
print(f"RESULT: {1 - mean(accuracies)}")

