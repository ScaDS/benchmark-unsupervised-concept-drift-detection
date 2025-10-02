import river.base
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier


class Classifiers:
    """
    Classifiers provides an interface to operate two HoeffdingTreeClassifiers
    and two GaussianNBs for a concept drift detector.
    """

    def __init__(self, clf_str):
        """
        Init two HoeffdingTreeClassifiers and two GaussianNBs.
        """
        self.clf_str = clf_str
        self.clf = None
        self.init_clf()

        self.nonadaptive_trains = 0
        self.adaptive_trains = 0

    def init_clf(self):
        if self.clf_str == "HT":
            self.clf = HoeffdingTreeClassifier()
        elif self.clf_str == "GN":
            self.clf = GaussianNB()

    def predict(self, x):
        """
        Predict the label of the features x.

        :param x: the features
        :return: the label
        """

        return self.clf.predict_one(x)

    def predict_proba(self, x):
        """
        Predict the label of the features x.

        :param x: the features
        :return: the label
        """
        return self.clf.predict_proba_one(x)

    def fit(self, x, y):
        """
        Fit the classifiers on the training data consisting of x and y. If nonadaptive is True, the base classifiers are
        trained as well.

        :param x: the features
        :param y: the label
        :param nonadaptive: True if base classifiers shall be trained as well, else False
        """
        self.clf.learn_one(x, y)

    def reset(self):
        """
        Reset the classifiers assisted by concept drift detectors.
        """
        self.init_clf()

    def clone(self):
        clf_clone = Classifiers(self.clf_str)
        clf_clone.clf = self.clf.clone()
        return clf_clone
