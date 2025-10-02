from .base import UnsupervisedDriftDetector, BatchDetector

import copy
import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import jensenshannon


class CDBD(UnsupervisedDriftDetector, BatchDetector):

    def __init__(self, divergence="KL", detect_batch=1, statistic="stdev",
                 significance=0.000001, subsets=50, feature_id=0,
                 recent_samples_size: int = 500,
                 batch_size=500, seed=None):

        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         batch_size=batch_size, feature_id=feature_id)
        self.current_distance = None
        self.beta = None
        self.feature_epsilons = None
        self._reference_density = None
        self.total_batches = 0
        self._drift_state = None
        self._input_col_dim = None
        self._prev_feature_distances = None
        self._bins = None
        self.total_epsilon = None
        self.epsilon = None
        self.reference = None
        self.reference_n = None
        self.batches_since_reset = 0
        self._prev_distance = None

        # Initialize parameters
        self.detect_batch = detect_batch
        self.statistic = statistic
        self.significance = significance
        self.subsets = subsets
        if divergence == "H":
            self.distance_function = self._hellinger_distance
        elif divergence == "KL":
            self.distance_function = self._KL_divergence
        else:
            self.distance_function = divergence

        self._lambda = 0  # batch number on which last drift was detected.

        # For visualizations
        self.distances = {}
        self.epsilon_values = {}
        self.thresholds = {}

        self.feature_id = feature_id
        self.single_variate = True
        self.batch_size = batch_size

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from batches_since_reset (lambda) onwards. Intended for use
        after ``drift_state == 'drift'``.
        """

        self.batches_since_reset = 0
        self._drift_state = None

        if self.detect_batch == 1:
            # The reference and test data will be (re-)concatenated by the
            # later call to update(), since drift cannot be detected on the
            # first batch, in this case.
            test_proxy = self.reference.iloc[
                         int(len(self.reference) / 2):,
                         ]
            self.reference = self.reference.iloc[
                             0: int(len(self.reference) / 2),
                             ]

        self.reference_n = self.reference.shape[0]
        self._bins = int(np.floor(np.sqrt(self.reference_n)))
        self.epsilon = []
        self.total_epsilon = 0

        if self.detect_batch == 1:
            self.update(test_proxy)

    def _build_histograms(self, dataset, min_values, max_values):
        """
        Computes histogram for each feature in dataset. Bins are equidistantly
        spaced from minimum value to maximum value to ensure exact alignment of
        bins between test and reference data sets.

        Args:
            dataset (DataFrame): DataFrame on which to estimate density using
                histograms.
            min_values (list): List of the minimum value for each feature.
            max_values (list): List of the maximum value for each feature.

        Returns:
            List of histograms for each feature. Histograms stored as list of
            frequency count of data in each bin.

        """

        histograms = [
            np.histogram(
                dataset.iloc[:, f],
                bins=self._bins,
                range=(min_values[f], max_values[f]),
            )[0]
            for f in range(self._input_col_dim)
        ]

        return histograms

    def _hellinger_distance(self, reference_density, test_density):
        """
        Computes Hellinger distance between reference and test histograms

        Args:
            reference_density (list): Univariate output of _build_histograms
                from reference batch.
            test_density (list): Univariate tput of _build_histograms from test
                batch.

        Returns:
            Hellinger distance between univariate reference and test density

        """

        f_distance = 0
        r_length = sum(reference_density)
        t_length = sum(test_density)
        for b in range(self._bins):
            f_distance += (
                                  np.sqrt(test_density[b] / t_length)
                                  - np.sqrt(reference_density[b] / r_length)
                          ) ** 2

        return np.sqrt(f_distance)

    def _adaptive_threshold(self, stat, test_n):
        """
        Computes adaptive threshold. If computing threshold for third test
        batch, removes our estimate of initial Epsilon from future estimates of
        epsilon_hat and std.

        Args:
            stat (string): Desired statistical method for computing threshold.
            test_n (integer): Number of samples in test batch.

        Returns:
            Adaptive threshold Beta.
        """

        if self.batches_since_reset == 3 and self.detect_batch != 3:
            self.total_epsilon -= self.epsilon[0]
            self.epsilon = self.epsilon[1:]

        # update scale for denominator (t - lambda - 1), 
        # accounting for our initial Epsilon estimate
        if self.batches_since_reset == 2 and self.detect_batch != 3:
            d_scale = 1
        else:
            d_scale = self.total_batches - self._lambda - 1

        # Increment running mean of epsilon from 
        # batches_since_reset (lambda) -> t-1
        self.total_epsilon += self.epsilon[
            -2]  # was -2 before total samples change...

        epsilon_hat = (1 / d_scale) * self.total_epsilon

        # Compute standard deviation for batches_since_reset (lambda) -> t-1
        total_stdev = sum(
            (self.epsilon[i] - epsilon_hat) ** 2 for i in
            range(len(self.epsilon) - 1)
        )
        stdev = np.sqrt(total_stdev / (d_scale))

        if stat == "tstat":
            t_stat = scipy.stats.t.ppf(
                1 - (self.significance / 2), self.reference_n + test_n - 2
            )
            beta = epsilon_hat + t_stat * (stdev / np.sqrt(d_scale))

        else:
            beta = epsilon_hat + self.significance * stdev

        return beta

    def _estimate_initial_epsilon(
            self, reference, num_subsets, histogram_mins, histogram_maxes
    ):
        """Computes a bootstrapped initial estimate of Epsilon on 2nd test
        batch, allowing HDM to detect drift on the 2nd batch.

        1. Subsets reference data with replacement
        2. Computes distance between each subset.
        3. Computes Epsilon: difference in distances.
        4. Averages Epsilon estimates.

        Args:
            reference (DataFrame): DataFrame consists of reference batch and
                first test batch.
            num_subsets (int): desired number of subsets to be sampled from
                reference data.
            histogram_mins (list): List of minimum values for each feature align
                histogram bins.
            histogram_maxes (list): List of maximum values for each feature
                align histogram bins.

        Returns:
            Bootstrapped estimate of intial Epsilon value.
        """

        # Resampling data
        bootstraps = []
        size = int((1 - (1 / num_subsets)) * self.reference_n)
        for i in range(num_subsets):
            subset = reference.sample(n=size, replace=True)
            bootstraps.append(
                self._build_histograms(subset, histogram_mins, histogram_maxes)
            )

        # Distance between each subset
        distances = []
        for df_indx in range(len(bootstraps)):
            j = df_indx + 1
            while j < len(bootstraps):

                subset1 = bootstraps[df_indx]
                subset2 = bootstraps[j]

                # Divergence metric
                total_distance = 0
                for f in range(self._input_col_dim):
                    f_distance = self.distance_function(subset1[f], subset2[f])
                    total_distance += f_distance
                distances.append(total_distance)

                j += 1

        # Epsilons between each distance
        epsilon = 0
        for delta_indx in range(len(distances)):
            j = delta_indx + 1
            while j < len(distances):
                epsilon += abs(distances[delta_indx] - distances[j]) * 1.0
                j += 1

        epsilon0 = epsilon / num_subsets

        return epsilon0

    def _KL_divergence(self, reference_density, test_density):
        """
        Computes Jensen Shannon (JS) divergence between reference and test
        histograms. JS is a bounded, symmetric form of KL divergence.

        Args:
            reference_density (list): Univariate output of _build_histograms
                from reference batch.
            test_density (list): Univariate output of _build_histograms from
                test batch.

        Returns:
            JS divergence between univariate reference and test density

        """

        return jensenshannon(reference_density, test_density)

    def set_reference(self, X):
        """
        Initialize detector with a reference batch. After drift, reference batch
        is automatically set to most recent test batch. Option for user to
        specify alternative reference batch using this method.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset - not used by CDBD
            y_pred (numpy.array): predicted labels for dataset - not used by CDBD
        """
        # Ensure only being used with 1 variable in reference
        if len(X.shape) > 1 and X.shape[1] != 1:
            raise ValueError("CDBD should only be used to monitor 1 variable.")

        """
        Initialize detector with a reference batch. After drift, reference batch
        is automatically set to most recent test batch. Option for user to
        specify alternative reference batch using this method.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset - not used by HDM
            y_pred (numpy.array): predicted labels for dataset - not used by HDM
        """
        # Initialize attributes
        self.reference = copy.deepcopy(X)
        self.reset()

    def update(self, X) -> bool:
        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
          X (DataFrame): next batch of data to detect drift on.
          y_true (numpy.ndarray): true labels of next batch - not used in CDBD
          y_pred (numpy.ndarray): predicted labels of next batch - not used in CDBD
        """
        if not isinstance(X, pd.DataFrame):
            # batch_stream method returns lists instead of dataframes
            # original implementation is based on pandas dataframes
            X = pd.DataFrame(
                [{self.feature_key: sample[0][self.feature_key]} for sample in
                 X])

        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
            X (DataFrame): next batch of data to detect drift on.
            y_true (numpy.ndarray): true labels of next batch - not used in HDM
            y_pred (numpy.ndarray): predicted labels of next batch - not used in HDM
        """

        if self._drift_state == "drift":
            self.reset()

        self.total_batches += 1
        self.batches_since_reset += 1
        test_n = X.shape[0]

        # Estimate reference and test histograms
        mins = []
        maxes = []
        reference_variable = self.reference.iloc[:, 0]
        test_variable = X.iloc[:, 0]
        mins.append(
            np.concatenate((reference_variable, test_variable)).min())
        maxes.append(
            np.concatenate((reference_variable, test_variable)).max())
        self._reference_density = self._build_histograms(self.reference, mins,
                                                         maxes)
        test_density = self._build_histograms(X, mins, maxes)

        # Divergence metric
        total_distance = 0
        feature_distances = []
        for f in range(self._input_col_dim):
            f_distance = self.distance_function(
                self._reference_density[f], test_density[f]
            )
            total_distance += f_distance
            feature_distances.append(f_distance)
        self.current_distance = (1 / self._input_col_dim) * total_distance
        self.distances[self.total_batches] = self.current_distance

        # For each feature, calculate Epsilon, difference in distances
        if self.total_batches > 1:
            self.feature_epsilons = [
                a_i - b_i
                for a_i, b_i in
                zip(feature_distances, self._prev_feature_distances)
            ]

        # Compute Epsilon and Beta
        if self.batches_since_reset >= 2:

            if self.batches_since_reset == 2 and self.detect_batch != 3:
                initial_epsilon = self._estimate_initial_epsilon(
                    self.reference, self.subsets, mins, maxes
                )
                self.epsilon.append(initial_epsilon)

            current_epsilon = abs(
                self.current_distance - self._prev_distance) * 1.0
            self.epsilon.append(current_epsilon)
            self.epsilon_values[self.total_batches] = current_epsilon

            condition1 = bool(
                self.batches_since_reset >= 2 and self.detect_batch != 3)
            condition2 = bool(
                self.batches_since_reset >= 3 and self.detect_batch == 3)
            if condition1 or condition2:

                self.beta = self._adaptive_threshold(self.statistic, test_n)
                self.thresholds[self.total_batches] = self.beta

                # Detect drift
                if current_epsilon > self.beta:
                    self._drift_state = "drift"
                    self.reference = X
                    self._lambda = self.total_batches

        if self._drift_state != "drift":
            self._prev_distance = self.current_distance
            self._prev_feature_distances = feature_distances
            self.reference = pd.concat([self.reference, X])
            self.reference_n = self.reference.shape[0]
            # number of bins for histogram, from reference batch
            self._bins = int(np.floor(np.sqrt(self.reference_n)))

        return self._drift_state == "drift"

    def run_stream(self, stream, n_training_samples: int, classifier_path):

        training_data, classifier = self.load_main_clf(stream,
                                                       n_training_samples,
                                                       classifier_path)
        # first sample, take the features [0] not the label [1], take the keys
        self.retrieve_key_from_idx(list(training_data[0][0].keys()))
        training_data_feature = (
            pd.DataFrame([{self.feature_key: features[self.feature_key]} for
                          features, label in training_data]))
        # training_data_feature = pd.DataFrame(training_data)[self.feature_key]
        # that's 1 per definition for CDBD
        self._input_col_dim = 1
        self.set_reference(training_data_feature)
        return super().process_main_batch_stream(stream, n_training_samples,
                                                 classifier)
