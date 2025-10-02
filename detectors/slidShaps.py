import itertools

import pandas as pd
from scipy.stats import stats

from .base import UnsupervisedDriftDetector, BatchDetector

import numpy as np
import math
import multiprocessing as mp
from pyitlib import discrete_random_variable as drv


def subsets(X):
    """ use it for the full computation of shapley values
    computes all the possible subsets of features in the data """

    arr = list(X[0][0].keys())
    for n_index in range(len(arr)):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)


def subsets_bounded(X, bound):
    """ use it for the bounded computation of shapley values
    computes all the possible subsets of the features in the data with size up
    to the bound """

    arr = list(X[0][0].keys())
    for n_index in range(bound):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)


class ShapleyValueCalculator:
    def __init__(self, char_function, subset_generator_function,
                 subset_size=20, cpu_count=4):

        """

        char_function =                         'entropy' or 'total_correlation'
        subset_generator_function =             sets to consider when computing the Shapley Values
        subset_size =                           bound or the subsets dimension (only relevant when subset_generator_function = 'subsets_bounded')

        """

        self.char_function = char_function
        self.cpu_count = cpu_count
        self.subset_generator_function = subset_generator_function
        self.subset_size = subset_size

        self.value_dict = {}
        self.data_size = 0
        self.data = []

    def value_function_helper(self, subset):
        key = []
        """
        For some reason Python mp unpacks the tuples,so we have to
        differentiate here. Not sure how stable Python mp handles this.
        """
        if not isinstance(subset, tuple):
            subset = (subset,)
        for my_set in subset[0]:
            key.append(my_set)
        key = str(key)
        selected_data = []
        for ft in subset[0]:
            ft_list = []
            for d in self.data:
                ft_list.append(d[0][ft])
            selected_data.append(ft_list)
        value = eval(self.char_function)(selected_data)
        return (key, value)

    def calculate_value_functions(self, data):
        self.data = data

        if self.subset_generator_function == "subsets":
            subset_generator = eval(self.subset_generator_function)(data)
        else:
            subset_generator = eval(self.subset_generator_function)(
                data, self.subset_size)

        pool = mp.Pool(self.cpu_count - 1)
        #pool = mp.Pool(4)
        results = pool.starmap(self.value_function_helper,
                               [(subset,) for subset in subset_generator])

        '''
        results = []
        for subset in subset_generator:
            results.append(self.value_function_helper((subset,)))
        '''

        self.value_dict = dict(results)

    def calculate_sv(self, data, feature_number):
        shapley_sum = 0
        normalize_sum = 0

        sets = self.value_dict.keys()
        feature_string = str(feature_number)
        for my_set_str in sets:
            my_set = eval(my_set_str)
            if feature_string in my_set:
                if len(my_set) < 2:
                    normalize_sum += math.factorial(
                        len(list(data[0][0].keys())) - 1)
                    continue
                try:
                    value_S_k = self.value_dict[my_set_str]
                except:
                    continue
                try:
                    # get the correlation value of the features without the
                    # feature string feature
                    my_set.remove(feature_string)
                    value_S = self.value_dict[str(my_set)]
                    l = len(my_set)
                    permutations_covered = math.factorial(
                        (len(list(
                            data[0][0].keys())) - l - 1)) * math.factorial(l)

                    normalize_sum += permutations_covered

                    fac = permutations_covered
                    shapley_sum = shapley_sum + (value_S_k - value_S) * fac
                except:
                    continue

        shapley_sum = shapley_sum / normalize_sum
        return shapley_sum

    def calculate_SVs(self, data):
        shapley_results = []
        if not bool(self.value_dict) or self.data_size != len(
                list(data[0][0].keys())):
            self.calculate_value_functions(data)
            self.data_size = len(list(data[0][0].keys()))
        data_keys = list(data[0][0].keys())
        for i in data_keys:
            shapley_results.append(self.calculate_sv(data, i))
        shapley_results = [i for i in shapley_results]
        if len(shapley_results) == 0:
            return [0]
        return shapley_results


def total_correlation(X):
    # compute the total correlation C of a set of random variables X_1,...,
    # X_n such that C(X_1,...,X_n) = H(X_1) + ... + H(X_n) - H(X_1,...,
    # X_n)
    # X = [list(x[0].values()) for x in X]
    return drv.information_multi(X)


class SlidShaps(UnsupervisedDriftDetector, BatchDetector):

    def __init__(
            self,
            detection_buf_size=10,
            batch_size=100,
            overlap=0.1,
            alpha=0.01,
            gamma=1,
            statistical_test="t-test",  # ks-test
            approximation_type="full",  # bounded
            subset_bound=-1, # -1
            seed : int = None,
            recent_samples_size: int = 500
    ):

        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         batch_size=batch_size)
        self.detection_buf_size = detection_buf_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.alpha = alpha
        self.gamma = gamma
        self.statistical_test = statistical_test
        self.approximation_type = approximation_type
        self.subset_bound = subset_bound

    def _detect(self, shaps):
        length = shaps.shape[0]
        detected_drifts = []
        p_value_buf = []
        predictions = [0 for _ in range(length)]
        slid = 0
        while slid < length - 2 * self.detection_buf_size:
            tmp = []
            for i in range(shaps.shape[1]):
                hist = shaps[slid:slid + self.detection_buf_size, i]
                new = shaps[
                      slid + self.detection_buf_size:slid + 2 * self.detection_buf_size,
                      i]
                st, p_value = stats.ks_2samp(
                    hist, new) if (self.statistical_test == 'ks-test') else (
                    stats.ttest_ind(hist, new))
                tmp.append(p_value)
            slid += 1
            p_value_buf.append(tmp)
            range_size = int(self.gamma * self.detection_buf_size)
            alarm = self._check_drift(pd.DataFrame(p_value_buf), range_size,
                                      self.alpha)
            if alarm:
                predictions[slid + 2 * self.detection_buf_size - 1] = 1
                detected_drifts.append([slid - range_size, min(tmp)])

        predictions = self._reduce_ajcent_alarms(predictions)
        return detected_drifts, predictions, pd.DataFrame(p_value_buf)

    def _check_drift(self, p_value_df, k, alpha):
        """input the p values of all dimensions, check whether there are
        continuously k p_values less than alpha on any of the dimensions,
        if so return True to trigger alarm."""

        if p_value_df.shape[0] < k:
            return False

        for i in range(p_value_df.shape[1]):
            tmp = p_value_df.iloc[-k:, i]
            if tmp[tmp < alpha].size != k:
                continue
            else:
                return True
        return False

    def _reduce_ajcent_alarms(self, predictions, align='left'):
        """
        reduce a series of adjacent drift alarms into one, based on the
        'align' argument.
        """
        buffer = []
        tmp = []
        for i in predictions:
            if i == 0 and len(tmp) == 0:
                buffer.append(0)
            elif i == 1:
                tmp.append(1)
            else:
                sub = [0 for _ in range(len(tmp))]
                position = 0 if align == 'left' else len(
                    sub) // 2 if align == 'middle' else -1
                sub[position] = 1
                for x in sub:
                    buffer.append(x)
                buffer.append(0)
                tmp = []
        if len(tmp) != 0:
            sub = [0 for _ in range(len(tmp))]
            position = 0 if align == 'left' else len(
                sub) // 2 if align == 'middle' else -1
            sub[position] = 1
            for x in sub:
                buffer.append(x)
        len_buffer = len(buffer)
        len_pred = len(predictions)
        assert len_buffer == len_pred, (f'Different length after reduction: '
                                        f'{buffer} != {predictions}')
        return buffer

    def update(self, batch) -> bool:
        ol = int(self.overlap * self.batch_size)
        data_shaps = self.run_slidshaps(batch, self.batch_size, ol,
                                        self.approximation_type,
                                        self.subset_bound, _approx='max')

        detected_drifts, _, _ = self._detect(data_shaps)
        return bool(detected_drifts)

    def reset(self):
        pass

    def entropy(X):
        # compute the Shannon entropy H of a set of random variables X_1,
        # ...,X_n
        return drv.entropy_joint(X)

    ''' computes the Shapley Value using the characteristic function '''

    def compute_shapleyvalues(self, _mydata, _type, _subsets_bound=-1,
                              approx='max'):

        if _type == 'bounded':
            SVC = ShapleyValueCalculator("total_correlation",
                                         "subsets_bounded", _subsets_bound)
        elif _type == 'full':
            SVC = ShapleyValueCalculator("total_correlation", "subsets")
        shapleys = SVC.calculate_SVs(_mydata)

        return shapleys

    def run_slidshaps(self, _mydata, _window_width, _overlap, _type,
                      _subsets_bound,
                      _approx):
        T_in = 0
        T_fin = _window_width
        data_current = _mydata[T_in:T_fin]
        data_shaps = np.asarray(
            self.compute_shapleyvalues(data_current, _type, _subsets_bound,
                                       _approx)).reshape(-1, 1)
        for T_in in range(_overlap, len(_mydata), _overlap):
            T_fin = T_in + _window_width
            data_current = _mydata[T_in:T_fin]
            shapleyvalues = np.asarray(
                self.compute_shapleyvalues(data_current, _type,
                                           _subsets_bound,
                                           _approx)).reshape(-1, 1)
            data_shaps = np.concatenate([data_shaps, shapleyvalues], axis=1)

        return data_shaps

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_batch_stream(stream, n_training_samples, classifier_path)
