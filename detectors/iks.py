from collections import deque
from random import random

from .base import UnsupervisedDriftDetector


class Treap:
    def __init__(self, key, value=0):
        self.key = key
        self.value = value
        self.priority = random()
        self.size = 1
        self.lazy = 0
        self.max_value = value
        self.min_value = value
        self.left = None
        self.right = None

    @staticmethod
    def sum_all(node, value):
        if node is None:
            return
        node.value += value
        node.max_value += value
        node.min_value += value
        node.lazy += value

    @classmethod
    def unlazy(cls, node):
        if node is None:
            return
        cls.sum_all(node.left, node.lazy)
        cls.sum_all(node.right, node.lazy)
        node.lazy = 0

    @classmethod
    def update(cls, node):
        if node is None:
            return
        cls.unlazy(node)
        node.size = 1
        node.max_value = node.value
        node.min_value = node.value

        if node.left is not None:
            node.size += node.left.size
            node.max_value = max(node.max_value, node.left.max_value)
            node.min_value = min(node.min_value, node.left.min_value)

        if node.right is not None:
            node.size += node.right.size
            node.max_value = max(node.max_value, node.right.max_value)
            node.min_value = min(node.min_value, node.right.min_value)

    @classmethod
    def split_keep_right(cls, node, key):
        if node is None:
            return None, None

        cls.unlazy(node)

        if key <= node.key:
            left, node.left = cls.split_keep_right(node.left, key)
            right = node
        else:
            node.right, right = cls.split_keep_right(node.right, key)
            left = node

        cls.update(left)
        cls.update(right)

        return left, right

    @classmethod
    def merge(cls, left, right):
        if left is None:
            return right
        if right is None:
            return left

        if left.priority > right.priority:
            cls.unlazy(left)
            left.right = cls.merge(left.right, right)
            node = left
        else:
            cls.unlazy(right)
            right.left = cls.merge(left, right.left)
            node = right

        cls.update(node)
        return node

    @classmethod
    def split_smallest(cls, node):
        if node is None:
            return None, None

        cls.unlazy(node)

        if node.left is not None:
            left, node.left = cls.split_smallest(node.left)
            right = node
        else:
            right = node.right
            node.right = None
            left = node

        cls.update(left)
        cls.update(right)

        return left, right

    @classmethod
    def split_greatest(cls, node):
        if node is None:
            return None, None

        cls.unlazy(node)

        if node.right is not None:
            node.right, right = cls.split_greatest(node.right)
            left = node
        else:
            left = node.left
            node.left = None
            right = node

        cls.update(left)
        cls.update(right)

        return left, right


class IKS(UnsupervisedDriftDetector):

    def __init__(self, threshold=0.9, window_size=100, feature_id=0,
                 seed: int = None, recent_samples_size: int = 500):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size,
                         feature_id=feature_id)
        self.treap = None
        self.n = [0, 0]
        self.threshold = threshold
        self.window_size = window_size
        self.recent_data = deque(maxlen=self.window_size)
        self.counter = 0

        self.feature_id = feature_id
        self.single_variate = True
        # self.iks_statistics = 0

    def ks(self):
        assert self.n[0] == self.n[1]
        N = self.n[0]
        if N == 0:
            return 0
        return max(self.treap.max_value, -self.treap.min_value) / N

    def add(self, obs, group):
        group = 0 if group == 2 else group
        assert group == 0 or group == 1
        key = (obs, group)

        self.n[group] += 1
        left, right = Treap.split_keep_right(self.treap, key)

        left, left_g = Treap.split_greatest(left)
        val = 0 if left_g is None else left_g.value
        left = Treap.merge(left, left_g)

        right = Treap.merge(Treap(key, val), right)
        Treap.sum_all(right, 1 if group == 0 else -1)

        self.treap = Treap.merge(left, right)

    def remove(self, obs, group):
        group = 0 if group == 2 else group
        assert group == 0 or group == 1
        key = (obs, group)

        self.n[group] -= 1
        left, right = Treap.split_keep_right(self.treap, key)
        right_l, right = Treap.split_smallest(right)

        if right_l is not None and right_l.key == key:
            Treap.sum_all(right, -1 if group == 0 else 1)
        else:
            right = Treap.merge(right_l, right)

        self.treap = Treap.merge(left, right)

    def update(self, x) -> bool:
        # can't handle dict (the attribute name) here, have to extract the
        # feature value
        x = x[self.feature_key]
        if self.counter < self.window_size:
            self.counter += 1
            self.add((x, random()), 0)
            temp_value = (x, random())
            self.add(temp_value, 1)
            self.recent_data.append(temp_value)
        else:
            self.remove(self.recent_data.popleft(), 1)
            temp_value = (x, random())
            self.add(temp_value, 1)
            self.recent_data.append(temp_value)
            statistic = self.ks()

            if statistic > self.threshold:
                return True
        return False

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_stream(stream, n_training_samples, classifier_path)
