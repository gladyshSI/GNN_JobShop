import copy

import numpy as np


class Task:
    def __init__(self, task_id: int, init_duration: int, discrete_prob_dist: dict):
        self._id = task_id
        self._duration = init_duration
        self._distribution = discrete_prob_dist  # {duration: int -> probability: float}
        self._min_possible_dur = min(discrete_prob_dist.keys())
        self._max_possible_dur = max(discrete_prob_dist.keys())

    def get_id(self):
        return self._id

    def get_duration(self):
        return self._duration

    def get_distribution(self):
        return copy.deepcopy(self._distribution)

    def get_min_possible_dur(self):
        return self._min_possible_dur

    def get_max_possible_dur(self):
        return self._max_possible_dur

    def set_id(self, task_id):
        self._id = task_id

    def set_duration(self, duration: int):
        if duration < 0:
            raise ValueError(f"Duration of the task {self._id} must not negative, but it is {duration}")
        if duration not in self._distribution.keys():
            raise ValueError(f"Duration of the task {self._id} must be contained in a distribution {self._distribution}")
        self._duration = duration

    def set_distribution(self, discrete_prob_dist: dict):
        # Check that all keys are integers not negative
        if not all(isinstance(k, int) and k >= 0 for k in discrete_prob_dist.keys()):
            raise ValueError("All keys in the distribution must be not negative integers.")

        # Check that the sum of values is equal to 1 (with some tolerance for floating point precision)
        total_prob = sum(discrete_prob_dist.values())
        if not abs(total_prob - 1.0) < 1e-9:  # Allowing for small floating-point errors
            raise ValueError("The sum of the probabilities must be equal to 1.")

        # If all checks pass, set the distribution
        self._distribution = discrete_prob_dist


def discrete_uniform_dist(mean: int, tail: int) -> dict:
    if tail >= mean or tail < 0:
        raise ValueError(
            "The tail must be smaller than the expectation and the tail must be greater than or equal to 0.")
    dist = {i: 1/(2 * tail + 1) for i in range(mean - tail, mean + tail + 1, 1)}
    return dist


def discrete_normal_dist(mean: int, std: float, tail: int) -> dict:
    if tail >= mean or tail < 0:
        raise ValueError(
            "The tail must be smaller than the expectation and the tail must be greater than or equal to 0.")
    N = 10000
    norm_array = np.random.normal(mean, std, N)
    dist = {i: 0. for i in range(mean - tail, mean + tail + 1, 1)}
    for n in norm_array:
        i = np.round(n)
        i = min([mean + tail, i])
        i = max([mean - tail, i])
        dist[i] += 1
    for i in range(mean - tail, mean + tail + 1, 1):
        dist[i] = dist[i] / N
    return dist


def discrete_exponential_dist(mean: int, lb: int, tail: int) -> dict:
    if tail >= mean + lb or tail < 0:
        raise ValueError(
            "The tail must be smaller than the expectation and the tail must be greater than or equal to 0.")
    N = 10000
    norm_array = np.random.exponential(mean, N)
    ub = lb + mean + tail
    dist = {i: 0. for i in range(lb, ub + 1, 1)}
    for n in norm_array:
        i = np.round(n + lb)
        i = min([ub, i])
        i = max([lb, i])
        dist[i] += 1
    for i in range(lb, ub + 1, 1):
        dist[i] = dist[i] / N
    return dist
