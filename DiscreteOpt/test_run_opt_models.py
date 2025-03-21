from unittest import TestCase

from class_schedule import crop_distribution, sum_distributions, normalize_distribution
from run_opt_models import get_buf_from_threshold
from run_opt_models import find_best_transition_time

class Test(TestCase):
    def test_get_buf_from_threshold(self):
        distr_unif_1 = {2: 0.14285714285714285, 3: 0.14285714285714285, 4: 0.14285714285714285, 5: 0.14285714285714285,
                        6: 0.14285714285714285, 7: 0.14285714285714285, 8: 0.14285714285714285}
        distr_unif_2 = {8: 0.3333333333333333, 9: 0.3333333333333333, 10: 0.3333333333333333}
        distr_unif_3 = {3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2}

        self.assertEqual(get_buf_from_threshold(4, distr_unif_1, 0.2), 3)
        self.assertEqual(get_buf_from_threshold(4, distr_unif_1, 0.4), 2)
        self.assertEqual(get_buf_from_threshold(4, distr_unif_1, 0.5), 1)

        self.assertEqual(get_buf_from_threshold(9, distr_unif_2, 0.2), 1)
        self.assertEqual(get_buf_from_threshold(9, distr_unif_2, 0.3), 1)
        self.assertEqual(get_buf_from_threshold(9, distr_unif_2, 0.4), 0)

        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.1), 2)
        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.2), 1)
        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.3), 1)
        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.4), 0)
        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.6), 0)
        self.assertEqual(get_buf_from_threshold(5, distr_unif_3, 0.8), 0)

    def test_find_best_transition_time(self):
        distr_left = {3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2, 7: 0.2}
        dur_left = 5
        distr_right = {0: 0.25, 1: 0.5, 2: 0.25}
        dur_right = 1

        # exceeding distr (transition=0) = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
        # exceeding distr (transition=1) = {0: 0.65, 1: 0.3, 2: 0.05}
        # exceeding distr (transition=2) = {0: 0.75, 1: 0.25}
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.55), 0)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.5), 0)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.4), 1)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.35), 1)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.3), 2)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.25), 2)
        self.assertEqual(find_best_transition_time(dur_left, distr_left, dur_right, distr_right, 0.01), 2)
