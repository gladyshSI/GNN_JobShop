from unittest import TestCase

from opt_experiments_analysis import calculate_best_metrics, calculate_place_distributions


class Test(TestCase):
    def test_calculate_best_metrics(self):
        all_metrics = [[{'m1': 1, 'm2': 2}, {'m1': 2, 'm2': 1}],
                       [{'m1': 2, 'm2': 2}, {'m1': 1, 'm2': 1}]]

        best_metrics = calculate_best_metrics(all_metrics)
        self.assertEqual(
            [{'m1': 1, 'm2': 2}, {'m1': 1, 'm2': 1}],
            best_metrics
        )

    def test_calculate_place_distributions(self):
        all_metrics = [[{'m1': 1.89, 'm2': 2.09}, {'m1': 2., 'm2': 1.09}],
                       [{'m1': 2., 'm2': 2.}, {'m1': 1.89, 'm2': 1}]]
        labels = ['Label1', 'Label2']
        step = 0.1
        place_distributions = calculate_place_distributions(all_metrics, labels, step)

        self.assertEqual(
            {
                'Label1': {
                    'm1': {1: 1, 2: 1},
                    'm2': {1: 2}
                },
                'Label2': {
                    'm1': {1: 1, 2: 1},
                    'm2': {1: 2}
                }
            },
            place_distributions
        )
