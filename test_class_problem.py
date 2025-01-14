from unittest import TestCase

from class_graph import PrecedenceGraph
from class_problem import Problem
from class_task import Task


class TestProblem(TestCase):
    def setUp(self):
        self.graph = PrecedenceGraph()
        self.graph.add_edge(0, 1)
        self.graph.add_edge(0, 2)
        self.graph.add_edge(1, 3)
        self.graph.add_edge(1, 4)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(2, 4)
        self.graph.add_edge(3, 5)
        self.graph.add_edge(4, 5)

        self.tasks = {
            0: Task(0, 0, {0: 1.}),
            1: Task(1, 1, {i: 1/4 for i in range(4)}),
            2: Task(2, 2, {i: 1/5 for i in range(5)}),
            3: Task(3, 3, {i: 1/6 for i in range(6)}),
            4: Task(4, 4, {i: 1/7 for i in range(7)}),
            5: Task(5, 0, {0: 1.}),
        }

        self.problem = Problem(self.graph, self.tasks)

    def test_left_longest_passes(self):
        left_lps = self.problem.left_longest_passes()
        self.assertEqual(
            {0: 0, 1: 0, 2: 0, 3: 2, 4: 2, 5: 6},
            left_lps
        )

    def test_right_longest_passes(self):
        right_lps = self.problem.right_longest_passes()
        self.assertEqual(
            {0: 6, 1: 5, 2: 6, 3: 3, 4: 4, 5: 0},
            right_lps
        )

    def test_get_longest_passes(self):
        longest_passes = self.problem.get_longest_passes()
        self.assertEqual(
            {0: (0, 6), 1: (0, 5), 2: (0, 6), 3: (2, 3), 4: (2, 4), 5: (6, 0)},
            longest_passes
        )

    def test_get_random_durations_from_distributions(self):
        for _ in range(10):
            task_to_chosen_durations = {task_id: [] for task_id in self.problem.get_all_ids()}  # task_id -> [chosen durations]
            N = 100000
            for i in range(N):
                new_durations = self.problem.get_random_durations_from_distributions()
                for task_id, duration in new_durations.items():
                    task_to_chosen_durations[task_id].append(duration)

            for task_id, durations in task_to_chosen_durations.items():
                for chosen_duration in durations:
                    self.assertTrue(chosen_duration in self.tasks[task_id].get_distribution().keys())

                for possible_duration, probability in self.tasks[task_id].get_distribution().items():
                    chosen_duration_num = task_to_chosen_durations[task_id].count(possible_duration)
                    self.assertTrue(abs(chosen_duration_num / N - probability) < 0.01)
