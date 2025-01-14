from unittest import TestCase

from class_graph import PrecedenceGraph
from class_problem import Problem
from class_schedule import Schedule, print_schedule, cut_distribution, normalize_distribution, \
    sum_distributions, max_distributions, crop_distribution, mean_of_distribution
from DiscreteOpt.heuristic_models import rand_sgs
from class_task import Task


class TestSchedule(TestCase):
    def setUp(self):
        self.graph = PrecedenceGraph()
        self.graph.add_edge(0, 1)
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(3, 5)
        self.graph.add_edge(0, 4)
        self.graph.add_edge(4, 5)

        self.tasks = {
            0: Task(0, 0, {0: 1.}),
            1: Task(1, 1, {1: 1.}),
            2: Task(2, 1, {1: 1.}),
            3: Task(3, 1, {1: 1.}),
            4: Task(4, 1, {1: 1.}),
            5: Task(5, 0, {0: 1.}),
        }

        self.problem = Problem(self.graph, self.tasks, 2)

    def test_initialization(self):
        schedule = Schedule(self.problem)
        self.assertEqual(
            2,
            schedule._machines_num
        )
        self.assertEqual(
            {},
            schedule._schedule
        )
        self.assertEqual(
            {},
            schedule._rev_sch
        )
        self.assertEqual(
            {0: {1: 0, 4: 0}, 1: {2: 0}, 2: {3: 0}, 3: {5: 0}, 4: {5: 0}},
            schedule._edges
        )
        self.assertEqual(
            {1: {0: 0}, 2: {1: 0}, 3: {2: 0}, 4: {0: 0}, 5: {3: 0, 4: 0}},
            schedule._reverse_edges
        )

    def test_schedule_task(self):
        schedule = Schedule(self.problem)
        schedule.schedule_task(0, 0, 0)
        schedule.schedule_task(0, 5, 3)
        schedule.schedule_task(0, 1, 0)
        schedule.schedule_task(0, 3, 2)
        schedule.schedule_task(1, 4, 0)
        self.assertEqual(
            2,
            schedule._machines_num
        )
        self.assertEqual(
            {0: [(0, 0), (1, 0), (3, 2), (5, 3)], 1: [(4, 0)]},
            schedule._schedule
        )
        self.assertEqual(
            {0: (0, 0), 1: (0, 0), 3: (0, 2), 4: (1, 0), 5: (0, 3)},
            schedule._rev_sch
        )
        self.assertEqual(
            {0: {1: 0, 4: 0}, 1: {2: 0, 3: 1}, 2: {3: 0}, 3: {5: 0}, 4: {5: 2}},
            schedule._edges
        )
        self.assertEqual(
            {1: {0: 0}, 2: {1: 0}, 3: {1: 1, 2: 0}, 4: {0: 0}, 5: {3: 0, 4: 2}},
            schedule._reverse_edges
        )

    def test_schedule_task_reverse_order(self):
        schedule = Schedule(self.problem)
        schedule.schedule_task(1, 4, 0)
        schedule.schedule_task(0, 3, 2)
        schedule.schedule_task(0, 1, 0)
        schedule.schedule_task(0, 5, 3)
        schedule.schedule_task(0, 0, 0)

        self.assertEqual(
            2,
            schedule._machines_num
        )
        self.assertEqual(
            {0: [(0, 0), (1, 0), (3, 2), (5, 3)], 1: [(4, 0)]},
            schedule._schedule
        )
        self.assertEqual(
            {0: (0, 0), 1: (0, 0), 3: (0, 2), 4: (1, 0), 5: (0, 3)},
            schedule._rev_sch
        )
        self.assertEqual(
            {0: {1: 0, 4: 0}, 1: {2: 0, 3: 1}, 2: {3: 0}, 3: {5: 0}, 4: {5: 2}},
            schedule._edges
        )
        self.assertEqual(
            {1: {0: 0}, 2: {1: 0}, 3: {1: 1, 2: 0}, 4: {0: 0}, 5: {3: 0, 4: 2}},
            schedule._reverse_edges
        )

    def test_schedule_task_violations(self):
        schedule = Schedule(self.problem)
        schedule.schedule_task(1, 4, 1)
        schedule.schedule_task(0, 0, 0)
        schedule.schedule_task(0, 1, 0)
        schedule.schedule_task(0, 5, 3)
        schedule.schedule_task(0, 3, 2)

        with self.assertRaises(ValueError):
            schedule.schedule_task(0, 0, 0)
        with self.assertRaises(ValueError):
            schedule.schedule_task(1, 2, 0)
        with self.assertRaises(ValueError):
            schedule.schedule_task(1, 2, 1)
        with self.assertRaises(ValueError):
            schedule.schedule_task(0, 2, 2)
        with self.assertRaises(ValueError):
            schedule.schedule_task(0, 2, 3)

        schedule.schedule_task(0, 2, 1)
        self.assertEqual(
            2,
            schedule._machines_num
        )
        self.assertEqual(
            {0: [(0, 0), (1, 0), (2, 1), (3, 2), (5, 3)], 1: [(4, 1)]},
            schedule._schedule
        )
        self.assertEqual(
            {0: (0, 0), 1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1), 5: (0, 3)},
            schedule._rev_sch
        )
        self.assertEqual(
            {0: {1: 0, 4: 1}, 1: {2: 0}, 2: {3: 0}, 3: {5: 0}, 4: {5: 1}},
            schedule._edges
        )
        self.assertEqual(
            {1: {0: 0}, 2: {1: 0}, 3: {2: 0}, 4: {0: 1}, 5: {3: 0, 4: 1}},
            schedule._reverse_edges
        )

    def test_function_rand_sgs_and_get_makespan(self):
        schedule = Schedule(self.problem)
        rand_sgs(schedule)
        print_schedule(schedule)

        self.assertEqual(
            3,
            schedule.get_makespan()
        )

    def test_calculate_new_starting_times_after_right_shift(self):
        schedule = Schedule(self.problem)
        schedule.schedule_task(0, 0, 0)
        schedule.schedule_task(0, 1, 0)
        schedule.schedule_task(0, 2, 1)
        schedule.schedule_task(0, 3, 2)
        schedule.schedule_task(1, 4, 0)
        schedule.schedule_task(0, 5, 3)

        self.assertEqual(
            {0: 0, 1: 0, 2: 1, 3: 3, 4: 0, 5: 5},
            schedule.calculate_new_starting_times_after_right_shift(new_durations={0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 0})
        )

    def test_calculate_exact_overlap_distributions_chain(self):
        graph = PrecedenceGraph()
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)

        tasks = {
            0: Task(0, 0, {0: 1.}),
            1: Task(1, 2, {1: 1/3, 2: 1/3, 3: 1/3}),
            2: Task(2, 2, {1: 1/3, 2: 1/3, 3: 1/3}),
            3: Task(3, 2, {1: 1/3, 2: 1/3, 3: 1/3}),
            4: Task(4, 0, {0: 1.})
        }

        problem = Problem(graph, tasks, 1)
        schedule_without_time_lags = Schedule(problem)
        schedule_without_time_lags.schedule_task(0, 0, 0)
        schedule_without_time_lags.schedule_task(0, 1, 0)
        schedule_without_time_lags.schedule_task(0, 2, 2)
        schedule_without_time_lags.schedule_task(0, 3, 4)
        schedule_without_time_lags.schedule_task(0, 4, 6)

        exact_overlap_distributions = schedule_without_time_lags.calculate_exact_overlap_distributions()
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[0]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[1]
        )
        self.assertEqual(
            {0: 2/3, 1: 1/3},
            exact_overlap_distributions[2]
        )
        self.assertEqual(
            {0: 5/9, 1: 3/9, 2: 1/9},
            exact_overlap_distributions[3]
        )
        self.assertEqual(
            {0: 13/27, 1: 9/27, 2: 4/27, 3: 1/27},
            exact_overlap_distributions[4]
        )

        schedule_with_time_lags = Schedule(problem)
        schedule_with_time_lags.schedule_task(0, 0, 0)
        schedule_with_time_lags.schedule_task(0, 1, 0)
        schedule_with_time_lags.schedule_task(0, 2, 3)
        schedule_with_time_lags.schedule_task(0, 3, 6)
        schedule_with_time_lags.schedule_task(0, 4, 9)

        exact_overlap_distributions = schedule_with_time_lags.calculate_exact_overlap_distributions()
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[0]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[1]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[2]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[3]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[4]
        )

    def test_calculate_exact_overlap_distributions_column(self):
        graph = PrecedenceGraph()
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(1, 4)
        graph.add_edge(2, 4)
        graph.add_edge(3, 4)

        tasks = {
            0: Task(0, 0, {0: 1.}),
            1: Task(1, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            2: Task(2, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            3: Task(3, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            4: Task(4, 0, {0: 1.})
        }

        problem = Problem(graph, tasks, 3)
        schedule_without_time_lags = Schedule(problem)
        schedule_without_time_lags.schedule_task(0, 0, 0)
        schedule_without_time_lags.schedule_task(0, 1, 0)
        schedule_without_time_lags.schedule_task(1, 2, 0)
        schedule_without_time_lags.schedule_task(2, 3, 0)
        schedule_without_time_lags.schedule_task(0, 4, 2)

        exact_overlap_distributions = schedule_without_time_lags.calculate_exact_overlap_distributions()
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[0]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[1]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[2]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[3]
        )
        self.assertEqual(
            {0: 8/27, 1: 19/27},
            exact_overlap_distributions[4]
        )

        schedule_with_one_time_lag = Schedule(problem)
        schedule_with_one_time_lag.schedule_task(0, 0, 0)
        schedule_with_one_time_lag.schedule_task(0, 1, 0)
        schedule_with_one_time_lag.schedule_task(1, 2, 1)
        schedule_with_one_time_lag.schedule_task(2, 3, 1)
        schedule_with_one_time_lag.schedule_task(0, 4, 3)

        exact_overlap_distributions = schedule_with_one_time_lag.calculate_exact_overlap_distributions()
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[0]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[1]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[2]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[3]
        )
        self.assertEqual(
            {0: 4 / 9, 1: 5 / 9},
            exact_overlap_distributions[4]
        )

    def test_calculate_exact_overlap_distributions_complex(self):
        graph = PrecedenceGraph()
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(1, 4)
        graph.add_edge(2, 4)
        graph.add_edge(3, 5)
        graph.add_edge(4, 5)

        tasks = {
            0: Task(0, 0, {0: 1.}),
            1: Task(1, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            2: Task(2, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            3: Task(3, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            4: Task(4, 2, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}),
            5: Task(5, 0, {0: 1.})
        }

        problem = Problem(graph, tasks, 2)
        optimal_schedule = Schedule(problem)
        optimal_schedule.schedule_task(0, 0, 0)
        optimal_schedule.schedule_task(0, 1, 0)
        optimal_schedule.schedule_task(1, 2, 0)
        optimal_schedule.schedule_task(0, 4, 2)
        optimal_schedule.schedule_task(1, 3, 2)
        optimal_schedule.schedule_task(0, 5, 4)

        exact_overlap_distributions = optimal_schedule.calculate_exact_overlap_distributions()
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[0]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[1]
        )
        self.assertEqual(
            {0: 1.},
            exact_overlap_distributions[2]
        )
        self.assertEqual(
            {0: 2/3, 1: 1/3},
            exact_overlap_distributions[3]
        )
        self.assertEqual(
            {0: 4/9, 1: 5/9},
            exact_overlap_distributions[4]
        )
        self.assertTrue(
            abs(exact_overlap_distributions[5][0] - 65/243) < 0.00001
        )
        self.assertTrue(
            abs(exact_overlap_distributions[5][1] - 111/243) < 0.00001
        )
        self.assertTrue(
            abs(exact_overlap_distributions[5][2] - 67/243) < 0.00001
        )


class TestDistributionFunctions(TestCase):
    def setUp(self):
        self.distribution_list = [{i: 1 / j for i in range(1, j + 1)} for j in range(1, 10)]

    def test_cut_distribution(self):
        distribution = {0: 1., 1: 1., 2: 0.01, 3: 1., 4: 1., 5: 0.01}
        cut_distribution(distribution, 0.01)
        self.assertEqual(
            {0: 1., 1: 1., 2: 0.01, 3: 1., 4: 1., 5: 0.01},
            distribution
        )
        cut_distribution(distribution, 0.02)
        self.assertEqual(
            {0: 1., 1: 1., 3: 1., 4: 1.},
            distribution
        )

    def test_normalize_distribution(self):
        n = 100
        distribution = {i: 1. / (n * 1.1) for i in range(n)}
        normalize_distribution(distribution)
        for key, value in distribution.items():
            self.assertTrue(abs(value - 1. / n) < 0.00001)

    def test_sum_distributions(self):
        dist_1 = {0: 1.}
        dist_2 = {0: 0.5, 1: 0.5}
        dist_3 = {1: 1.}

        sum_none = sum_distributions([])
        self.assertEqual(dict(), sum_none)
        sum_one = sum_distributions([dist_2])
        self.assertEqual(dist_2, sum_one)
        sum_two = sum_distributions([dist_1, dist_3])
        self.assertEqual({1: 1.}, sum_two)
        sum_three = sum_distributions([dist_1, dist_2, dist_3])
        self.assertEqual({1: 0.5, 2: 0.5}, sum_three)

    def test_max_distributions(self):
        dist_1 = {0: 1.}
        dist_2 = {0: 0.5, 1: 0.5}
        dist_3 = {1: 1.}

        max_none = max_distributions([])
        self.assertEqual(dict(), max_none)
        max_one = max_distributions([dist_2])
        self.assertEqual(dist_2, max_one)
        max_two = max_distributions([dist_1, dist_3])
        self.assertEqual({1: 1.}, max_two)
        max_three = max_distributions([dist_1, dist_2, dist_3])
        self.assertEqual({1: 1.}, max_three)

    def test_crop_distribution(self):
        dist = {0: 1., 1: 1., 3: 1., 4: 1., 5: 0.01}
        self.assertEqual(
            {0: 3., 1: 1., 2: 0.01},
            crop_distribution(dist, 3)
        )
        dist = {0: 1., 1: 1., 3: 1., 4: 1., 5: 0.01}
        self.assertEqual(
            {0: 2., 1: 1., 2: 1., 3: 0.01},
            crop_distribution(dist, 2)
        )

    def test_mean_of_distribution(self):
        dist = {0: 1/2, 1: 1/4, 2: 1/4}
        self.assertEqual(
            3/4,
            mean_of_distribution(dist)
        )
