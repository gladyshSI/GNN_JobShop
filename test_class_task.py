from unittest import TestCase


class TestTask(TestCase):
    def setUp(self):
        from class_task import Task
        self.task = Task(task_id=1, init_duration=10, discrete_prob_dist={1: 0.5, 2: 0.5})

    def test_initialization(self):
        """
        Test if the Task object initializes with the correct values.
        """
        self.assertEqual(self.task._id, 1)
        self.assertEqual(self.task._duration, 10)
        self.assertEqual(self.task._distribution, {1: 0.5, 2: 0.5})

    def test_get_id(self):
        self.assertEqual(self.task.get_id(), 1)

    def test_get_duration(self):
        self.assertEqual(self.task.get_duration(), 10)

    def test_get_distribution(self):
        self.assertEqual(self.task.get_distribution(), {1: 0.5, 2: 0.5})

    def test_set_duration_valid(self):
        """
        Test setting a valid duration value.
        """
        self.task.set_duration(1)
        self.assertEqual(self.task._duration, 1)
        self.task.set_duration(2)
        self.assertEqual(self.task._duration, 2)

    def test_set_duration_invalid(self):
        """
        Test that setting an invalid duration raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.task.set_duration(-5)  # Negative duration should raise ValueError
        with self.assertRaises(ValueError):
            self.task.set_duration(0)  # duration should be in the distribution

    def test_set_distribution_valid(self):
        """
        Test setting a valid distributions.
        -> Should be a dict.
        -> all keys should be non-negative integers
        -> sum of all values should be equal to 1.
        """
        self.task.set_distribution({0: 0.5, 1: 0.5})
        self.assertEqual(self.task._distribution, {0: 0.5, 1: 0.5})
        self.task.set_distribution({5: 0.2, 6: 0.5, 8: 0.3})
        self.assertEqual(self.task._distribution, {5: 0.2, 6: 0.5, 8: 0.3})

    def test_set_distribution_invalid(self):
        """
        Test that setting an invalid distribution raises a ValueError.
        -> if some keys are negative or float
        -> the sum of all values not equal to 1.
        """
        with self.assertRaises(ValueError):
            self.task.set_distribution({-1: 0.1, 0: 0.9})  # Negative key

        with self.assertRaises(ValueError):
            self.task.set_distribution({0: 0.1, 1: 0.8})  # sum of values < 1

        with self.assertRaises(ValueError):
            self.task.set_distribution({5: 0.2, 6: 0.1, 8: 0.8})  # sum of values > 1

    def test_get_min_or_max_possible_dur(self):
        self.assertEqual(1, self.task.get_min_possible_dur())
        self.assertEqual(2, self.task.get_max_possible_dur())
