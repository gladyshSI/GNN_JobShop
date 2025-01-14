from unittest import TestCase
from class_graph_algs import *


class Test(TestCase):
    def setUp(self):
        from class_graph import PrecedenceGraph
        """
        Disjunctive Graphs:
        """
        self.graph_parallel_lines = PrecedenceGraph()
        self.graph_parallel_lines.add_edge(0, 1)
        self.graph_parallel_lines.add_edge(0, 2)
        self.graph_parallel_lines.add_edge(1, 3)
        self.graph_parallel_lines.add_edge(2, 4)
        self.graph_parallel_lines.add_edge(3, 5)
        self.graph_parallel_lines.add_edge(4, 5)

        self.graph_parallel_and_cross = copy.deepcopy(self.graph_parallel_lines)
        self.graph_parallel_and_cross.add_edge(1, 4)
        self.graph_parallel_and_cross.add_edge(2, 3)

        self.graph_rhombus = PrecedenceGraph()
        self.graph_rhombus.add_edge(1, 2)
        self.graph_rhombus.add_edge(1, 3)
        self.graph_rhombus.add_edge(2, 4)
        self.graph_rhombus.add_edge(3, 4)

        self.graph_rhombus_tailed = copy.deepcopy(self.graph_rhombus)
        self.graph_rhombus_tailed.add_edge(0, 1)
        self.graph_rhombus_tailed.add_edge(4, 5)

        """
        Redundant Graphs:
        """
        self.graph_triangle_tailed = PrecedenceGraph()
        self.graph_triangle_tailed.add_edge(0, 1)
        self.graph_triangle_tailed.add_edge(1, 2)
        self.graph_triangle_tailed.add_edge(1, 3)
        self.graph_triangle_tailed.add_edge(2, 3)
        self.graph_triangle_tailed.add_edge(3, 4)
        self.graph_triangle_tailed.add_edge(2, 5)

        self.graph_rhombus_crossed1 = copy.deepcopy(self.graph_rhombus)
        self.graph_rhombus_crossed1.add_edge(1, 4)

        self.graph_rhombus_crossed2 = copy.deepcopy(self.graph_rhombus)
        self.graph_rhombus_crossed2.add_edge(2, 3)

        self.graph_rhombus_tailed_24redundant = copy.deepcopy(self.graph_rhombus_tailed)
        self.graph_rhombus_tailed_24redundant.add_edge(2, 6)
        self.graph_rhombus_tailed_24redundant.add_edge(6, 7)
        self.graph_rhombus_tailed_24redundant.add_edge(7, 4)

        self.graph_rhombus_tailed_13redundant = copy.deepcopy(self.graph_rhombus_tailed)
        self.graph_rhombus_tailed_13redundant.add_edge(1, 6)
        self.graph_rhombus_tailed_13redundant.add_edge(6, 7)
        self.graph_rhombus_tailed_13redundant.add_edge(7, 3)

    def test_is_redundancy_check(self):
        self.assertFalse(is_redundancy_check(self.graph_parallel_lines.get_copy_of_all_edges(), [0]))
        self.assertFalse(is_redundancy_check(self.graph_parallel_and_cross.get_copy_of_all_edges(), [0]))
        self.assertFalse(is_redundancy_check(self.graph_rhombus.get_copy_of_all_edges(), [1]))
        self.assertFalse(is_redundancy_check(self.graph_rhombus_tailed.get_copy_of_all_edges(), [0]))

        self.assertTrue(is_redundancy_check(self.graph_triangle_tailed.get_copy_of_all_edges(), [0]))
        self.assertTrue(is_redundancy_check(self.graph_rhombus_crossed1.get_copy_of_all_edges(), [1]))
        self.assertTrue(is_redundancy_check(self.graph_rhombus_crossed2.get_copy_of_all_edges(), [1]))
        self.assertTrue(is_redundancy_check(self.graph_rhombus_tailed_24redundant.get_copy_of_all_edges(), [0]))
        self.assertTrue(is_redundancy_check(self.graph_rhombus_tailed_13redundant.get_copy_of_all_edges(), [0]))

    def test_is_graph_disjunctive(self):
        self.assertTrue(is_graph_disjunctive(self.graph_parallel_lines))
        self.assertTrue(is_graph_disjunctive(self.graph_parallel_and_cross))
        self.assertTrue(is_graph_disjunctive(self.graph_rhombus))
        self.assertTrue(is_graph_disjunctive(self.graph_rhombus_tailed))

        self.assertFalse(is_graph_disjunctive(self.graph_rhombus_crossed1))
        self.assertFalse(is_graph_disjunctive(self.graph_rhombus_crossed2))
        self.assertFalse(is_graph_disjunctive(self.graph_rhombus_tailed_24redundant))
        self.assertFalse(is_graph_disjunctive(self.graph_triangle_tailed))
        self.assertFalse(is_graph_disjunctive(self.graph_rhombus_tailed_13redundant))
