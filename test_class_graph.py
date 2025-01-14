from unittest import TestCase
from class_graph_algs import is_graph_disjunctive
from class_graph import PrecedenceGraph


def check_random_graph(graph: PrecedenceGraph):
    # Check that there is only one start and end vertex
    edges = graph.get_copy_of_all_edges()
    left_vertices = set(edges.keys())
    right_vertices = set([j for i in left_vertices for j in edges[i]])
    start_vertex = left_vertices - right_vertices
    end_vertex = right_vertices - left_vertices
    if len(start_vertex) != 1 or len(end_vertex) != 1:
        return False
    return is_graph_disjunctive(graph)


class TestPrecedenceGraph(TestCase):
    def setUp(self):
        self.graph = PrecedenceGraph()
        self.good_graph = PrecedenceGraph()
        self.good_graph.add_edge(0, 1)
        self.good_graph.add_edge(1, 2)
        self.good_graph.add_edge(1, 3)
        self.good_graph.add_edge(2, 4)
        self.good_graph.add_edge(3, 4)
        self.good_graph.add_edge(4, 5)

    def test_add_edge(self):
        self.graph.add_edge(0, 1)

        self.assertEqual([0], list(self.graph._edges.keys()))
        self.assertEqual([1], self.graph._edges[0])
        self.assertEqual([1], list(self.graph._reverse_edges.keys()))
        self.assertEqual([0], self.graph._reverse_edges[1])

    def test_check_edge(self):
        self.graph.add_edge(0, 1)

        self.assertTrue(self.graph.check_edge(0, 1))
        self.assertFalse(self.graph.check_edge(1, 0))
        self.assertFalse(self.graph.check_edge(0, 2))
        self.assertFalse(self.graph.check_edge(1, 1))

    def test_remove_edge(self):
        self.graph.add_edge(0, 1)
        self.graph.add_edge(0, 2)
        self.graph.add_edge(1, 2)

        self.graph.remove_edge(0, 1)
        self.assertEqual({0, 1}, set(self.graph._edges.keys()))
        self.assertEqual([2], self.graph._edges[0])
        self.assertEqual([2], self.graph._edges[1])
        self.assertEqual([2], list(self.graph._reverse_edges.keys()))
        self.assertEqual({0, 1}, set(self.graph._reverse_edges[2]))

        # Repeat: result should not change since we should do nothing
        self.graph.remove_edge(0, 1)
        self.assertEqual({0, 1}, set(self.graph._edges.keys()))
        self.assertEqual([2], self.graph._edges[0])
        self.assertEqual([2], self.graph._edges[1])
        self.assertEqual([2], list(self.graph._reverse_edges.keys()))
        self.assertEqual({0, 1}, set(self.graph._reverse_edges[2]))

    def test_bfs(self):
        # Direct bfs
        order = self.good_graph.bfs(0)
        self.assertEqual({0, 1, 2, 3, 4, 5}, set(order))
        order = self.good_graph.bfs(2)
        self.assertEqual({2, 4, 5}, set(order))

        # Reverse bfs
        reverse_order = self.good_graph.bfs(5, True)
        self.assertEqual({5, 4, 2, 3, 1, 0}, set(reverse_order))
        reverse_order = self.good_graph.bfs(3, True)
        self.assertEqual({3, 1, 0}, set(reverse_order))

    def test_get_all_successors(self):
        all_successors = self.good_graph.get_all_successors(0)
        self.assertEqual({1, 2, 3, 4, 5}, set(all_successors))
        all_successors = self.good_graph.get_all_successors(2)
        self.assertEqual({4, 5}, set(all_successors))

    def test_get_all_predecessors(self):
        all_predecessors = self.good_graph.get_all_predecessors(5)
        self.assertEqual({0, 1, 2, 3, 4}, set(all_predecessors))
        all_predecessors = self.good_graph.get_all_predecessors(3)
        self.assertEqual({0, 1}, set(all_predecessors))

    def test_random_network(self):
        for i in range(1000):
            self.graph.clear()
            self.graph.random_network(50, seed=i)
            self.assertTrue(check_random_graph(self.graph))
