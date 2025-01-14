import copy
import numpy as np
from collections import deque
import random
from class_task import Task


class PrecedenceGraph:
    def __init__(self):
        self._edges = dict()  # t_id -> [successor_ids]
        self._reverse_edges = dict()  # t_id -> [predecessor_ids]

    def clear(self):
        self._edges = dict()
        self._reverse_edges = dict()

    def check_edge(self, fr_id, to_id) -> bool:
        if (fr_id not in self._edges.keys()) or (to_id not in self._edges[fr_id]):
            return False
        else:
            return True

    def get_all_ids(self) -> set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        all_vertices = left_vertices | right_vertices
        return all_vertices

    def get_start_ids(self) -> set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        start_vertices = left_vertices - right_vertices
        return start_vertices

    def get_end_ids(self) -> set[int]:
        left_vertices = set(self._edges.keys())
        right_vertices = set(self._reverse_edges.keys())
        end_vertices = right_vertices - left_vertices
        return end_vertices

    def get_copy_of_all_edges(self) -> dict[int, list[int]]:
        return copy.deepcopy(self._edges)

    def add_edge(self, fr_id: int, to_id: int) -> None:
        # Check that this edge have not already added:
        if not self.check_edge(fr_id, to_id):
            # Add direct edge
            if fr_id not in self._edges.keys():
                self._edges[fr_id] = []
            self._edges[fr_id].append(to_id)
            # Add reverse edge
            if to_id not in self._reverse_edges.keys():
                self._reverse_edges[to_id] = []
            self._reverse_edges[to_id].append(fr_id)

    def remove_edge(self, fr_id: int, to_id: int) -> None:
        if self.check_edge(fr_id, to_id):
            self._edges[fr_id].remove(to_id)  # Remove direct edge
            if not self._edges[fr_id]:
                del self._edges[fr_id]
            self._reverse_edges[to_id].remove(fr_id)  # Remove reverse edge
            if not self._reverse_edges[to_id]:
                del self._reverse_edges[to_id]

    # TODO: Think how to make all graphs not isomorphic to each other
    def random_network(self, number_of_vertices, start_num_diap=(3, 5), end_num_diap=(3, 5), seed=1):
        random.seed = seed
        start_num = np.round(np.random.uniform(start_num_diap[0], start_num_diap[1])).astype(int)
        end_num = np.min([number_of_vertices - 2 - start_num,
                         np.round(np.random.uniform(end_num_diap[0], end_num_diap[1])).astype(int)])

        # Step 1: Connect two dummy tasks with start and end tasks
        dummy_st_id = 0
        self._edges[dummy_st_id] = []
        for i in range(start_num):
            to_id = i + 1
            self._edges[dummy_st_id].append(to_id)
            self._reverse_edges[to_id] = [dummy_st_id]

        dummy_end_id = number_of_vertices - 1
        self._reverse_edges[dummy_end_id] = []
        for i in range(end_num):
            fr_id = number_of_vertices - 2 - i
            self._edges[fr_id] = [dummy_end_id]
            self._reverse_edges[dummy_end_id].append(fr_id)
        # print("Step 1: edges: ", self._edges)
        # print("Step 1: rev:   ", self._reverse_edges)

        # Step 2: Find random predecessor:
        predecessors = list(range(1, start_num + 1))
        for to_id in list(range(start_num + 1, dummy_end_id)):
            fr_id = random.choice(predecessors)
            self.add_edge(fr_id, to_id)
            if to_id < dummy_end_id - end_num:
                predecessors.append(to_id)
        # print("Step 2: edges: ", self._edges)
        # print("Step 2: rev:   ", self._reverse_edges)

        # Step 3: Find random successor & delete redundant edges:
        no_out_ids = [i for i in range(dummy_end_id)
                      if i not in self._edges.keys()]
        for fr_id in no_out_ids:
            to_id_list = list(range(max([start_num + 1, fr_id + 1]), dummy_end_id))
            to_id = random.choice(to_id_list)
            self.add_edge(fr_id, to_id)

            # Find & delete redundant edges:
            all_predecessors = self.get_all_predecessors(fr_id)
            all_predecessors.append(fr_id)
            all_successors = self.get_all_successors(to_id)
            all_successors.append(to_id)
            for i in all_predecessors:
                for j in [] if i not in self._edges.keys() else self._edges[i]:
                    if j in all_successors and (i, j) != (fr_id, to_id):
                        self.remove_edge(i, j)
        # print("Step 3: edges: ", self._edges)
        # print("Step 3: rev:   ", self._reverse_edges)

    def bfs(self, start_id: int, reverse=False) -> list:
        edges = self._reverse_edges if reverse else self._edges
        order = []

        # Check is there such vertex:
        if start_id not in edges.keys():
            # There is no such start id
            return [start_id]

        successors = set()
        q = deque([start_id])
        while q:
            v = q.popleft()
            for next_v in [] if v not in edges.keys() else edges[v]:
                if next_v not in successors:
                    successors.add(next_v)
                    q.append(next_v)
            order.append(v)
        return order

    def get_successors(self, fr_id: int) -> list[int]:
        return self._edges[fr_id] if fr_id in self._edges.keys() else []

    def get_predecessors(self, fr_id: int) -> list[int]:
        return self._reverse_edges[fr_id] if fr_id in self._reverse_edges.keys() else []

    def get_all_successors(self, v_id: int) -> list[int]:
        return self.bfs(v_id, False)[1:]

    def get_all_predecessors(self, v_id) -> list[int]:
        return self.bfs(v_id, True)[1:]
