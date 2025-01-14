import copy

from writers_readers import *


class Problem:
    def __init__(self, precedence_graph: PrecedenceGraph, tasks_dict: dict[int, Task], machines_num=0):
        if set(tasks_dict.keys()) != precedence_graph.get_all_ids():
            raise ValueError(f'The tasks dictionary does not have the correct keys\n '
                             f'EXPECTED: {precedence_graph.get_all_ids()}\n '
                             f'GOT: {set(tasks_dict.keys())}')
        self._graph = copy.deepcopy(precedence_graph)  # PrecedenceGraph()
        self._tasks = tasks_dict  # task_id -> Task()
        self._ids = precedence_graph.get_all_ids()  # set([ids])
        self._machines_num = machines_num  # int

    def get_machines_num(self) -> int:
        return self._machines_num

    def get_random_durations_from_distributions(self) -> dict[int, int]:
        durations = dict()  # task_id -> new_duration
        for t_id, task in self._tasks.items():
            distribution = task.get_distribution()
            possible_durations = list(distribution.keys())
            probabilities = list(distribution.values())
            durations[t_id] = random.choices(possible_durations, probabilities, k=1)[0]

        return durations

    def change_task(self, task: Task):
        self._tasks[task.get_id()] = copy.deepcopy(task)

    def change_graph(self, new_graph: PrecedenceGraph):
        self._graph = copy.deepcopy(new_graph)

    def get_duration(self, task_id: int) -> int:
        if task_id not in self._tasks.keys():
            raise ValueError(f'Task {task_id} not found')
        return self._tasks[task_id].get_duration()

    def set_duration(self, task_id: int, new_duration: int) -> None:
        if task_id not in self._tasks.keys():
            raise ValueError(f'Task {task_id} not found')
        self._tasks[task_id].set_duration(new_duration)

    def get_task_min_possible_dur(self, task_id: int) -> int:
        if task_id not in self._tasks.keys():
            raise ValueError(f'Task {task_id} not found')
        return self._tasks[task_id].get_min_possible_dur()

    def get_task_max_possible_dur(self, task_id: int) -> int:
        if task_id not in self._tasks.keys():
            raise ValueError(f'Task {task_id} not found')
        return self._tasks[task_id].get_max_possible_dur()


    def get_task_distribution(self, task_id: int) -> dict[int, float]:
        if task_id not in self._tasks.keys():
            raise ValueError('Task not found')
        return self._tasks[task_id].get_distribution()

    def get_copy_of_all_edges(self) -> dict[int, list[int]]:
        return self._graph.get_copy_of_all_edges()

    def get_all_ids(self) -> set[int]:
        return self._ids

    def get_start_ids(self) -> set[int]:
        return self._graph.get_start_ids()

    def get_end_ids(self) -> set[int]:
        return self._graph.get_end_ids()

    def get_predecessors(self, task_id: int) -> list[int]:
        return self._graph.get_predecessors(task_id)

    def get_all_predecessors(self, task_id: int) -> list[int]:
        return self._graph.get_all_predecessors(task_id)

    def get_successors(self, task_id: int) -> list[int]:
        return self._graph.get_successors(task_id)

    def get_all_successors(self, task_id: int) -> list[int]:
        return self._graph.get_all_successors(task_id)

    def left_longest_passes(self) -> dict[int, int]:
        llps = dict()
        first_vs = self._graph.get_start_ids()
        q = deque()
        for v in first_vs:
            q.appendleft(v)
        while q:
            v = q.pop()
            max_left = 0 if v not in llps.keys() else llps[v]
            predecessors = self._graph.get_predecessors(v)
            for p_id in predecessors:
                llps_res = 0 if p_id not in llps.keys() else llps[p_id]
                max_left = max(max_left, llps_res + self.get_duration(p_id))
            llps[v] = max_left

            successors = self._graph.get_successors(v)
            for s_id in successors:
                q.appendleft(s_id)
        return llps

    def right_longest_passes(self) -> dict[int, int]:
        rlps = dict()
        last_vs = self._graph.get_end_ids()
        q = deque()
        for v in last_vs:
            q.appendleft(v)
        while q:
            v = q.pop()
            v_dur = self.get_duration(v)
            max_right = v_dur if v not in rlps.keys() else rlps[v]
            successors = self._graph.get_successors(v)
            for s_id in successors:
                rlps_res = 0 if s_id not in rlps.keys() else rlps[s_id]
                max_right = max(max_right, rlps_res + v_dur)
            rlps[v] = max_right

            predecessors = self._graph.get_predecessors(v)
            for p_id in predecessors:
                q.appendleft(p_id)
        return rlps

    def get_longest_passes(self):
        llps = self.left_longest_passes()
        rlps = self.right_longest_passes()
        lps = {v: (llps[v], rlps[v]) for v in list(self._tasks.keys())}
        return lps

    def get_copy_of_the_graph(self) -> PrecedenceGraph:
        return copy.deepcopy(self._graph)


if __name__ == '__main__':
    graph = read_graph("Data/PrecedenceGraphs/FasterGeneratedGraphs/50_notDummyVertices/graph_52_0.txt")
    tasks_list = read_tasks("Data/Tasks/uniform/tasks_uniform_52_0.txt")
    problem = Problem(graph, {task.get_id(): task for task in tasks_list})
