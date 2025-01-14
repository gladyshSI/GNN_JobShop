import copy
import pandas as pd
from collections import deque
import plotly.express as px

from class_problem import Problem



class Schedule:
    def __init__(self, problem: Problem) -> None:
        self._problem = copy.deepcopy(problem)
        self._machines_num = self._problem.get_machines_num()
        self._schedule = dict()  # machine_id -> [(task_id, start_time)]  # ALWAYS SORTED BY TIME
        self._rev_sch = dict()  # task_id -> (machine_id, start_time)
        self._edges = dict()  # task_id -> successor_id -> time_lag
        self._reverse_edges = dict()  # task_id -> predecessor_id -> time_lag

        for fr_id, to_ids in self._problem.get_copy_of_all_edges().items():
            self._edges[fr_id] = dict()
            for to_id in to_ids:
                self._edges[fr_id][to_id] = 0
                if to_id not in self._reverse_edges.keys():
                    self._reverse_edges[to_id] = dict()
                self._reverse_edges[to_id][fr_id] = 0

    def get_task_ids(self) -> set[int]:
        return self._problem.get_all_ids()

    def find_neighbors_if_equal_start_time(self,
                                           sch_on_machine: list[tuple[int, int]],
                                           task_id: int,
                                           index_to_any_task_with_eq_st: int) -> tuple[int, int]:
        # Detect the section with equal starting times
        equal_start_time = sch_on_machine[index_to_any_task_with_eq_st][1]
        left_border_excluded = index_to_any_task_with_eq_st
        while left_border_excluded >= 0 and sch_on_machine[left_border_excluded][1] == equal_start_time:
            left_border_excluded -= 1
        right_border_excluded = index_to_any_task_with_eq_st
        while right_border_excluded < len(sch_on_machine) and sch_on_machine[right_border_excluded][1] == equal_start_time:
            right_border_excluded += 1

        all_successors = self._problem.get_all_successors(task_id)
        pointer = left_border_excluded
        while pointer + 1 < right_border_excluded and sch_on_machine[pointer + 1][0] not in all_successors:
            pointer += 1

        left_neighbor = sch_on_machine[pointer][0] if pointer >= 0 else None
        right_neighbor = sch_on_machine[pointer + 1][0] if pointer + 1 < len(sch_on_machine) else None

        return left_neighbor, right_neighbor

    def nearest_left_and_right_task_ids(self, machine_id: int, task_id: int, time: int) -> (int, int):
        sch_on_machine = self._schedule[machine_id] if machine_id in self._schedule.keys() else []

        # Edge Cases:
        if len(sch_on_machine) == 0:
            return None, None
        elif time < sch_on_machine[0][1]:
            return None, sch_on_machine[0][0]
        elif time > sch_on_machine[-1][1]:
            return sch_on_machine[-1][0], None
        elif time == sch_on_machine[0][1]:
            return self.find_neighbors_if_equal_start_time(sch_on_machine, task_id, 0)
        elif time == sch_on_machine[-1][1]:
            return self.find_neighbors_if_equal_start_time(sch_on_machine, task_id, len(sch_on_machine) - 1)

        # (Here we know, that len >= 2):
        left_iterator = 0
        right_iterator = len(sch_on_machine) - 1
        middle_iterator = (left_iterator + right_iterator) // 2
        while middle_iterator != left_iterator:
            if time < sch_on_machine[middle_iterator][1]:
                right_iterator = middle_iterator
            elif time > sch_on_machine[middle_iterator][1]:
                left_iterator = middle_iterator
            else:
                return self.find_neighbors_if_equal_start_time(sch_on_machine, task_id, middle_iterator)
            middle_iterator = (left_iterator + right_iterator) // 2
        left_neighbor_id = sch_on_machine[left_iterator][0]
        right_neighbor_id = sch_on_machine[right_iterator][0]
        return left_neighbor_id, right_neighbor_id

    def check_is_there_space(self, machine_id: int, task_id: int, time: int, duration: int) -> bool:
        left_neighbor_id, right_neighbor_id = self.nearest_left_and_right_task_ids(machine_id, task_id, time)
        if left_neighbor_id is not None:
            if time < self._rev_sch[left_neighbor_id][1] + self._problem.get_duration(left_neighbor_id):
                return False
        if right_neighbor_id is not None:
            if time + duration > self._rev_sch[right_neighbor_id][1]:
                return False
        return True

    def check_precedence_relationships(self, task_id: int, time: int) -> bool:
        all_predecessors = self._problem.get_all_predecessors(task_id)
        all_successors = self._problem.get_all_successors(task_id)
        for predecessor in all_predecessors:
            if predecessor in self._rev_sch.keys():
                predecessor_duration = self._problem.get_duration(predecessor)
                predecessor_end_time = self._rev_sch[predecessor][1] + predecessor_duration
                if predecessor_end_time > time:
                    return False
        for successor in all_successors:
            if successor in self._rev_sch.keys():
                successor_start_time = self._rev_sch[successor][1]
                our_duration = self._problem.get_duration(task_id)
                our_end_time = time + our_duration
                if our_end_time > successor_start_time:
                    return False
        return True

    # TODO: Make order: task_id, machine_id, start_time
    def schedule_task(self, machine_id, task_id, start_time):
        if machine_id >= self._machines_num:
            raise ValueError(f'Machine id {machine_id} out of range')
        if task_id not in self.get_task_ids():
            raise ValueError(f'Task id {task_id} out of range')
        if task_id in self._rev_sch.keys():
            raise ValueError(f'Task id {task_id} is already scheduled')
        if not self.check_is_there_space(machine_id, task_id, start_time, self._problem.get_duration(task_id)):
            raise ValueError(f'There is no space between tasks {self.nearest_left_and_right_task_ids(machine_id, task_id, start_time)}')
        if not self.check_precedence_relationships(task_id, start_time):
            raise ValueError(f'Precedence relationship violated for task {task_id}')

        # Find neighbors
        left_neighbor_id, right_neighbor_id = self.nearest_left_and_right_task_ids(machine_id, task_id, start_time)

        # fill schedule in right order
        self._rev_sch[task_id] = (machine_id, start_time)
        if machine_id not in self._schedule.keys():
            self._schedule[machine_id] = []
        index = 0
        if left_neighbor_id is not None:
            index = self._schedule[machine_id].index((left_neighbor_id, self._rev_sch[left_neighbor_id][1])) + 1
        elif right_neighbor_id is not None:
            index = self._schedule[machine_id].index((right_neighbor_id, self._rev_sch[right_neighbor_id][1]))
        self._schedule[machine_id].insert(index, (task_id, start_time))

        # fill edges and reverse edges
        if left_neighbor_id is not None:
            left_neighbor_end_time = self._rev_sch[left_neighbor_id][1] + self._problem.get_duration(left_neighbor_id)
            if left_neighbor_id not in self._edges.keys():
                self._edges[left_neighbor_id] = dict()
            time_lag = start_time - left_neighbor_end_time
            self._edges[left_neighbor_id][task_id] = time_lag
            self._reverse_edges[task_id][left_neighbor_id] = time_lag
            # Since we added task between them
            if right_neighbor_id is not None and right_neighbor_id in self._edges[left_neighbor_id].keys():
                self._edges[left_neighbor_id].pop(right_neighbor_id)
        if right_neighbor_id is not None:
            our_end_time = start_time + self._problem.get_duration(task_id)
            if task_id not in self._edges.keys():
                self._edges[task_id] = dict()
            time_lag = self._rev_sch[right_neighbor_id][1] - our_end_time
            self._edges[task_id][right_neighbor_id] = time_lag
            self._reverse_edges[right_neighbor_id][task_id] = time_lag
            # Since we added task between them
            if left_neighbor_id is not None and left_neighbor_id in self._reverse_edges[right_neighbor_id].keys():
                self._reverse_edges[right_neighbor_id].pop(left_neighbor_id)

        # change time_lags to successors and from predecessors
        for p in self._problem.get_predecessors(task_id):
            if p not in self._rev_sch.keys():
                continue
            pred_st = self._rev_sch[p][1]
            pred_dur = self._problem.get_duration(p)
            pred_et = pred_st + pred_dur
            self._edges[p][task_id] = start_time - pred_et
            self._reverse_edges[task_id][p] = start_time - pred_et
        for s in self._problem.get_successors(task_id):
            if s not in self._rev_sch.keys():
                continue
            our_et = start_time + self._problem.get_duration(task_id)
            s_st = self._rev_sch[s][1]
            self._edges[task_id][s] = s_st - our_et
            self._reverse_edges[s][task_id] = s_st - our_et

    def get_machines_num(self):
        return self._machines_num

    def get_copy_of_the_problem(self) -> Problem:
        return copy.deepcopy(self._problem)

    def get_task_to_r_map(self):
        answ = dict()
        for r, r_sch in self._schedule.items():
            for t, _ in r_sch:
                answ[t] = r
        return answ

    def get_first_tasks(self) -> set[int]:
        left_tasks = set(self._edges.keys())
        right_tasks = set(self._reverse_edges.keys())
        start_tasks = left_tasks - right_tasks
        return start_tasks

    def get_last_tasks(self) -> set[int]:
        left_tasks = set(self._edges.keys())
        right_tasks = set(self._reverse_edges.keys())
        end_tasks = right_tasks - left_tasks
        return end_tasks

    def get_makespan(self):
        makespan = 0
        last_tasks = self.get_last_tasks()
        for t in last_tasks:
            st = self._rev_sch[t][1]
            et = st + self._problem.get_duration(t)
            makespan = max(makespan, et)
        return makespan

    def calculate_new_starting_times_after_right_shift(self, new_durations: dict) -> dict:
        new_sts = dict()  # task_id -> new_st
        first_tasks = self.get_first_tasks()

        q = deque(first_tasks)
        was_in_q = set(first_tasks)
        while q:
            t = q.pop()
            predecessors_dict = dict() if t not in self._reverse_edges.keys() else self._reverse_edges[t]
            pred_ets = [new_sts[p] + new_durations[p] for p, _ in predecessors_dict.items()]
            last_et = max([0] + [et for et in pred_ets])
            new_sts[t] = max(last_et, self._rev_sch[t][1])  # We Don't move to the left
            successors_dict = dict() if t not in self._edges.keys() else self._edges[t]
            for s, _ in successors_dict.items():
                check_list = [p for p, _ in self._reverse_edges[s].items()]
                if (s not in was_in_q) and (set(check_list) <= was_in_q):
                    q.appendleft(s)
                    was_in_q.add(s)
        return new_sts

    def calculate_exact_overlap_distributions(self, error_value=0.000001) -> dict[int, dict[int, float]]:
        end_times_distribution = dict()  # task_id -> (end_time -> probability)
        overlap_distributions = dict()   # task_id -> (overlap_value -> probability)
        first_tasks = self.get_first_tasks()

        q = deque(first_tasks)
        was_in_q = set(first_tasks)
        while q:
            t = q.pop()
            predecessors_dict = dict() if t not in self._reverse_edges.keys() else self._reverse_edges[t]

            pred_delta_distributions = [{0: 1.}]
            for pred, time_lag in predecessors_dict.items():
                scheduled_pred_end_time = self._rev_sch[pred][1] + self._problem.get_duration(pred)
                pred_et_distribution = end_times_distribution[pred].copy()
                delta_dist = crop_distribution(pred_et_distribution, scheduled_pred_end_time + time_lag)
                pred_delta_distributions.append(delta_dist)

            overlap_distributions[t] = normalize_distribution(
                cut_distribution(
                    max_distributions(pred_delta_distributions),
                    error_value
                )
            )
            scheduled_start_time = self._rev_sch[t][1]
            duration_distribution = self._problem.get_task_distribution(t)
            end_times_distribution[t] = sum_distributions([{scheduled_start_time: 1.},
                                                           duration_distribution,
                                                           overlap_distributions[t]])

            successors_dict = dict() if t not in self._edges.keys() else self._edges[t]
            for s in successors_dict.keys():
                check_list = [p for p, _ in self._reverse_edges[s].items()]
                if (s not in was_in_q) and (set(check_list) <= was_in_q):
                    q.appendleft(s)
                    was_in_q.add(s)
        return overlap_distributions

    def estimate_overlap_distributions_by_monte_carlo(self, num_points: int, error_value=0.00001) -> dict[int, dict[int, float]]:
        overlaps = dict()  # task_id -> [overlaps]
        for i in range(num_points):
            new_durations = self._problem.get_random_durations_from_distributions()
            new_starting_times = self.calculate_new_starting_times_after_right_shift(new_durations)
            for t_id, new_st in new_starting_times.items():
                scheduled_start_time = self._rev_sch[t_id][1]
                if t_id not in overlaps.keys():
                    overlaps[t_id] = []
                overlaps[t_id].append(max(0, new_st - scheduled_start_time))

        overlap_distributions = dict()
        for t_id, overlaps in overlaps.items():
            overlap_distributions[t_id] = cut_distribution({o: overlaps.count(o)/num_points for o in set(overlaps)},
                                                           error_value)
        return overlap_distributions

    def to_pandas(self):
        df = pd.DataFrame(columns=['Task', 'Start', 'Finish', 'Resource'])
        for r, task_sch in self._schedule.items():
            for t, st in task_sch:
                duration = self._problem.get_duration(t)
                df.loc[-1] = [t, st, st + duration, r]
                df.index = df.index + 1
                df = df.sort_index()
        return df


class SchAlgorithms:
    def __init__(self, schedule):
        self._sch = schedule

    def get_first_tasks(self):
        first_tasks = []
        for t in self._sch._edges.keys():
            if t not in self._sch._reverse_edges.keys():
                first_tasks.append(t)
        return first_tasks

    def calc_deltas(self, new_durations):
        deltas = dict()
        new_sts = self.calc_new_sts(new_durations)
        for t, new_st in new_sts.items():
            deltas[t] = new_st - self._sch._rev_sch[t][1]
        return deltas

    def ranking(self):
        ranks = dict()
        first_vs = self.get_first_tasks()
        q = deque()
        for v in first_vs:
            q.append((v, 0))
            ranks[v] = 0
        while q:
            v, rank = q.pop()
            if v in self._sch._pg._edges.keys():
                for next_v in self._sch._pg._edges[v]:
                    if (next_v not in ranks.keys()) or (rank + 1 > ranks[next_v]):
                        q.append((next_v, rank + 1))
                        ranks[next_v] = rank + 1
        return ranks


def print_schedule(schedule: Schedule, colors=None) -> None:
    if colors is None:
        colors = {i: 1 for i in range(len(schedule.get_task_ids()))}
    df = schedule.to_pandas()
    df['delta'] = df['Finish'] - df['Start']
    df['Color'] = df['Task'].map(colors)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", text="Task", color='Color',
                      color_continuous_scale=[(0, "green"), (0.5, "yellow"), (1, "red")])
    fig.update_yaxes(autorange="reversed")

    fig.layout.xaxis.type = 'linear'
    fig.data[0].x = df.delta.tolist()
    fig.show()


# TODO: MOVE to class_discrete_distribution
def crop_distribution(distribution: dict[int, float], key_to_zero: int) -> dict[int, float]:
    new_distribution = dict()
    probability_of_zero = 0.
    for k, v in distribution.items():
        if k <= key_to_zero:
            probability_of_zero += v
        else:
            new_distribution[k - key_to_zero] = v
    new_distribution[0] = probability_of_zero
    return new_distribution


def cut_distribution(distribution: dict[int, float], error_rate: float) -> dict[int, float]:
    for key in [k for k, v in distribution.items() if v < error_rate]:
        del distribution[key]
    return distribution


def normalize_distribution(distribution: dict[int, float]) -> dict[int, float]:
    sum_values = sum(distribution.values())
    for key, value in distribution.items():
        distribution[key] = value / sum_values
    return distribution


def sum_of_two_distributions(distribution_l: dict[int, float], distribution_r: dict[int, float]) -> dict[int, float]:
    sum_distribution = dict()
    for key_l, value_l in distribution_l.items():
        for key_r, value_r in distribution_r.items():
            new_key = key_l + key_r
            if new_key not in sum_distribution.keys():
                sum_distribution[new_key] = 0
            sum_distribution[new_key] += value_l * value_r
    return sum_distribution


def max_of_two_distributions(distribution_l: dict[int, float], distribution_r: dict[int, float]) -> dict[int, float]:
    max_distribution = dict()
    for key_l, value_l in distribution_l.items():
        for key_r, value_r in distribution_r.items():
            new_key = max(key_l, key_r)
            if new_key not in max_distribution.keys():
                max_distribution[new_key] = 0
            max_distribution[new_key] += value_l * value_r
    return max_distribution


def sum_distributions(distributions_list: list[dict[int, float]]) -> dict[int, float]:
    if len(distributions_list) == 0:
        return dict()
    elif len(distributions_list) == 1:
        return distributions_list[0]

    sum_distribution = sum_of_two_distributions(distributions_list[0], distributions_list[1])
    for d_id in range(2, len(distributions_list)):
        sum_distribution = sum_of_two_distributions(sum_distribution, distributions_list[d_id])
    return sum_distribution


def max_distributions(distributions_list: list[dict[int, float]]) -> dict[int, float]:
    if len(distributions_list) == 0:
        return dict()
    elif len(distributions_list) == 1:
        return distributions_list[0]

    max_distribution = max_of_two_distributions(distributions_list[0], distributions_list[1])
    for d_id in range(2, len(distributions_list)):
        max_distribution = max_of_two_distributions(max_distribution, distributions_list[d_id])
    return max_distribution


def mean_of_distribution(distribution: dict[int, float]) -> float:
    mean = 0.
    for k, v in distribution.items():
        mean += v * k
    return mean
