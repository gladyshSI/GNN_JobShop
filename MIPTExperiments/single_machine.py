import copy
from asyncio import tasks

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Task:
    def __init__(self, id: int, e: float, d: float):
        self._id = id
        self._e = e  # Expectation
        self._d = d  # Deviation

    def get_id(self) -> int:
        return self._id

    def get_e(self) -> float:
        return self._e

    def get_d(self) -> float:
        return self._d

    def update_e_d(self, e: float, d: float):
        self._e = e
        self._d = d
        return self

    def get_initial_dur(self) -> float:
        return self._e

    def get_random_dur(self) -> float:
        return min(2*self._e - 1., max(1., np.random.normal(self._e, self._d)))


class Machine:
    def __init__(self):
        self.tasks = []

    def append_task(self, task: Task):
        self.tasks.append(task)

    def schedule_tasks(self, tasks_to_schedule: list[Task]):
        self.tasks = tasks_to_schedule

    def swap_tasks(self, left_id: int, right_id: int):
        task_to_swap = copy.deepcopy(self.tasks[left_id])
        self.tasks[left_id] = copy.deepcopy(self.tasks[right_id])
        self.tasks[right_id] = copy.deepcopy(task_to_swap)

    def get_initial_st_times(self) -> list[float]:
        starting_times = []
        st = 0.
        for task in self.tasks:
            starting_times.append(st)
            st = st + task.get_initial_dur()
        return starting_times

    def get_random_durs(self) -> list[float]:
        random_durs = []
        for task in self.tasks:
            random_durs.append(task.get_random_dur())
        return random_durs

    def right_shift(self, new_durs: list[float]) -> list[float]:  # returns new starting times
        init_st_times = self.get_initial_st_times()
        new_st_times = []
        et = 0.
        for i in range(len(self.tasks)):
            new_st = max(et, init_st_times[i])
            new_st_times.append(new_st)
            et = new_st + new_durs[i]
        return new_st_times

    def get_overlaps(self, new_st_times: list[float]) -> list[float]:
        overlaps = []
        init_st_times = self.get_initial_st_times()
        if len(init_st_times) != len(new_st_times):
            raise ValueError(
                "len(init_st_times) != len(new_st_times)")
        for i in range(len(init_st_times)):
            overlaps.append(new_st_times[i] - init_st_times[i])
        return overlaps

    def get_avg_overlaps(self, n: int) -> list[float]:
        sum_overlaps = []
        for i in tqdm(range(n)):
            random_durs = self.get_random_durs()
            new_st_times = self.right_shift(random_durs)
            overlaps = self.get_overlaps(new_st_times)
            if len(sum_overlaps) == 0:
                sum_overlaps = overlaps
            else:
                sum_overlaps = [sum_overlaps[i] + overlaps[i] for i in range(len(sum_overlaps))]
        return [sum_overlaps[i] / n for i in range(len(sum_overlaps))]


def machine_bubble_sort(machine: Machine):
    n = 10**6
    machine_num = len(machine.tasks)
    avg_overlaps_by_iteration = []
    max_overlaps_by_iteration = []
    machine_overlaps = machine.get_avg_overlaps(n)
    avg_overlaps_0 = sum(machine_overlaps) / machine_num
    avg_overlaps_by_iteration.append(avg_overlaps_0)

    max_overlap_0 = np.max(machine_overlaps)
    max_overlaps_by_iteration.append(max_overlap_0)

    for j in range(1, machine_num + 1):
        for i in range(0, machine_num - j):
            if machine.tasks[i].get_d() > machine.tasks[i + 1].get_d():
                machine.swap_tasks(i, i + 1)
                machine_overlaps = machine.get_avg_overlaps(n)
                avg_overlaps_0 = sum(machine_overlaps) / machine_num
                avg_overlaps_by_iteration.append(avg_overlaps_0)

                max_overlap_0 = np.max(machine_overlaps)
                max_overlaps_by_iteration.append(max_overlap_0)
    return avg_overlaps_by_iteration, max_overlaps_by_iteration

if __name__ == '__main__':
    machine1 = Machine()
    tasks_num = 10
    tasks = [Task(id=i, e=np.random.uniform(3, 10), d=np.random.uniform(0, 4)) for i in range(tasks_num)]
    print([(t.get_id(), t.get_e(), t.get_d()) for t in tasks])

    machine1.schedule_tasks(tasks)
    avg_overlaps_by_iteration, max_overlaps_by_iteration = machine_bubble_sort(machine1)
    print([machine1.tasks[i].get_id() for i in range(len(machine1.tasks))])
    print(avg_overlaps_by_iteration)

    # PRINT PLOT
    fig, ax = plt.subplots()
    ax.set_xlabel('iterations num')
    ax.set_ylabel('avg. overlap')
    ys = avg_overlaps_by_iteration
    xs = [x for x in range(len(ys))]

    plt.plot(xs, ys)
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('iterations num')
    ax.set_ylabel('max. overlap')
    ys = max_overlaps_by_iteration
    xs = [x for x in range(len(ys))]

    plt.plot(xs, ys)
    plt.show()
    plt.close()




