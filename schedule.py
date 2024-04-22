import random
import pandas as pd
from collections import deque
import plotly.express as px

from precedence_graph import PGAlgorithms


class Schedule:
    def __init__(self, pg, t_to_res):
        self._pg = pg  # DO NOT CHANGE
        self._t_to_res = t_to_res  # task_id -> [resource_ids]
        self._schedule = dict()  # resource_id -> [(task_id, start_time)]
        self._rev_sch = dict()  # task_id -> (resource_id, start_time)
        self._edges = dict()  # task_id -> successor_id -> time_lag
        self._reverse_edges = dict()  # task_id -> predecessor_id -> time_lag

        for fr_id, to_ids in pg._edges.items():
            self._edges[fr_id] = dict()
            for to_id in to_ids:
                self._edges[fr_id][to_id] = 0
                if to_id not in self._reverse_edges.keys():
                    self._reverse_edges[to_id] = dict()
                self._reverse_edges[to_id][fr_id] = 0

    def get_task_list(self):
        return [t._id for t in self._pg._vertices]

    def schedule_task(self, resource_id, task_id, start_time):
        self._rev_sch[task_id] = (resource_id, start_time)
        if resource_id not in self._schedule.keys():
            self._schedule[resource_id] = []
        self._schedule[resource_id].append((task_id, start_time))

        last_task_r, last_st = (None, None) if len(self._schedule[resource_id]) < 2 else self._schedule[resource_id][-2]
        if last_task_r != None:
            last_et = last_st + self._pg._vertices[last_task_r]._duration
            if last_task_r not in self._edges.keys():
                self._edges[last_task_r] = dict()
            self._edges[last_task_r][task_id] = start_time - last_et
            if task_id not in self._reverse_edges.keys():
                self._reverse_edges[task_id] = dict()
            self._reverse_edges[task_id][last_task_r] = start_time - last_et

        predecessors = [] if task_id not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[task_id]
        for pred in predecessors:
            pred_st = self._rev_sch[pred][1]
            pred_dur = self._pg._vertices[pred]._duration
            pred_et = pred_st + pred_dur
            self._edges[pred][task_id] = start_time - pred_et
            self._reverse_edges[task_id][pred] = start_time - pred_et

    def get_task_to_r_map(self):
        answ = dict()
        for r, r_sch in self._schedule.items():
            for t, _ in r_sch:
                answ[t] = r
        return answ

    def rand_sgs(self, seed=1):
        random.seed = seed
        pg = self._pg
        t_to_res = self._t_to_res
        pga = PGAlgorithms(pg)
        res_first_free = dict()  # resource_id -> first free time
        for t, rs in t_to_res.items():
            for r in rs:
                res_first_free[r] = 0

        pre_candidates = dict()  # task_id -> #[not scheduled predecessors]
        candidates = set(pga.get_first_vertices())  # ready to schedule
        scheduled = dict()  # task_id -> end_time

        while len(candidates) > 0:
            next = random.choice(list(candidates))
            # next = min(list(candidates))
            duration = pg.get_duration(next)
            # find est
            predecessors = [] if next not in pg._reverse_edges.keys() else pg._reverse_edges[next]
            end_times = [0] + [scheduled[pred] for pred in predecessors]
            est = max(end_times)
            # find resource and time
            r, first_free = min([(r, res_first_free[r]) for r in t_to_res[next]], key=lambda x: x[1])
            est = max(est, first_free)
            # schedule
            self.schedule_task(r, next, est)
            # print("SCHEDULED: res=", r, "task=", next, "st=", est)

            # update structures
            candidates.remove(next)
            scheduled[next] = est + duration
            res_first_free[r] = est + duration
            next_pre_candidates = [] if next not in pg._edges.keys() else pg._edges[next]
            for c in next_pre_candidates:
                if c not in pre_candidates.keys():
                    pre_candidates[c] = 0 if next not in pg._reverse_edges.keys() else len(pg._reverse_edges[c]) - 1
                else:
                    pre_candidates[c] -= 1
                if pre_candidates[c] == 0:
                    candidates.add(c)
                    pre_candidates.pop(c)

    def get_first_tasks(self):
        first_tasks = []
        for t in self._edges.keys():
            if t not in self._reverse_edges.keys():
                first_tasks.append(t)
        return first_tasks

    def to_pandas(self):
        df = pd.DataFrame(columns=['Task', 'Start', 'Finish', 'Resource'])
        for r, task_sch in self._schedule.items():
            for t, st in task_sch:
                duration = self._pg.get_duration(t)
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

    def calc_new_sts(self, new_durations):
        new_sts = dict()  # task_id -> new_st
        first_tasks = self._sch.get_first_tasks()

        q = deque(first_tasks)
        was_in_q = set(first_tasks)
        while q:
            t = q.pop()
            predecessors_dict = dict() if t not in self._sch._reverse_edges.keys() else self._sch._reverse_edges[t]
            # print("t =", t)
            # print("q =", q)
            # print("preds =", [p for p, _ in predecessors_dict.items()])
            pred_ets = [new_sts[p] + new_durations[p] for p, _ in predecessors_dict.items()]
            last_et = max([0] + [et for et in pred_ets])

            # new_sts[t] = last_et
            new_sts[t] = max(last_et, self._sch._rev_sch[t][1])

            successors_dict = dict() if t not in self._sch._edges.keys() else self._sch._edges[t]
            # print("sucs =", [s for s, _ in successors_dict.items()])
            # print("-------")
            for s, _ in successors_dict.items():
                check_list = [p for p, _ in self._sch._reverse_edges[s].items()]
                if (s not in was_in_q) and (set(check_list) <= was_in_q):
                    q.appendleft(s)
                    was_in_q.add(s)
        return new_sts

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


def print_schedule(schedule, colors=None):
    if colors is None:
        colors = {i: 1 for i in range(len(schedule._pg._vertices))}
    df = schedule.to_pandas()
    df['delta'] = df['Finish'] - df['Start']
    df['Color'] = df['Task'].map(colors)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", text="Task", color='Color',
                      color_continuous_scale=[(0, "green"), (0.5, "yellow"), (1, "red")])
    fig.update_yaxes(autorange="reversed")

    fig.layout.xaxis.type = 'linear'
    fig.data[0].x = df.delta.tolist()
    fig.show()
