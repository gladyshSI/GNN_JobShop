import random
from typing import Callable

from class_graph import PrecedenceGraph
from class_graph_algs import PGAlgorithms, print_networkx_graph
from class_problem import Problem
from class_schedule import Schedule, print_schedule, mean_of_distribution
from class_task import Task


def rand_sgs(schedule: Schedule, f=random.choice, seed=1) -> None:
    random.seed = seed
    problem = schedule.get_copy_of_the_problem()
    machines_num = schedule.get_machines_num()
    res_first_free = {i: 0 for i in range(machines_num)}  # resource_id -> first free time

    pre_candidates = dict()  # task_id -> # Number of not scheduled predecessors
    candidates = problem.get_start_ids()  # ready to schedule
    scheduled = dict()  # task_id -> end_time

    while len(candidates) > 0:
        # next_candidate = random.choice(list(candidates))
        next_candidate = f(list(candidates))
        duration = problem.get_duration(next_candidate)
        # find est
        predecessors = problem.get_predecessors(next_candidate)
        end_times = [0] + [scheduled[pred] for pred in predecessors]
        est = max(end_times)
        # find resource and time
        r, first_free = min([(r, res_first_free[r]) for r in range(machines_num)], key=lambda x: x[1])
        est = max(est, first_free)
        # schedule
        schedule.schedule_task(r, next_candidate, est)
        # print("SCHEDULED: res=", r, "task=", next_candidate, "st=", est)

        # update structures
        candidates.remove(next_candidate)
        scheduled[next_candidate] = est + duration
        res_first_free[r] = est + duration
        next_pre_candidates = problem.get_successors(next_candidate)
        for c in next_pre_candidates:
            if c not in pre_candidates.keys():
                pre_candidates[c] = len(problem.get_predecessors(c)) - 1
            else:
                pre_candidates[c] -= 1
            if pre_candidates[c] == 0:
                candidates.add(c)
                pre_candidates.pop(c)


if __name__ == '__main__':
    graph = PrecedenceGraph()
    graph.random_network(12)

    alg = PGAlgorithms(graph)
    print_networkx_graph(alg.make_networkx_graph())

    tasks = {
        i: Task(i, 2, {1: 1/3, 2: 1/3, 3: 1/3}) for i in range(1, 11)
    }
    tasks[0] = Task(0, 0, {0: 1.})
    tasks[11] = Task(11, 0, {0: 1.})

    problem = Problem(graph, tasks, 2)
    schedule = Schedule(problem)

    rand_sgs(schedule)
    print_schedule(schedule)

    exact_overlap_dist = schedule.calculate_exact_overlap_distributions()
    estimated_overlap_dist = schedule.estimate_overlap_distributions_by_monte_carlo(100000)
    print(f'Exact overlaps: {exact_overlap_dist}\n'
          f'Estimated overlaps: {estimated_overlap_dist}\n')
    errors = []
    for task_id, exact_dist in exact_overlap_dist.items():
        for overlap, exact_prob in exact_dist.items():
            estimated_prob = estimated_overlap_dist[task_id][overlap] if overlap in estimated_overlap_dist[task_id].keys() else 0.
            if abs(exact_prob - estimated_prob) > 0.01:
                errors.append((task_id, overlap, exact_prob, estimated_prob))
    print(f'Errors [task, overlap, exact, estimation]: {errors}')

    print_schedule(schedule, {i: mean_of_distribution(exact_overlap_dist[i]) for i in exact_overlap_dist.keys()})
    print_schedule(schedule, {i: mean_of_distribution(estimated_overlap_dist[i]) for i in estimated_overlap_dist.keys()})
