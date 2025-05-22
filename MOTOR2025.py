import time

from docplex.cp.model import *
import copy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import sqrt
from graphlib import TopologicalSorter
from collections import deque

from writers_readers import read_graph


def calculate_overlaps(init_durations, task_durations, execution_order):
    size = len(task_durations[0])
    N = len(execution_order)
    init_st = np.zeros(N)
    for i in range(N-1):
        init_st[i+1] = init_st[i] + init_durations[execution_order[i]]
    overlaps = [0.]
    sts = np.zeros(shape=size)
    for i in tqdm(range(N - 1)):
        new_sts = np.maximum(init_st[i+1], np.add(sts, task_durations[execution_order[i]]))
        overlap = np.average(np.subtract(new_sts, init_st[i+1]))
        overlaps.append(overlap)
        sts = new_sts
    # print(overlaps)
    return overlaps


# Returns list of sequences after each step of bubble sort in increasing order of weight
def step_by_step_bubble_sort(seq: list[int], weights: np.array) -> list[list[int]]:
    seq_cpy = copy.deepcopy(seq)
    weights_cpy = copy.deepcopy(weights)
    sequences = [copy.deepcopy(seq)]
    for i in range(len(seq_cpy) - 1):
        for j in range(len(seq_cpy) - i - 1):
            if weights_cpy[j] > weights_cpy[j + 1]:
                a, b = seq_cpy[j+1], weights_cpy[j+1]
                seq_cpy[j+1], weights_cpy[j+1] = seq_cpy[j], weights_cpy[j]
                seq_cpy[j], weights_cpy[j] = a, b
                sequences.append(copy.deepcopy(seq_cpy))
    return sequences


def bubble_sort_heuristic(init_durs, weights, task_durs, init_execution_order):
    sequences = step_by_step_bubble_sort(init_execution_order, weights)
    objs = []
    for seq in sequences:
        avg_overlap = np.average(calculate_overlaps(init_durations=init_durs, task_durations=task_durs, execution_order=seq))
        objs.append(avg_overlap)
    return objs


def draw_bubble_sort_heur_results(objs_for_experiments: list[list[float]], type: str, title='Heuristic Bubble Sort'):
    experiments_num = len(objs_for_experiments)
    xss = [range(len(objs_for_experiments[i])) for i in range(experiments_num)]

    # Plot each list on the same graph
    for i in range(experiments_num):
        plt.plot(xss[i], objs_for_experiments[i], color='black', linewidth=1.)  # , label=f'Задача {i + 1}')

    # Add labels and legend
    # plt.xlabel('Num of swaps in bubble sort')
    # plt.xlabel('Число перестановок при сортировке пузырьком')
    plt.xlabel('t')
    # plt.ylabel('Average expected start time deviation')
    # plt.ylabel('Среднее ожидаемое отклонение стартов работ')
    plt.ylabel('Q')  # (r'$\frac{1}{n}\sum_{j=1}^n E(\delta_j)$')
    plt.title(title)
    # plt.legend()
    plt.savefig(f'./Output/plots/MOTORexp1_{type}.svg')
    plt.show()


def draw_boxplot_for_heuristic_comp(file, d_type, title):
    with open(file, 'r') as f:
        lines = f.readlines()
    heuristic_objs = []
    cplex_objs = []
    for line in lines:
        heuristic_objs.append(float(line.split(':')[2].split(',')[0]))
        cplex_objs.append(float(line.split(':')[3]))
    heuristic_d = [heuristic_objs[i] - min(heuristic_objs[i], cplex_objs[i]) for i in range(len(heuristic_objs))]
    cplex_d = [cplex_objs[i] - min(heuristic_objs[i], cplex_objs[i]) for i in range(len(cplex_objs))]
    plt.boxplot([heuristic_d, cplex_d], labels=['Algorithm 1', 'Algorithm 2'])
    # plt.ylabel('Отклонение целевой функции от лучшего результата')
    plt.ylabel(r'$\Delta_{\rm{obj}}$')
    plt.title(title)
    plt.savefig(f'./Output/plots/MOTORexp2_{d_type}.svg')
    plt.show()


def get_seq_from_cplex_sol(msol, x_is, tasks):
    starting_times = dict()  # task_id -> st
    for i in tasks:
        var_sol = msol.get_var_solution(x_is[(i, 0)])
        starting_times[i] = var_sol.get_start()

    return sorted(tasks, key=lambda x: starting_times[x])


def stoch_cplex_model(tasks_num: int,
                      last_task_id: int,
                      edge_dict: dict,
                      initial_durations: np.array,
                      task_durations: list[np.array],
                      time_limit: int, scenarios_num=60, scale=10**3):
    tasks = list(range(tasks_num))
    scenarios = [np.round(initial_durations * scale).astype(int)]
    for _ in range(scenarios_num - 1):
        ps = [np.round(task_durations[i][_] * scale).astype(int) for i in tasks]
        scenarios.append(ps)
    # print('scenarios:', scenarios)
    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    x_is = {}  # job i in the scenario s
    for s in range(scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            x_is[(i, s)] = mdl.interval_var(size=pis)

    # sequence variables:
    seq_s = {s: mdl.sequence_var([x_is[(i, s)] for i in tasks], name="_scenario_" + str(s))
             for s in range(scenarios_num)}

    # CONSTRAINTS:
    # end before start:
    for i, js in edge_dict.items():
        for j in js:
            for s in range(scenarios_num):
                mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # no overlap:
    for s in range(scenarios_num):
        mdl.add(mdl.no_overlap(seq_s[s]))

    # same sequences:
    for s in range(1, scenarios_num):
        mdl.add(mdl.same_sequence(seq_s[0], seq_s[s]))

    # without idle times:
    mdl.add(mdl.end_of(x_is[(last_task_id, 0)]) == np.sum(initial_durations))

    # OBJECTIVE:
    obj_s = {}
    for s in range(scenarios_num):
        sum_delta = mdl.sum([(mdl.start_of(x_is[(i, s)])
                              - mdl.start_of(x_is[(i, 0)])) for i in tasks])
        obj_s[s] = sum_delta
    agg_obj = mdl.sum([obj_s[s] for s in range(scenarios_num)])

    mdl.add(mdl.minimize(agg_obj))
    # mdl.add(mdl.minimize_static_lex([mdl.end_of(x_is[(last_task_id, 0)]),
    #                                  agg_obj]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=True)
    gap = msol.get_objective_gap()

    return get_seq_from_cplex_sol(msol, x_is, tasks), gap


def get_all_successors(edges: dict, v_id):
    order = []
    # Check is there such vertex:
    if v_id not in edges.keys():
        # There is no such start id
        return [v_id]

    successors = set()
    q = deque([v_id])
    while q:
        v = q.popleft()
        for next_v in [] if v not in edges.keys() else edges[v]:
            if next_v not in successors:
                successors.add(next_v)
                q.append(next_v)
        order.append(v)
    return order[1:]


def topological_sort(edges: dict):
    ts = TopologicalSorter(edges)
    order = list(ts.static_order())
    order.reverse()
    return order


def weighted_topological_sort(tasks_num: int, edge_dict: dict, weights: np.array, reverse=False):
    edge_dict_copy = copy.deepcopy(edge_dict)

    c = -1 if reverse else 1

    for i in range(tasks_num - 1):
        for j in range(i + 1, tasks_num):
            suc_i = get_all_successors(edge_dict_copy, i)
            suc_j = get_all_successors(edge_dict_copy, j)
            if weights[i] != weights[j] and i not in suc_j and j not in suc_i:
                fr = i if c * weights[i] < c * weights[j] else j
                to = j if c * weights[i] < c * weights[j] else i
                # print(f'Add edge from {fr} to {to}')
                if fr not in edge_dict_copy.keys():
                    edge_dict_copy[fr] = []
                edge_dict_copy[fr].append(to)

    # print(f'edges after adding: {edge_dict_copy}')
    order = topological_sort(edge_dict_copy)
    return order


def greedy_heuristic(tasks_num: int, edge_dict: dict, weights: np.array):
    edge_dict_copy = copy.deepcopy(edge_dict)
    all_suc_dict = dict()
    for i in range(tasks_num):
        all_suc_dict[i] = get_all_successors(edge_dict_copy, i)

    scheduled = []
    d = deque()
    for i in range(tasks_num):
        wi = weights[i]
        best_value, best_position = np.sum([1 if weights[j] >= wi else -1 for j in scheduled]), 0
        value, position = best_value, best_position
        for di in d:
            position += 1
            if i in all_suc_dict.keys() and di in all_suc_dict[i]:
                break
            if di in all_suc_dict.keys() and i in all_suc_dict[di]:
                best_position = position
            value += -1 if weights[di] >= wi else 1
            if value > best_value:
                best_value, best_position = value, position
        d.insert(best_position, i)
        scheduled.append(i)
    # print(d)
    return list(d)


def generate_durations(tasks_num: int, size: int, pi, di, d_type: str):
    for i in range(tasks_num):
        if di[i] < 0 or pi[i] < 0:
            raise ValueError("pi and di must be positive")

    if d_type == 'uniform':
        durations = [np.random.uniform(low=pi[i]-sqrt(6*di[i]), high=pi[i]+sqrt(6*di[i]), size=size) for i in tqdm(range(tasks_num))]
    elif d_type == 'normal':
        durations = [np.maximum(pi[i] - 4, np.minimum(pi[i] + 4, np.random.normal(pi[i], di[i], size))) for i in tqdm(range(tasks_num))]
    elif d_type == '3discrete':
        for dii in di:
            if dii > 0.5:
                raise ValueError('di should be between 0 and 0.5 for 3discrete distribution')
        durations = [np.random.choice([pi[i]-1, pi[i], pi[i]+1], size, p=[di[i], 1-2*di[i], di[i]]) for i in tqdm(range(tasks_num))]
    else:
        raise ValueError(f'Type {d_type} not supported')
    return durations


def exp_bubblesort(type: str):
    size = 10 ** 6
    N = 20
    experiments_num = 10
    objs_for_experiments = []
    for _ in range(experiments_num):
        print('Experiment', _, 'out of ', experiments_num)
        pi = np.random.uniform(5, 10, N)
        di = np.random.uniform(0, 1.5, N)
        tasks = generate_durations(N, size, pi, di, type)
        initial_seq = [i for i in range(N)]
        objs = bubble_sort_heuristic(pi, di, tasks, initial_seq)
        objs_for_experiments.append(objs)
    title = ''  # 'Равномерно распределенные продолжительности работ' if type == 'uniform' else 'Нормально распределенные продолжительности работ'
    draw_bubble_sort_heur_results(objs_for_experiments, type, title)


def exp_heuristic_vs_cp(d_type: str):
    size = 10 ** 6
    NOT_DUMMY_V_NUM = 50
    PROBLEMS_NUM = 50
    TIME_LIMIT = 120
    SCENARIOS_NUM = 60
    GRAPH_DIR = './Data/PrecedenceGraphs/FasterGeneratedGraphs/'

    output_dir = f'./Output/opt_experiment_metrics/MOTOR2025exp2_Output_{d_type}.txt'
    open(output_dir, 'w').close()

    graph_paths = [GRAPH_DIR + f'{NOT_DUMMY_V_NUM}_notDummyVertices/graph_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]

    for problem_id in range(PROBLEMS_NUM):
        graph = read_graph(graph_paths[problem_id])
        edges = graph.get_copy_of_all_edges()

        if d_type == 'uniform' or d_type == 'normal':
            pi = np.random.uniform(5, 10, NOT_DUMMY_V_NUM + 2)
            di = np.random.uniform(0, 1.5, NOT_DUMMY_V_NUM + 2)
        else:
            pi = np.random.choice(range(3, 10), NOT_DUMMY_V_NUM+2)
            di = np.random.uniform(0, 0.5, NOT_DUMMY_V_NUM + 2)

        tasks = generate_durations(NOT_DUMMY_V_NUM + 2, size, pi, di, d_type)
        # print(pi, di, tasks)

        heur_seq = greedy_heuristic(NOT_DUMMY_V_NUM + 2, edges, di)
        print(f'heur_seq: {heur_seq}')
        scale = 1 if d_type == '3discrete' else 10**4
        cplex_seq, gap = stoch_cplex_model(NOT_DUMMY_V_NUM + 2, NOT_DUMMY_V_NUM + 1, edges, pi, tasks, TIME_LIMIT,
                                           SCENARIOS_NUM, scale=scale)
        print(f'cplex_seq: {cplex_seq}')

        heur_obj = np.average(calculate_overlaps(init_durations=pi, task_durations=tasks, execution_order=heur_seq))
        cplex_obj = np.average(calculate_overlaps(init_durations=pi, task_durations=tasks, execution_order=cplex_seq))

        print(f'Heuristic: {heur_obj}, CPLEX: {cplex_obj}')
        with open(output_dir, 'a') as f:
            f.write(f'{problem_id}: Heuristic: {heur_obj}, CPLEX: {cplex_obj}' + '\n')

    title = ''
    draw_boxplot_for_heuristic_comp(output_dir, d_type, title)


def calculate_next_overlaps(dist_l, dist_r):
    dist = {}
    for kl, vl in dist_l.items():
        for kr, vr in dist_r.items():
            k = max([0, max([0, kl]) + kr])
            v = vl * vr
            if k not in dist.keys():
                dist[k] = 0
            dist[k] += v
    return dist


def calculate_overlaps_discrete_case(distributions, seq):
    overlaps = [{0: 1.}, calculate_next_overlaps({0: 1}, distributions[seq[0]])]
    for i in range(len(seq) - 1):
        next_overlaps = calculate_next_overlaps(distributions[seq[i]], distributions[seq[i + 1]])
        overlaps.append(next_overlaps)
    return overlaps


def calculate_expected_overlap(overlaps: dict):
    expected_overlap = 0
    for k, v in overlaps.items():
        expected_overlap += v*k
    return expected_overlap


def exp_discrete_case():
    num_tasks = 5
    distributions = [{-1: i/(3 * num_tasks), 0: 1 - 2 * i/(3 * num_tasks), 1: i/(3 * num_tasks)} for i in range(1, num_tasks + 1)]
    print(f'distributions: {distributions}')
    seqs = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [1, 2, 0, 3, 4], [2, 1, 0, 3, 4]]
    for seq in seqs:
        print(f'seq: {seq}')
        overlaps = calculate_overlaps_discrete_case(distributions, seq)
        print(f'Overlaps: {overlaps}')
        print(f'last overlaps: {overlaps[-1]}')
        print(f'expected last overlaps: {calculate_expected_overlap(overlaps[-1])}')
        sum_overlaps = 0.
        for overlap in overlaps:
            sum_overlaps += calculate_expected_overlap(overlap)
        print(f'sum expected overlaps: {sum_overlaps}')


if __name__ == '__main__':
    d_types = ['uniform', 'normal'] #  +['3discrete']
    for d_type in d_types:
        # exp_bubblesort(d_type)
        # exp_heuristic_vs_cp(d_type)
        output_dir = f'./Output/opt_experiment_metrics/MOTOR/MOTOR2025_Output_{d_type}.txt'
        draw_boxplot_for_heuristic_comp(output_dir, d_type, '')





