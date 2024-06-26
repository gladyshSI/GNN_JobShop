import math

import numpy as np
from matplotlib import pyplot as plt

from DiscreteOpt.run_opt_models import run_cp_simp, get_metrics, run_qp_simp, run_cp_precedence_max, \
    run_milp_simp, run_milp_weights, run_qp_weights, run_cp_weights, run_cp_transitions, run_milp_durations, \
    run_qp_durations, add_metrics_to_file, run_cp_buffer_times, run_cp_combined, run_cp_stochastic, \
    run_cp_stochastic_avg_delta, run_cp_stochastic_max_delta
from dadaset_generation import generate_graph, generate_complete_t_to_res
from opt_experiments_analysis import make_box_plot
from precedence_graph import PrecedenceGraph
from schedule import SchAlgorithms, print_schedule
from utilities import get_avg_deltas, rand_f_geom

CREATE_GRAPHS = False
V_NUM = 50
R_NUM = 6
GRAPH_NUM = 100
TIME_LIMIT = 60
GRAPH_DIR = "./Data/PrecedenceGraphs/ForOptExperiments/"

if CREATE_GRAPHS:
    t_to_res = generate_complete_t_to_res(V_NUM, R_NUM)
    for i in range(GRAPH_NUM):
        path_to_the_graph = GRAPH_DIR + "gr_" + str(V_NUM) + "_" + str(i) + '.txt'
        pg = generate_graph(V_NUM)
        pg.write_to_file(path_to_the_graph)

# EXPERIMENTS:
graph_paths = [GRAPH_DIR + "gr_" + str(V_NUM) + "_" + str(i) + '.txt' for i in range(GRAPH_NUM)]
sch_map = dict()
metrics_map = dict()
# [name, runner, metrics file path]
experiments = np.array([
    # ['milp_simp', run_milp_simp, './Output/metrics_milp_simp.txt'],
    # ['qp_simp', run_qp_simp, './Output/metrics_qp_simp.txt'],
    ['cp_simp', run_cp_simp, './Output/metrics_cp_simp.txt'],

    # ['milp_weights', run_milp_weights, './Output/metrics_milp_weights.txt'],
    # ['qp_weights', run_qp_weights, './Output/metrics_qp_weights.txt'],
    # ['cp_weights', run_cp_weights, './Output/metrics_cp_weights.txt'],
    #
    # ['milp_durations', run_milp_durations, './Output/metrics_milp_durations.txt'],
    # ['qp_durations', run_qp_durations, './Output/metrics_qp_durations.txt'],
    ['cp_buffer_times', run_cp_buffer_times, './Output/metrics_cp_buffer_times.txt'],
    ['cp_transitions', run_cp_transitions, './Output/metrics_cp_transitions.txt'],
    ['cp_stochastic', run_cp_stochastic, './Output/metrics_cp_stochastic.txt'],
    ['cp_stochastic_avg_delta', run_cp_stochastic_avg_delta, './Output/metrics_cp_stochastic_avg_delta.txt'],
    ['cp_stochastic_max_delta', run_cp_stochastic_max_delta, './Output/metrics_cp_stochastic_max_delta.txt'],
    #
    # ['cp_precedence_max', run_cp_precedence_max, './Output/metrics_cp_precedence_max.txt'],
    # ['cp_combined', run_cp_combined, './Output/metrics_cp_combined.txt'],
])

for metrics_file_path in experiments[:, 2]:
    open(metrics_file_path, 'w').close()


def make_experiment(name, graph_path, runner, metrics_file_path):
    print("### " + graph_path + " ### " + name)
    gap, sch, time = runner(pg, t_to_res, TIME_LIMIT)
    if name not in sch_map.keys():
        sch_map[name] = []
    sch_map[name].append(sch)
    m = get_metrics(sch)
    m['gap'] = gap
    m['time'] = time
    if name not in metrics_map.keys():
        metrics_map[name] = []
    metrics_map[name].append(m)
    add_metrics_to_file(m, metrics_file_path, additional="from " + graph_path)


for graph_path in graph_paths:
    pg = PrecedenceGraph()
    pg.read_from_file(graph_path)

    # Create map from task_id to list of available resources
    v_num = len(pg._vertices)
    t_to_res = generate_complete_t_to_res(v_num, R_NUM)

    for experiment in experiments:
        make_experiment(name=experiment[0],
                        graph_path=graph_path,
                        runner=experiment[1],
                        metrics_file_path=experiment[2])

# PRINT ONE SCHEDULE:
def print_sch_with_deltas(sch):
    scha = SchAlgorithms(sch)
    deltas = get_avg_deltas(scha, 10000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
    print_schedule(sch, deltas)


problem_id = 0
schedules = [sch_map[name][problem_id] for name in experiments[:, 0]]
for sch in schedules:
    ##############################################
    # LAST DELTA DEVIATION
    #
    # last_deltas = []
    # scha = SchAlgorithms(sch)
    # for i in range(100):
    #     deltas = get_avg_deltas(scha, 10000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
    #     last_deltas.append(deltas[49])
    # print(np.min(last_deltas), np.max(last_deltas), np.mean(last_deltas), np.std(last_deltas))
    # plt.hist(last_deltas, 10)
    # plt.show()
    ##############################################
    print_sch_with_deltas(sch)

# PLOT CREATION
all_metrics = [metrics_map[name] for name in experiments[:, 0]]
labels = experiments[:, 0].tolist()
make_box_plot(all_metrics, labels, problem_id)
