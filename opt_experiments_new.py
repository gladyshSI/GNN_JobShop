import math
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from DiscreteOpt.run_opt_models import run_cp_simp, get_metrics, run_qp_simp, run_cp_precedence_max, \
    run_milp_simp, run_milp_weights, run_qp_weights, run_cp_weights, run_cp_transitions, run_milp_durations, \
    run_qp_durations, add_metrics_to_file, run_cp_buffer_times, run_cp_combined, run_cp_stochastic, \
    run_cp_stochastic_avg_delta, run_cp_stochastic_max_delta, run_cp_multimode_buf, run_sgs_sjf, run_sgs_rand, \
    run_sgs_sjl
from class_graph_algs import print_networkx_graph, PGAlgorithms
from class_problem import Problem
from z_old_dadaset_generation import generate_graph, generate_complete_t_to_res
from opt_experiments_analysis import make_box_plot
from class_graph import PrecedenceGraph
from class_schedule import SchAlgorithms, print_schedule, Schedule, mean_of_distribution
from utilities import get_avg_deltas, rand_f_geom
from writers_readers import read_graph, read_tasks_to_dict


def make_experiment(problem_to_solve: Problem,
                    runner: Callable[[Problem, int], tuple[Schedule, float, float]]) -> tuple[Schedule, dict]:
    # TODO: FIX ALL RUNNERS (Don't forget to change the order of return parameters)
    schedule, gap, time = runner(problem_to_solve, TIME_LIMIT)
    m = get_metrics(schedule)
    m['gap'] = gap
    m['time'] = time
    return schedule, m


def print_sch_with_deltas(schedule: Schedule):
    exact_overlap_dist = schedule.calculate_exact_overlap_distributions()
    print_schedule(schedule, {i: mean_of_distribution(exact_overlap_dist[i]) for i in exact_overlap_dist.keys()})


if __name__ == "__main__":
    NOT_DUMMY_V_NUM = 50
    DISTRIBUTION_TYPE = 'normal'
    MACHINES_NUM = 6

    PROBLEMS_NUM = 100
    TIME_LIMIT = 25
    GRAPH_DIR = './Data/PrecedenceGraphs/FasterGeneratedGraphs/'
    TASKS_DIR = './Data/Tasks/'

    # EXPERIMENTS:
    graph_paths = [GRAPH_DIR + f'{NOT_DUMMY_V_NUM}_notDummyVertices/graph_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]
    tasks_paths = [TASKS_DIR + f'{DISTRIBUTION_TYPE}/tasks_{DISTRIBUTION_TYPE}_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]

    # [name, runner, metrics file path]
    experiments = np.array([
        # ['milp_simp', run_milp_simp, './Output/opt_experiment_metrics/metrics_milp_simp.txt'],
        # ['qp_simp', run_qp_simp, './Output/opt_experiment_metrics/metrics_qp_simp.txt'],
        ['cp_simp', run_cp_simp, './Output/opt_experiment_metrics/metrics_cp_simp.txt'],

        # ['milp_weights', run_milp_weights, './Output/opt_experiment_metrics/metrics_milp_weights.txt'],
        # ['qp_weights', run_qp_weights, './Output/opt_experiment_metrics/metrics_qp_weights.txt'],
        # ['cp_weights', run_cp_weights, './Output/opt_experiment_metrics/metrics_cp_weights.txt'],
        #
        # ['milp_durations', run_milp_durations, './Output/opt_experiment_metrics/metrics_milp_durations.txt'],
        # ['qp_durations', run_qp_durations, './Output/opt_experiment_metrics/metrics_qp_durations.txt'],
        # ['cp_buffer_times', run_cp_buffer_times, './Output/opt_experiment_metrics/metrics_cp_buffer_times.txt'],
        # ['cp_transitions', run_cp_transitions, './Output/opt_experiment_metrics/metrics_cp_transitions.txt'],
        # ['cp_stochastic', run_cp_stochastic, './Output/opt_experiment_metrics/metrics_cp_stochastic.txt'],
        # ['cp_stochastic_avg_delta', run_cp_stochastic_avg_delta, './Output/opt_experiment_metrics/metrics_cp_stochastic_avg_delta.txt'],
        # ['cp_stochastic_max_delta', run_cp_stochastic_max_delta, './Output/opt_experiment_metrics/metrics_cp_stochastic_max_delta.txt'],
        #
        # ['cp_precedence_max', run_cp_precedence_max, './Output/opt_experiment_metrics/metrics_cp_precedence_max.txt'],
        # ['cp_combined', run_cp_combined, './Output/opt_experiment_metrics/metrics_cp_combined.txt'],
        # ['cp_multimode_buf', run_cp_multimode_buf, './Output/opt_experiment_metrics/metrics_cp_multimode_buf.txt'] # TODO: Refactor

        ['SGS_Rand', run_sgs_rand, './Output/opt_experiment_metrics/metrics_sgs_rand.txt'],
        ['SGS_SJF', run_sgs_sjf, './Output/opt_experiment_metrics/metrics_sgs_sjf.txt'],
        ['SGS_SJL', run_sgs_sjl, './Output/opt_experiment_metrics/metrics_sgs_sjl.txt'],
    ])

    # Clean files with metrics:
    for metrics_file_path in experiments[:, 2]:
        open(metrics_file_path, 'w').close()

    sch_map = dict()
    metrics_map = dict()  # experiment_name (model) -> [metrics (different problems)]
    for problem_id in range(PROBLEMS_NUM):
        graph_file = graph_paths[problem_id]
        tasks_file = tasks_paths[problem_id]
        problem = Problem(read_graph(graph_file), read_tasks_to_dict(tasks_file), MACHINES_NUM)

        for experiment in experiments:
            print(f'#######\nName: {experiment[0]} \ngraph_f = {graph_file}\ntasks_file = {tasks_file}\n#######')
            schedule, metrics = make_experiment(problem_to_solve=problem, runner=experiment[1])

            experiment_name = experiment[0]
            if experiment_name not in sch_map.keys():
                sch_map[experiment_name] = []
            sch_map[experiment_name].append(schedule)
            if experiment_name not in metrics_map.keys():
                metrics_map[experiment_name] = []
            metrics_map[experiment_name].append(metrics)
            add_metrics_to_file(metrics, experiment[2], additional="from " + graph_file)


    # PRINT ONE SCHEDULE:
    problem_id = 0
    graph_to_draw = read_graph(graph_paths[problem_id])
    alg = PGAlgorithms(graph_to_draw)
    print_networkx_graph(alg.make_networkx_graph())
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
    all_metrics = [metrics_map[name] for name in experiments[:, 0]]  # 2D array (row: list of metrics for one model)
    labels = experiments[:, 0].tolist()
    make_box_plot(all_metrics, labels, problem_id)
