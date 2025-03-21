import math
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from DiscreteOpt.run_opt_models import run_cp_simp, get_metrics, run_qp_simp, run_cp_precedence_max, \
    run_milp_simp, run_milp_weights, run_qp_weights, run_cp_weights, run_cp_transitions, run_milp_durations, \
    run_qp_durations, add_metrics_to_file, run_cp_buffer_times, run_cp_combined, run_cp_stochastic, \
    run_cp_stochastic_avg_delta, run_cp_stochastic_max_delta, run_cp_multimode_buf, run_sgs_sjf, run_sgs_rand, \
    run_sgs_sjl, run_cp_stochastic_multi_mode_buf, run_cp_stochastic_avg_d_makespan_bound
from class_graph_algs import print_networkx_graph, PGAlgorithms
from class_problem import Problem
from z_old_dadaset_generation import generate_graph, generate_complete_t_to_res
from opt_experiments_analysis import make_box_plot
from class_graph import PrecedenceGraph
from class_schedule import SchAlgorithms, print_schedule, Schedule, mean_of_distribution
from utilities import get_avg_deltas, rand_f_geom
from writers_readers import read_graph, read_tasks_to_dict


def make_experiment(problem_to_solve: Problem,
                    runner: Callable[[Problem, int, dict], tuple[Schedule, float, float]],
                    parameters: dict) -> tuple[Schedule, dict]:
    # TODO: Add parameters to all runners
    schedule, gap, time = runner(problem_to_solve, TIME_LIMIT, parameters)
    m = get_metrics(schedule)
    m['gap'] = gap
    m['time'] = time
    return schedule, m


def print_sch_with_deltas(schedule: Schedule):
    exact_overlap_dist = schedule.calculate_exact_overlap_distributions()
    print_schedule(schedule, {i: mean_of_distribution(exact_overlap_dist[i]) for i in exact_overlap_dist.keys()})


if __name__ == "__main__":
    NOT_DUMMY_V_NUM = 60  # 50
    DISTRIBUTION_TYPE = 'normal'
    MACHINES_NUM = 5  # 6

    PROBLEMS_NUM = 1
    TIME_LIMIT = 30
    # GRAPH_DIR = './Data/PrecedenceGraphs/FasterGeneratedGraphs/'
    GRAPH_DIR = './Data/PrecedenceGraphs/parsedPSPLib/'
    TASKS_DIR = './Data/Tasks/'
    OUTPUT_DIR = './Output/opt_experiment_metrics/'

    # EXPERIMENTS:
    # graph_paths = [GRAPH_DIR + f'{NOT_DUMMY_V_NUM}_notDummyVertices/graph_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
    #                for i in range(PROBLEMS_NUM)]
    graph_paths = [GRAPH_DIR + f'graph_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]
    tasks_paths = [TASKS_DIR + f'{DISTRIBUTION_TYPE}/tasks_{DISTRIBUTION_TYPE}_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]

    # [name, runner, output_postfix, parameters]
    experiments = np.array([
        # {'name': 'milp_simp', 'runner': run_milp_simp},
        # {'name': 'qp_simp', 'runner': run_qp_simp},
        {'name': 'DET', 'runner': run_cp_simp},

        # {'name': 'milp_weights', 'runner': run_milp_weights},
        # {'name': 'qp_weights', 'runner': run_qp_weights},
        # {'name': 'cp_weights', 'runner': run_cp_weights},
        #
        # {'name': 'milp_durations', 'runner': run_milp_durations},
        # {'name': 'qp_durations', 'runner': run_qp_durations},
        # {'name': 'BT10', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.1}},
        {'name': 'BT40', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.4}},
        {'name': 'TR40', 'runner': run_cp_transitions, 'params': {'threshold': 0.4}},
        {'name': 'STrm', 'runner': run_cp_stochastic, 'params': {'N': 50}},
        {'name': 'STavg', 'runner': run_cp_stochastic_avg_delta, 'params': {'N': 50}},
        {'name': 'STmax', 'runner': run_cp_stochastic_max_delta, 'params': {'N': 50}},
        {'name': 'MB4', 'runner': run_cp_stochastic_avg_d_makespan_bound, 'params': {'N': 50, 'makespan_delta': 4, 'first_runner': run_cp_simp, 'first_runner_params': {}, 'first_time_limit': 20}},
        #
        # {'name': 'cp_precedence_max', 'runner': run_cp_precedence_max},
        # {'name': 'cp_combined', 'runner': run_cp_combined},
        {'name': 'MMbuf', 'runner': run_cp_stochastic_multi_mode_buf, 'params': {'N': 50, 'max_buf_size': 4*MACHINES_NUM}},

        # {'name': 'SGS_Rand', 'runner': run_sgs_rand},
        # {'name': 'SGS_SJF', 'runner': run_sgs_sjf},
        # {'name': 'SGS_SJL', 'runner': run_sgs_sjl},
    ])

    # Clean files with metrics:
    output_files = {experiment['name']: OUTPUT_DIR + f'metrics_{experiment['name']}.txt' for experiment in experiments}
    for i in range(len(experiments)):
        name = experiments[i]['name']
        open(output_files[name], 'w').close()

    sch_map = dict()
    metrics_map = dict()  # experiment_name (model) -> [metrics (different problems)]
    for problem_id in range(PROBLEMS_NUM):
        graph_file = graph_paths[problem_id]
        tasks_file = tasks_paths[problem_id]
        print(f'graph_f = {graph_file}\ntasks_f = {tasks_file}')
        problem = Problem(read_graph(graph_file), read_tasks_to_dict(tasks_file), MACHINES_NUM)

        for experiment in experiments:
            print(f'#######\nName: {experiment['name']} \ngraph_f = {graph_file}\ntasks_file = {tasks_file}\n#######')
            parameters = {} if 'params' not in experiment else experiment['params']
            schedule, metrics = make_experiment(problem_to_solve=problem, runner=experiment['runner'],
                                                parameters=parameters)

            experiment_name = experiment['name']
            if experiment_name not in sch_map.keys():
                sch_map[experiment_name] = []
            sch_map[experiment_name].append(schedule)
            if experiment_name not in metrics_map.keys():
                metrics_map[experiment_name] = []
            metrics_map[experiment_name].append(metrics)
            add_metrics_to_file(metrics, output_files[experiment_name], additional="from " + graph_file)

    # PRINT ONE SCHEDULE:
    problem_id = 0
    graph_to_draw = read_graph(graph_paths[problem_id])
    alg = PGAlgorithms(graph_to_draw)
    print_networkx_graph(alg.make_networkx_graph())
    experiment_names = [experiment['name'] for experiment in experiments]
    schedules = [sch_map[name][problem_id] for name in experiment_names]
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
    all_metrics = [metrics_map[name] for name in experiment_names]  # 2D array (row: list of metrics for one model)
    labels = experiment_names
    make_box_plot(all_metrics, labels, problem_id)
