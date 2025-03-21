import os
from typing import Callable

import numpy as np

from DiscreteOpt.run_opt_models import run_cp_simp, get_metrics, add_metrics_to_file, run_cp_stochastic_multi_mode_buf
from class_graph_algs import print_networkx_graph, PGAlgorithms
from class_problem import Problem
from opt_experiments_analysis import make_box_plot
from class_schedule import print_schedule, Schedule, mean_of_distribution
from writers_readers import read_graph, read_tasks_to_dict


def add_sensitivity_metrics_to_file(m: dict, metrics_to_print, path_to_file):
    with open(path_to_file, 'a') as f:
        row_to_print = ', '.join([str(m[metric]) for metric in metrics_to_print])
        f.write(row_to_print+'\n')


def make_experiment_sensitivity(problem_to_solve: Problem,
                                runner: Callable[[Problem, int, int], tuple[Schedule, float, float]],
                                time_limits: list[int],
                                scenarios_nums: list[int],
                                problem_name: str,
                                output: str):
    metrics_to_print = ['problem_name', 'time_limit', 'scenarios_num', 'time', 'gap', 'makespan', 'avg_delta', 'max_delta', 'last_delta']
    if os.stat(output).st_size == 0:
        with open(output, 'a') as f:
            f.write(', '.join(metrics_to_print) + '\n')

    for scenarios_num in scenarios_nums:
        for time_limit in time_limits:
            schedule, gap, time = runner(problem_to_solve, time_limit, scenarios_num)
            m = get_metrics(schedule)
            m['gap'] = gap
            m['time'] = time
            m['time_limit'] = time_limit
            m['problem_name'] = problem_name
            m['scenarios_num'] = scenarios_num
            add_sensitivity_metrics_to_file(m, metrics_to_print, output)


if __name__ == "__main__":
    NOT_DUMMY_V_NUM = 50
    DISTRIBUTION_TYPE = 'normal'
    MACHINES_NUM = 6

    PROBLEMS_NUM = 1
    GRAPH_DIR = './Data/PrecedenceGraphs/FasterGeneratedGraphs/'
    TASKS_DIR = './Data/Tasks/'

    # EXPERIMENTS:
    graph_paths = [GRAPH_DIR + f'{NOT_DUMMY_V_NUM}_notDummyVertices/graph_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]
    tasks_paths = [TASKS_DIR + f'{DISTRIBUTION_TYPE}/tasks_{DISTRIBUTION_TYPE}_{NOT_DUMMY_V_NUM + 2}_{i}.txt'
                   for i in range(PROBLEMS_NUM)]

    # [name, runner, metrics file path]
    experiments = np.array([
        ['cp_stoch_multi_mode_buf', run_cp_stochastic_multi_mode_buf, './Output/opt_experiment_sensitivity/metrics_cp_stoch_multimode_buf.csv']
    ])

    # Clean files with metrics:
    for metrics_file_path in experiments[:, 2]:
        open(metrics_file_path, 'w').close()

    time_limits = [60 * k for k in [20]]
    scenarios_nums = [5, 50, 100, 150, 180]
    for experiment in experiments:
        for problem_id in range(PROBLEMS_NUM):
            graph_file = graph_paths[problem_id]
            tasks_file = tasks_paths[problem_id]
            problem = Problem(read_graph(graph_file), read_tasks_to_dict(tasks_file), MACHINES_NUM)

            print(f'#######\nName: {experiment[0]} \ngraph_f = {graph_file}\ntasks_file = {tasks_file}\n#######')
            problem_name = str(NOT_DUMMY_V_NUM) + DISTRIBUTION_TYPE + '_' + str(MACHINES_NUM) + '_' + str(problem_id)
            make_experiment_sensitivity(problem_to_solve=problem, runner=experiment[1], time_limits=time_limits,
                                        scenarios_nums=scenarios_nums, problem_name=problem_name, output=experiment[2])
