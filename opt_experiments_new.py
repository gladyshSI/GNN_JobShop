import math
import os
from importlib.metadata import distributions
from typing import Callable

import numpy as np
import csv
import time
import datetime

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
                    runner: Callable[[Problem, int, bool | None, dict], tuple[Schedule, float, float]],
                    time_limit: int,
                    parameters: dict) -> tuple[Schedule, dict]:
    # TODO: Add parameters to all runners
    log_output = None  # or True
    schedule, gap, time = runner(problem_to_solve, time_limit, log_output, parameters)
    m = get_metrics(schedule)
    m['gap'] = gap
    m['time'] = time
    return schedule, m


def print_sch_with_deltas(schedule: Schedule):
    exact_overlap_dist = schedule.calculate_exact_overlap_distributions()
    print_schedule(schedule, {i: mean_of_distribution(exact_overlap_dist[i]) for i in exact_overlap_dist.keys()})


def run_experiment(graph_file: str,
                   jobs_file: str,
                   output_file: str,
                   distribution: str,
                   machines_num: int,
                   time_limit: int,
                   experiments: np.array):
    # EXPERIMENTS:
    fieldnames = ['description', 'name', 'graph_f', 'jobs_f', 'distribution', 'jobs_num', 'workers_num', 'time_limit', 'schedule', 'gap', 'time', 'makespan', 'avg_delta',
                  'max_delta', 'last_delta', 'all_params']
    # Clean files with metrics:
    if not os.path.isfile(output_file):
        with open(output_file, 'w', newline='') as of:
            writer = csv.DictWriter(of, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

    # sch_map = dict()
    # metrics_map = dict()  # experiment_name (model) -> [metrics (different problems)]

    problem = Problem(read_graph(graph_file), read_tasks_to_dict(jobs_file), machines_num)
    jobs_num = len(problem.get_all_ids())
    machines_num = problem.get_machines_num()

    finished_exp = 0
    exp_t = 0
    total_max_t = time_limit * len(experiments)
    t_start = time.time()
    for exp_i, experiment in enumerate(experiments):
        t_now = time.time()
        dt = t_now - t_start
        rate = 1 if exp_t == 0 else dt / exp_t
        max_rest_t = total_max_t - exp_t
        exp_rest_t = max_rest_t * rate
        print(f'####### FINISHED EXP {finished_exp} / {len(experiments)} '
              f'TIME {dt // 3600:.0f}:{(dt % 3600) // 60:.0f}:{dt % 60:.0f} '
              f'RATE {rate:.2f} '
              f'MAX REST {max_rest_t // 3600:.0f}:{(max_rest_t % 3600) // 60:.0f}:{max_rest_t % 60:.0f} '
              f'EXPECTED {exp_rest_t // 3600:.0f}:{(exp_rest_t % 3600) // 60:.0f}:{exp_rest_t % 60:.0f}')
        finished_exp += 1
        exp_t += time_limit

        print(f'####### Name: {experiment['name']} ({exp_i + 1} / {len(experiments)}) TIME LIMIT {time_limit}')
        parameters = {} if 'params' not in experiment else experiment['params']
        schedule, metrics = make_experiment(problem_to_solve=problem, runner=experiment['runner'], time_limit=time_limit,
                                            parameters=parameters)
        print(f'Solved in {metrics['time']:.1f} s.; METRICS: {metrics}')

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writerow({'description': experiment['description'], 'name': experiment['name'],
                             'graph_f': graph_file, 'jobs_f': jobs_file, 'schedule': schedule.to_str(),
                             'distribution': distribution,
                             'jobs_num': jobs_num, 'workers_num': machines_num, 'time_limit': time_limit,
                             'gap': metrics['gap'], 'time': metrics['time'], 'makespan': metrics['makespan'],
                             'avg_delta': metrics['avg_delta'], 'max_delta': metrics['max_delta'],
                             'last_delta': metrics['last_delta'],
                             'all_params': str(parameters)})


    # # PRINT ONE SCHEDULE:
    # problem_id = 0
    # graph_to_draw = read_graph(graph_paths[problem_id])
    # alg = PGAlgorithms(graph_to_draw)
    # print_networkx_graph(alg.make_networkx_graph())
    # experiment_names = [experiment['name'] for experiment in experiments]
    # schedules = [sch_map[name][problem_id] for name in experiment_names]
    # for sch in schedules:
    #     ##############################################
    #     # LAST DELTA DEVIATION
    #     #
    #     # last_deltas = []
    #     # scha = SchAlgorithms(sch)
    #     # for i in range(100):
    #     #     deltas = get_avg_deltas(scha, 10000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
    #     #     last_deltas.append(deltas[49])
    #     # print(np.min(last_deltas), np.max(last_deltas), np.mean(last_deltas), np.std(last_deltas))
    #     # plt.hist(last_deltas, 10)
    #     # plt.show()
    #     ##############################################
    #     print_sch_with_deltas(sch)
    #
    # # PLOT CREATION
    # all_metrics = [metrics_map[name] for name in experiment_names]  # 2D array (row: list of metrics for one model)
    # labels = experiment_names
    # make_box_plot(all_metrics, labels, problem_id)


if __name__ == "__main__":
    NOT_DUMMY_V_NUMS = [192, 120, 60]
    MACHINES_NUMS = [10, 8, 5]
    FROM_PROBLEM_IDS = [0, 1, 0]
    PROBLEMS_NUMS = [0, 1, 0]
    TIME_LIMITS = [15 * 60, 10 * 60, 5 * 60]
    # DISTRIBUTIONS = ['uniform']  #, 'normal', 'exponential']
    DISTRIBUTIONS = ['normal'] #, 'exponential']

    GRAPH_FILES_F = [lambda ndv_num, i: './Data/PrecedenceGraphs/LOVATO_gr_aircraft1.txt',
                     lambda ndv_num, i: './Data/PrecedenceGraphs/parsedPSPLib/' + f'graph_{ndv_num + 2}_{i}.txt',
                     lambda ndv_num, i: './Data/PrecedenceGraphs/parsedPSPLib/' + f'graph_{ndv_num + 2}_{i}.txt']
    JOBS_FILES_F = [lambda dist_type, ndv_num, i: './Data/Tasks/' + f'{dist_type}/tasks_{dist_type}_{194}_{0}.txt',
                    lambda dist_type, ndv_num, i: './Data/Tasks/' + f'{dist_type}/tasks_{dist_type}_{ndv_num + 2}_{i}.txt',
                    lambda dist_type, ndv_num, i: './Data/Tasks/' + f'{dist_type}/tasks_{dist_type}_{ndv_num + 2}_{i}.txt']
    OUTPUT_FILES_F = [lambda dist_type, ndv_num, m_n, t_l, i: f'./Output/opt_experiment_schedules/output.csv',
                      lambda dist_type, ndv_num, m_n, t_l, i: f'./Output/opt_experiment_schedules/output.csv',
                      lambda dist_type, ndv_num, m_n, t_l, i: f'./Output/opt_experiment_schedules/output.csv']

    # [name, runner, output_postfix, parameters]
    experiments = np.array([
        # {'name': 'milp_simp', 'runner': run_milp_simp},
        # {'name': 'qp_simp', 'runner': run_qp_simp},


        # {'name': 'milp_weights', 'runner': run_milp_weights},
        # {'name': 'qp_weights', 'runner': run_qp_weights},
        # {'name': 'cp_weights', 'runner': run_cp_weights},
        #
        # {'name': 'milp_durations', 'runner': run_milp_durations},
        # {'name': 'qp_durations', 'runner': run_qp_durations},

        {'description': 'deterministic baseline model',
         'name': 'DET', 'runner': run_cp_simp},

        # Plot 1:
        {'description': 'Sensitivity analysis buffer time model (param q)',
         'name': 'BT10', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.1}},
        {'description': 'Sensitivity analysis buffer time model (param q)',
         'name': 'BT20', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.2}},
        {'description': 'Sensitivity analysis buffer time model (param q)',
         'name': 'BT30', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.3}},
        {'description': 'Sensitivity analysis buffer time model (param q)',
         'name': 'BT40', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.4}},
        {'description': 'Sensitivity analysis buffer time model (param q)',
         'name': 'BT50', 'runner': run_cp_buffer_times, 'params': {'threshold': 0.5}},

        {'description': 'Sensitivity analysis transitions time model (param q)',
         'name': 'TR40', 'runner': run_cp_transitions, 'params': {'threshold': 0.4}},
        {'description': 'Sensitivity analysis transitions time model (param q)',
         'name': 'TR45', 'runner': run_cp_transitions, 'params': {'threshold': 0.45}},
        {'description': 'Sensitivity analysis transitions time model (param q)',
         'name': 'TR50', 'runner': run_cp_transitions, 'params': {'threshold': 0.5}},
        {'description': 'Sensitivity analysis transitions time model (param q)',
         'name': 'TR55', 'runner': run_cp_transitions, 'params': {'threshold': 0.55}},
        {'description': 'Sensitivity analysis transitions time model (param q)',
         'name': 'TR60', 'runner': run_cp_transitions, 'params': {'threshold': 0.6}},

        # Plot 2:
        {'description': 'Obj. F. analysis SAA model (RM)',
         'name': 'STr', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'avg_rm'}},
        {'description': 'Obj. F. analysis SAA model (S1)',
         'name': 'STs1', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'avg_exp_ovp'}},
        {'description': 'Obj. F. analysis SAA model (S2)',
         'name': 'STs2', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'max_exp_ovp'}},
        {'description': 'Obj. F. analysis SAA model (additional Obj F 1)',
         'name': 'STmR', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'max_rm'}},
        {'description': 'Obj. F. analysis SAA model (additional Obj F 2)',
         'name': 'STmS', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'max_max_ovp'}},

        # Plot 3:
        {'description': 'Obj. F. analysis Model With Bounded Makespan (RM)',
         'name': 'MBr', 'runner': run_cp_stochastic_avg_d_makespan_bound,
         'params': {'N': 30, 'makespan_delta': 5, 'first_runner': run_cp_simp, 'first_runner_params': {},
                    'first_time_limit': 60, 'obj': 'avg_rm'}},
        {'description': 'Obj. F. analysis Model With Bounded Makespan (S1)',
         'name': 'MBs1', 'runner': run_cp_stochastic_avg_d_makespan_bound,
         'params': {'N': 30, 'makespan_delta': 5, 'first_runner': run_cp_simp, 'first_runner_params': {},
                    'first_time_limit': 60, 'obj': 'avg_exp_ovp'}},
        {'description': 'Obj. F. analysis Model With Bounded Makespan (S2)',
         'name': 'MBs2', 'runner': run_cp_stochastic_avg_d_makespan_bound,
         'params': {'N': 30, 'makespan_delta': 5, 'first_runner': run_cp_simp, 'first_runner_params': {},
                    'first_time_limit': 60, 'obj': 'max_exp_ovp'}},
        {'description': 'Obj. F. analysis Model With Bounded Makespan (additional Obj F 1)',
         'name': 'MBmR', 'runner': run_cp_stochastic_avg_d_makespan_bound,
         'params': {'N': 30, 'makespan_delta': 5, 'first_runner': run_cp_simp, 'first_runner_params': {},
                    'first_time_limit': 60, 'obj': 'max_rm'}},
        {'description': 'Obj. F. analysis Model With Bounded Makespan (additional Obj F 2)',
         'name': 'MBmS', 'runner': run_cp_stochastic_avg_d_makespan_bound,
         'params': {'N': 30, 'makespan_delta': 5, 'first_runner': run_cp_simp, 'first_runner_params': {},
                    'first_time_limit': 60, 'obj': 'max_max_ovp'}},

        # Plot 4:
        {'description': 'Obj. F. analysis Model With Bounded Number of Buffer Times (RM)',
         'name': 'BBr', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'avg_rm'}},
        {'description': 'Obj. F. analysis Model With Bounded Number of Buffer Times (S1)',
         'name': 'BBs1', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'avg_exp_ovp'}},
        {'description': 'Obj. F. analysis Model With Bounded Number of Buffer Times (S2)',
         'name': 'BBs2', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'max_exp_ovp'}},
        {'description': 'Obj. F. analysis Model With Bounded Number of Buffer Times (additional Obj F 1)',
         'name': 'BBmR', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'max_rm'}},
        {'description': 'Obj. F. analysis Model With Bounded Number of Buffer Times (additional Obj F 2)',
         'name': 'BBmS', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'max_max_ovp'}},

        # Plot 5:
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr2', 'runner': run_cp_stochastic, 'params': {'N': 2, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr10', 'runner': run_cp_stochastic, 'params': {'N': 10, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr30', 'runner': run_cp_stochastic, 'params': {'N': 30, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr50', 'runner': run_cp_stochastic, 'params': {'N': 50, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr100', 'runner': run_cp_stochastic, 'params': {'N': 100, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis SAA model (N)',
         'name': 'STr150', 'runner': run_cp_stochastic, 'params': {'N': 150, 'obj': 'avg_rm'}},

        # Plot 6:
        {'description': 'Sensitivity analysis Model With Bounded Number of Buffer Times (|b_j|)',
         'name': 'BBr1', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 1, 'sum_of_buf': 25, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis Model With Bounded Number of Buffer Times (|b_j|)',
         'name': 'BBr2', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 2, 'sum_of_buf': 25, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis Model With Bounded Number of Buffer Times (|b_j|)',
         'name': 'BBr3', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 3, 'sum_of_buf': 25, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis Model With Bounded Number of Buffer Times (|b_j|)',
         'name': 'BBr4', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 4, 'sum_of_buf': 25, 'obj': 'avg_rm'}},
        {'description': 'Sensitivity analysis Model With Bounded Number of Buffer Times (|b_j|)',
         'name': 'BBr5', 'runner': run_cp_stochastic_multi_mode_buf,
         'params': {'N': 30, 'max_b': 5, 'sum_of_buf': 25, 'obj': 'avg_rm'}},

        # {'name': 'SGS_Rand', 'runner': run_sgs_rand},
        # {'name': 'SGS_SJF', 'runner': run_sgs_sjf},
        # {'name': 'SGS_SJL', 'runner': run_sgs_sjl},
    ])

    distributions_num = len(DISTRIBUTIONS)
    pr_num = sum(PROBLEMS_NUMS)
    exp_num = len(experiments)
    total_runs = pr_num * distributions_num * exp_num
    print(f'EXPERIMENTS PER PROBLEM: {exp_num}')
    print(f'TOTAL RUNS: {total_runs}')
    t_st = time.time()
    finished = 0
    exp_dt = 0
    total_t = sum([t_l * p_n * distributions_num * exp_num for p_n, t_l in zip(TIME_LIMITS, PROBLEMS_NUMS)])
    print(f'TOTAL TIME: {total_t} s.')
    for i, (ndv_num, m_n, fr_p_i, p_n, t_l) in enumerate(zip(NOT_DUMMY_V_NUMS,
                                                  MACHINES_NUMS,
                                                  FROM_PROBLEM_IDS,
                                                  PROBLEMS_NUMS,
                                                  TIME_LIMITS)):
        # update experiments if needed:
        for e in experiments:
            if 'params' in e.keys() and 'sum_of_buf' in e['params'].keys():
                e['params']['sum_of_buf'] = 5 * m_n

        # Do experiments for each problem:
        for id in range(fr_p_i, fr_p_i + p_n):
            for distribution in DISTRIBUTIONS:
                t_now = time.time()
                dt = t_now - t_st
                rate = 1 if exp_dt == 0 else dt / exp_dt
                max_rest_t = total_t - exp_dt
                exp_rest_t = max_rest_t * rate
                print(f'\nFINISHED {finished * exp_num} / {total_runs} ({(100 * finished * exp_num / total_runs):.1f} %) '
                      f'TIME {dt//3600:.0f}:{(dt % 3600)//60:.0f}:{dt % 60:.0f} '
                      f'RATE {rate:.2f} '
                      f'MAX REST {max_rest_t//3600:.0f}:{(max_rest_t % 3600)//60:.0f}:{max_rest_t % 60:.0f} '
                      f'EXPECTED {exp_rest_t//3600:.0f}:{(exp_rest_t % 3600)//60:.0f}:{exp_rest_t % 60:.0f}')
                finished += 1
                exp_dt += t_l * exp_num


                graph_file = GRAPH_FILES_F[i](ndv_num, id)
                job_file = JOBS_FILES_F[i](distribution, ndv_num, id)
                output_file = OUTPUT_FILES_F[i](distribution, ndv_num, m_n, t_l, id)

                now = datetime.datetime.now()
                print(f'@@@@@ NOW: {now}')
                print(f'@@@@@ START EXPERIMENTS FOR DIST {distribution}, graph_file = {graph_file}, job_file = {job_file}, output_file = {output_file}')
                run_experiment(graph_file, job_file, output_file, distribution, m_n, t_l, experiments)

