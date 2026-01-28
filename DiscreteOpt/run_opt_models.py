# The idea is to write functions that takes pg and t_to_res and returns optimal Schedule
import copy
import time

import numpy as np

from DiscreteOpt.cplex_models import cplex_simple, cplex_pg_time_lags_max, cplex_weights, cplex_transitions, \
    cplex_buffer_times, cplex_combined_trans_pc_max, cplex_stochastic, cplex_stochastic_avg_delta, \
    cplex_stochastic_max_delta, cplex_multimode_buffer_times, cplex_stochastic_multi_mode_buf, \
    cplex_stochastic_avg_d_makespan_bound
from DiscreteOpt.gurobi_models import gurobi_qp_simple, gurobi_milp_simple, gurobi_qp_weights, gurobi_milp_weights, \
    gurobi_milp_durations, gurobi_qp_durations
from class_graph import PrecedenceGraph
from class_problem import Problem
from class_schedule import Schedule, SchAlgorithms, mean_of_distribution, print_schedule, cut_distribution, \
    crop_distribution, sum_distributions, normalize_distribution
from DiscreteOpt.heuristic_models import rand_sgs
from utilities import get_avg_deltas, rand_f_geom


def get_makespan_ub(problem: Problem) -> float:
    sch = Schedule(problem)
    rand_sgs(sch)
    return sch.get_makespan()


def get_longest_ps_dict(problem: Problem) -> dict:
    return problem.get_longest_passes()


def get_metrics(sch: Schedule):
    metrics = dict()
    exact_overlap_dist = sch.calculate_exact_overlap_distributions()
    avg_deltas = {i: mean_of_distribution(exact_overlap_dist[i]) for i in exact_overlap_dist.keys()}

    last_id = next(iter(sch.get_last_tasks()))

    metrics['makespan'] = sch.get_makespan()
    metrics['avg_delta'] = np.mean(list(avg_deltas.values()))
    metrics['max_delta'] = np.max(list(avg_deltas.values()))
    metrics['last_delta'] = avg_deltas[last_id]
    return metrics


def add_metrics_to_file(metrics: dict, path_to_file, additional=""):
    with open(path_to_file, 'a') as f:
        f.write(additional + '\n')
        f.write(str(metrics) + '\n')


def run_cp_simp(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    makespan = get_makespan_ub(problem)
    longest_ps_dict = problem.get_longest_passes()
    longest_ps_list = [longest_ps_dict[i] for i in range(len(problem.get_all_ids()))]

    start_time = time.time()
    sch, gap = cplex_simple(problem, longest_ps_list, makespan, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_weights(problem: Problem, time_limit: int, execfile: str|None) -> (Schedule, float, float):
    makespan = get_makespan_ub(problem)
    longest_ps_dict = problem.get_longest_passes()
    longest_ps_list = [longest_ps_dict[i] for i in range(len(problem.get_all_ids()))]

    start_time = time.time()
    sch, gap = cplex_weights(problem, longest_ps_list, makespan, time_limit=time_limit, log_output=True, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_precedence_max(problem: Problem, time_limit: int, execfile: str|None) -> (Schedule, float, float):
    makespan = get_makespan_ub(problem)
    longest_ps_dict = problem.get_longest_passes()
    longest_ps_list = [longest_ps_dict[i] for i in range(len(problem.get_all_ids()))]

    start_time = time.time()
    sch, gap = cplex_pg_time_lags_max(problem, longest_ps_list, makespan, p=None, time_limit=time_limit,
                                      log_output=True, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_qp_simp(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str | None) -> (Schedule, float, float):
    makespan = get_makespan_ub(problem)
    longest_ps_dict = problem.get_longest_passes()
    longest_ps_list = [longest_ps_dict[i] for i in range(len(problem.get_all_ids()))]

    start_time = time.time()
    gap, sch = gurobi_qp_simple(problem, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                log_output=log_output)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_qp_weights(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_qp_weights(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                 log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_milp_simp(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_milp_simple(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                  log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_milp_weights(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_milp_weights(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                   log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def get_buf_from_threshold(init_dur: int, distribution: dict, threshold: float) -> int:
    durations_reversed = sorted(list(distribution.keys()), reverse=True)
    exceeding_p = 0.
    new_dur = durations_reversed[0]
    for dur_i in durations_reversed:
        exceeding_p = exceeding_p + distribution[dur_i]
        if dur_i <= init_dur:
            new_dur = init_dur
            break
        if exceeding_p > threshold:
            new_dur = dur_i
            break
    return new_dur - init_dur


def run_cp_buffer_times(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    # The buffer selection is determined by the threshold probability.
    # The buffer must be such that the probability of exceeding it is less than the threshold probability
    threshold_p = parameters['threshold']

    buffers = dict()  # task_id -> buffer
    for t_id in problem.get_all_ids():
        init_dur = problem.get_duration(t_id)
        distr = problem.get_task_distribution(t_id)
        buffers[t_id] = get_buf_from_threshold(init_dur, distr, threshold_p)

    start_time = time.time()
    sch, gap = cplex_buffer_times(problem, buffers, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_multimode_buf(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    # The buffer selection is determined by the threshold probability.
    # The buffer must be such that the probability of exceeding it is less than the threshold probability
    modes = parameters['modes']

    start_time = time.time()
    sch, gap = cplex_multimode_buffer_times(problem, modes, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def find_best_transition_time(first_duration: int,
                              first_distribution: dict[int, float],
                              second_duration: int,
                              second_distribution: dict[int, float],
                              threshold: float) -> int:
    cut_distribution(first_distribution, 0.001)
    cut_distribution(second_distribution, 0.001)
    max_transition = max(first_distribution.keys()) - first_duration
    t = 0
    for t in range(max_transition + 1):
        overlap_distribution = crop_distribution(first_distribution, first_duration + t)
        sum_distribution = sum_distributions([overlap_distribution, second_distribution])
        exceed_distribution = crop_distribution(sum_distribution, second_duration)
        normalize_distribution(exceed_distribution)
        prob_to_exceed = 1. - exceed_distribution[0]
        if prob_to_exceed <= threshold:
            break
    return t


def run_cp_transitions(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    # The buffer selection is determined by the threshold probability.
    # transition between tasks i and j -> t
    # P([sum of new durations] > p_i + t + p_j) <= threshold probability
    threshold_p = parameters['threshold']
    transitions = dict()  # first_job -> second_job -> transition_time
    for first_id in problem.get_all_ids():
        if first_id not in transitions.keys():
            transitions[first_id] = dict()

        first_duration = problem.get_duration(first_id)
        first_distribution = problem.get_task_distribution(first_id)
        for second_id in problem.get_all_ids():
            if first_id == second_id:
                transitions[first_id][second_id] = 0
            else:
                second_duration = problem.get_duration(second_id)
                second_distribution = problem.get_task_distribution(second_id)
                transitions[first_id][second_id] = find_best_transition_time(first_duration,
                                                                             first_distribution,
                                                                             second_duration,
                                                                             second_distribution,
                                                                             threshold_p)

    start_time = time.time()
    sch, gap = cplex_transitions(problem, transitions, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_combined(pg, t_to_res, time_limit, execfile: str|None):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_combined_trans_pc_max(pg, t_to_res, r_num, longest_ps_list, makespan=makespan,
                                           time_limit=time_limit,
                                           log_output=True, execfile=execfile)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_stochastic(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    scenarios_num = parameters['N']
    obj = parameters['obj']
    start_time = time.time()
    sch, gap = cplex_stochastic(problem, obj=obj, scenarios_num=scenarios_num, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_stochastic_avg_delta(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    scenarios_num = parameters['N']
    start_time = time.time()
    sch, gap = cplex_stochastic_avg_delta(problem, scenarios_num=scenarios_num, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_stochastic_max_delta(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    scenarios_num = parameters['N']
    start_time = time.time()
    sch, gap = cplex_stochastic_max_delta(problem, scenarios_num=scenarios_num, time_limit=time_limit, log_output=log_output, execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_stochastic_avg_d_makespan_bound(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    first_runner = parameters['first_runner']
    first_runner_params = parameters['first_runner_params']
    first_time_limit = parameters['first_time_limit']
    makespan_delta = parameters['makespan_delta']
    scenarios_num = parameters['N']
    obj = parameters['obj']

    # Find opt makespan, using first runner
    sch, gap, solv_time = first_runner(problem, first_time_limit, log_output, first_runner_params, execfile=execfile)
    makespan = sch.get_makespan() + makespan_delta

    start_time = time.time()
    sch, gap = cplex_stochastic_avg_d_makespan_bound(problem, makespan=makespan,
                                                     scenarios_num=scenarios_num,
                                                     obj=obj,
                                                     time_limit=time_limit,
                                                     log_output=log_output,
                                                     execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_cp_stochastic_multi_mode_buf(problem: Problem, time_limit: int, log_output: bool | None, parameters: dict, execfile: str|None) -> (Schedule, float, float):
    sum_of_buf = parameters['sum_of_buf']
    scenarios_num = parameters['N']
    max_b = parameters['max_b']
    obj = parameters['obj']
    start_time = time.time()
    sch, gap = cplex_stochastic_multi_mode_buf(problem,
                                               num_of_buf_modes=max_b+1,  # including buf=0
                                               sum_of_buf=sum_of_buf,
                                               scenarios_num=scenarios_num,
                                               obj=obj,
                                               time_limit=time_limit,
                                               log_output=log_output,
                                               execfile=execfile)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_milp_durations(pg, t_to_res, time_limit):
    for v in pg._vertices:
        duration = v._duration
        duration = max([5, duration])
        v._duration = duration

    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_milp_durations(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                     log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_qp_durations(pg, t_to_res, time_limit):
    for v in pg._vertices:
        duration = v._duration
        duration = max([5, duration])
        v._duration = duration

    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_qp_durations(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                   log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_sgs_rand(problem: Problem, time_limit: int) -> (Schedule, float, float):
    start_time = time.time()
    sch = Schedule(problem)
    rand_sgs(sch)
    gap = 101.
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_sgs_sjf(problem: Problem, time_limit: int) -> (Schedule, float, float):
    start_time = time.time()
    sch = Schedule(problem)
    rand_sgs(sch, f=lambda t_ids_list: min(t_ids_list, key=lambda t_id: len(problem.get_task_distribution(t_id))))
    gap = 101.
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def run_sgs_sjl(problem: Problem, time_limit: int) -> (Schedule, float, float):
    start_time = time.time()
    sch = Schedule(problem)
    rand_sgs(sch, f=lambda t_ids_list: max(t_ids_list, key=lambda t_id: len(problem.get_task_distribution(t_id))))
    gap = 101.
    end_time = time.time()

    return sch, gap, (end_time - start_time)
