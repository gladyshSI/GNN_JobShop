# The idea is to write functions that takes pg and t_to_res and returns optimal Schedule
import time

import numpy as np

from DiscreteOpt.cplex_models import cplex_simple, cplex_pg_time_lags_max, cplex_weights, cplex_transitions, \
    cplex_buffer_times, cplex_combined_trans_pc_max, cplex_stochastic, cplex_stochastic_avg_delta, \
    cplex_stochastic_max_delta
from DiscreteOpt.gurobi_models import gurobi_qp_simple, gurobi_milp_simple, gurobi_qp_weights, gurobi_milp_weights, \
    gurobi_milp_durations, gurobi_qp_durations
from precedence_graph import PGAlgorithms
from schedule import Schedule, SchAlgorithms
from utilities import get_avg_deltas, rand_f_geom


def get_makespan_ub(pg, t_to_res):
    sch = Schedule(pg, t_to_res)
    sch.rand_sgs()
    return sch.get_makespan()


def get_longest_ps_dict(pg):
    pga = PGAlgorithms(pg)
    return pga.get_longest_passes()


def get_metrics(sch):
    metrics = dict()
    scha = SchAlgorithms(sch)
    deltas = get_avg_deltas(scha, 10000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
    last_id = sch.get_last_tasks()[0]

    metrics['makespan'] = sch.get_makespan()
    metrics['avg_delta'] = np.mean(list(deltas.values()))
    metrics['max_delta'] = np.max(list(deltas.values()))
    metrics['last_delta'] = deltas[last_id]

    return metrics


def add_metrics_to_file(metrics: dict, path_to_file, additional=""):
    with open(path_to_file, 'a') as f:
        f.write(additional + '\n')
        f.write(str(metrics) + '\n')


def run_cp_simp(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_simple(pg, t_to_res, r_num, longest_ps_list, makespan, time_limit=time_limit, log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_weights(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_weights(pg, t_to_res, r_num, longest_ps_list, makespan, time_limit=time_limit, log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_precedence_max(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_pg_time_lags_max(pg, t_to_res, r_num, longest_ps_list, makespan, p=None, time_limit=time_limit,
                                      log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_qp_simp(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res)
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = gurobi_qp_simple(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


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


def run_cp_buffer_times(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_buffer_times(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                  log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_transitions(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_transitions(pg, t_to_res, r_num, longest_ps_list, makespan=makespan, time_limit=time_limit,
                                 log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_combined(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_combined_trans_pc_max(pg, t_to_res, r_num, longest_ps_list, makespan=makespan,
                                           time_limit=time_limit,
                                           log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_stochastic(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_stochastic(pg, t_to_res, r_num, longest_ps_list, makespan=makespan,
                                time_limit=time_limit,
                                log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_stochastic_avg_delta(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_stochastic_avg_delta(pg, t_to_res, r_num, longest_ps_list, makespan=makespan,
                                time_limit=time_limit,
                                log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


def run_cp_stochastic_max_delta(pg, t_to_res, time_limit):
    makespan = get_makespan_ub(pg, t_to_res) + 10
    longest_ps_dict = get_longest_ps_dict(pg)
    longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]
    r_num = max([len(val) for val in t_to_res.values()])

    start_time = time.time()
    gap, sch = cplex_stochastic_max_delta(pg, t_to_res, r_num, longest_ps_list, makespan=makespan,
                                time_limit=time_limit,
                                log_output=True)
    end_time = time.time()

    return gap, sch, (end_time - start_time)


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
