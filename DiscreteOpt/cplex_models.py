import numpy as np
from docplex.cp.model import *

from schedule import Schedule, SchAlgorithms, print_schedule
from utilities import rand_f_geom, get_avg_deltas


def make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik):
    sch = Schedule(pg, t_to_res)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for i in range(len(pg._vertices)):
        for k in range(r_num):
            var_sol = msol.get_var_solution(rik[(i, k)])
            if var_sol.is_present():
                starting_times[i] = var_sol.get_start()
                chosen_resources[i] = k

    tasks = sorted(list(range(len(pg._vertices))), key=lambda x: starting_times[x])
    for t in tasks:
        sch.schedule_task(chosen_resources[t], t, starting_times[t])

    return sch


def make_schedule_from_cplex_stochastic(pg, t_to_res, r_num, msol, r_iks, fr_scenario=0):
    sch = Schedule(pg, t_to_res)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for i in range(len(pg._vertices)):
        for k in range(r_num):
            var_sol = msol.get_var_solution(r_iks[(i, k, fr_scenario)])
            if var_sol.is_present():
                starting_times[i] = var_sol.get_start()
                chosen_resources[i] = k

    tasks = sorted(list(range(len(pg._vertices))), key=lambda x: starting_times[x])
    for t in tasks:
        sch.schedule_task(chosen_resources[t], t, starting_times[t])

    return sch


def cplex_simple(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap([rik[(i, k)] for i in tasks]))

    # OBJECTIVE:
    # original:
    mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_pg_time_lags_max(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap([rik[(i, k)] for i in tasks]))

    # ADDITIONAL:
    # additional variables:

    # dk_sum = -1 * mdl.sum([
    #   mdl.abs(
    #     mdl.start_of(rik[(j, k)]) - mdl.start_of(rik[(i, k)])
    #   ) for j in tasks for i in range(j) for k in resources
    # ])

    prec_sum = -1 * mdl.sum(
        mdl.abs(
            mdl.start_of(xi[j]) - mdl.end_of(xi[i])
        ) for i, j in edge_list
    )

    # additional constraints:
    for k in resources[:-1]:
        mdl.add(mdl.sum([(i + 1) * mdl.presence_of(rik[(i, k)]) for i in tasks]) <=
                mdl.sum([(i + 1) * mdl.presence_of(rik[(i, k + 1)]) for i in tasks]))

    # OBJECTIVE:
    # original:
    # mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # with additional:
    mdl.add(mdl.minimize_static_lex([mdl.end_of(xi[last_task]),
                                     prec_sum  #+ dk_sum
                                     ]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    # print("makespan =", msol.get_objective_values()[0])
    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_weights(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # CREATE WEIGHTS AND COEFFICIENT
    weights = [1. / pi for pi in p]
    c = makespan * task_num

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap([rik[(i, k)] for i in tasks]))

    # OBJECTIVE:
    # original:
    mdl.add(mdl.minimize_static_lex([mdl.end_of(xi[last_task]),
                                     mdl.sum([weights[i] * mdl.end_of(xi[i]) for i in tasks])]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_buffer_times(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    transition_times = transition_matrix(11)
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        tr = np.max([0, 5 - i])
        for j in [3, 4, 5, 6, 7, 8, 9, 10]:
            transition_times.set_value(i, j, tr)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)
    # sequence variables:
    seq_k = {k: mdl.sequence_var([rik[(i, k)] for i in tasks],
                                 types=[p[i] for i in tasks], name="resource_" + str(k))
             for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap(seq_k[k], transition_times))

    # OBJECTIVE:
    # original:
    mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_transitions(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    transition_times = transition_matrix(11)
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        for j in [3, 4, 5, 6, 7, 8, 9, 10]:
            tr = 0
            if i <= 5:
                tr = np.max([0, 10 - i - j])
            transition_times.set_value(i, j, tr)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)
    # sequence variables:
    seq_k = {k: mdl.sequence_var([rik[(i, k)] for i in tasks],
                                 types=[p[i] for i in tasks], name="resource_" + str(k))
             for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap(seq_k[k], transition_times))

    # OBJECTIVE:
    # original:
    mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_combined_trans_pc_max(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2,
                                log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    transition_times = transition_matrix(11)
    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        for j in [3, 4, 5, 6, 7, 8, 9, 10]:
            tr = 0
            if i <= 5:
                tr = np.max([0, 10 - i - j])
            transition_times.set_value(i, j, tr)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        p_i = p[i]
        est_i = rl_passes_list[i][0]
        lst_i = makespan - rl_passes_list[i][1] + 1
        xi[i] = mdl.interval_var(start=[est_i, lst_i + p_i], size=p_i)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)
    # sequence variables:
    seq_k = {k: mdl.sequence_var([rik[(i, k)] for i in tasks],
                                 types=[p[i] for i in tasks], name="resource_" + str(k))
             for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        mdl.add(mdl.end_before_start(xi[i], xi[j]))

    # alternative:
    for i in tasks:
        mdl.add(mdl.alternative(xi[i], [rik[(i, k)] for k in resources]))

    # no overlap:
    for k in resources:
        mdl.add(mdl.no_overlap(seq_k[k], transition_times))

    prec_sum = -1 * mdl.sum(
        mdl.abs(
            mdl.start_of(xi[j]) - mdl.end_of(xi[i])
        ) for i, j in edge_list
    )

    # additional constraints:
    for k in resources[:-1]:
        mdl.add(mdl.sum([(i + 1) * mdl.presence_of(rik[(i, k)]) for i in tasks]) <=
                mdl.sum([(i + 1) * mdl.presence_of(rik[(i, k + 1)]) for i in tasks]))

    # OBJECTIVE:
    # original:
    # mdl.add(mdl.minimize(mdl.end_of(xi[last_task])))

    # with additional:
    mdl.add(mdl.minimize_static_lex([mdl.end_of(xi[last_task]),
                                     prec_sum  #+ dk_sum
                                     ]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_simple(pg, t_to_res, r_num, msol, rik)


def cplex_stochastic(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    scenarios_num = 200
    scenarios = [p]
    for _ in range(scenarios_num - 1):
        ps = [rand_f_geom(lb, ub) for (lb, ub) in [(v._d_min, v._d_max) for v in pg._vertices]]
        scenarios.append(ps)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    x_is = {}  # job i in the scenario s
    r_iks = {}  # job i assigned to the worker k at the scenario s
    for s in range(scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            est_i = rl_passes_list[i][0]
            lst_i = makespan - rl_passes_list[i][1] + 1
            x_is[(i, s)] = mdl.interval_var(start=[est_i, lst_i + pis], size=pis)
            for k in resources:
                r_iks[(i, k, s)] = mdl.interval_var(optional=True)

    # sequence variables:
    seq_ks = {(k, s): mdl.sequence_var([r_iks[(i, k, s)] for i in tasks],
                                       name="resource_" + str(k) + "_scenario_" + str(s))
              for s in range(scenarios_num) for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        for s in range(scenarios_num):
            mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # alternative:
    for i in tasks:
        for s in range(scenarios_num):
            mdl.add(mdl.alternative(x_is[(i, s)], [r_iks[(i, k, s)] for k in resources]))

    # no overlap:
    for k in resources:
        for s in range(scenarios_num):
            mdl.add(mdl.no_overlap(seq_ks[(k, s)]))

    # same sequences:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.same_sequence(seq_ks[(k, 0)], seq_ks[(k, s)]))

    # OBJECTIVE:
    obj_s = {}
    for s in range(scenarios_num):
        obj_s[s] = mdl.max([0,
                            mdl.start_of(x_is[(last_task, s)]) - mdl.start_of(x_is[(last_task, 0)])])
    agg_obj = mdl.max([obj_s[s] for s in range(scenarios_num)])

    mdl.add(mdl.minimize_static_lex([mdl.end_of(x_is[(last_task, 0)]),
                                     agg_obj]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_stochastic(pg, t_to_res, r_num, msol, r_iks, fr_scenario=0)


def cplex_stochastic_avg_delta(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    scenarios_num = 200
    scenarios = [p]
    for _ in range(scenarios_num - 1):
        ps = [rand_f_geom(lb, ub) for (lb, ub) in [(v._d_min, v._d_max) for v in pg._vertices]]
        scenarios.append(ps)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    x_is = {}  # job i in the scenario s
    r_iks = {}  # job i assigned to the worker k at the scenario s
    for s in range(scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            est_i = rl_passes_list[i][0]
            lst_i = makespan - rl_passes_list[i][1] + 1
            x_is[(i, s)] = mdl.interval_var(start=[est_i, lst_i + pis], size=pis)
            for k in resources:
                r_iks[(i, k, s)] = mdl.interval_var(optional=True)

    # sequence variables:
    seq_ks = {(k, s): mdl.sequence_var([r_iks[(i, k, s)] for i in tasks],
                                       name="resource_" + str(k) + "_scenario_" + str(s))
              for s in range(scenarios_num) for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        for s in range(scenarios_num):
            mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # alternative:
    for i in tasks:
        for s in range(scenarios_num):
            mdl.add(mdl.alternative(x_is[(i, s)], [r_iks[(i, k, s)] for k in resources]))

    # no overlap:
    for k in resources:
        for s in range(scenarios_num):
            mdl.add(mdl.no_overlap(seq_ks[(k, s)]))

    # same sequences:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.same_sequence(seq_ks[(k, 0)], seq_ks[(k, s)]))

    # OBJECTIVE:
    obj_s = {}
    for s in range(scenarios_num):
        sum_delta = mdl.sum([(mdl.start_of(x_is[(i, s)])
                              - mdl.start_of(x_is[(i, 0)])) for i in tasks])
        obj_s[s] = sum_delta
    agg_obj = mdl.sum([obj_s[s] for s in range(scenarios_num)])

    mdl.add(mdl.minimize_static_lex([mdl.end_of(x_is[(last_task, 0)]),
                                     agg_obj]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_stochastic(pg, t_to_res, r_num, msol, r_iks, fr_scenario=0)


def cplex_stochastic_max_delta(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=2, log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))

    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    # TODO: fix hardcode
    scenarios_num = 200
    scenarios = [p]
    for _ in range(scenarios_num - 1):
        ps = [rand_f_geom(lb, ub) for (lb, ub) in [(v._d_min, v._d_max) for v in pg._vertices]]
        scenarios.append(ps)

    # MODEL
    mdl = CpoModel()

    # VARIABLES:
    x_is = {}  # job i in the scenario s
    r_iks = {}  # job i assigned to the worker k at the scenario s
    for s in range(scenarios_num):
        for i in tasks:
            pis = scenarios[s][i]
            est_i = rl_passes_list[i][0]
            lst_i = makespan - rl_passes_list[i][1] + 1
            x_is[(i, s)] = mdl.interval_var(start=[est_i, lst_i + pis], size=pis)
            for k in resources:
                r_iks[(i, k, s)] = mdl.interval_var(optional=True)

    # sequence variables:
    seq_ks = {(k, s): mdl.sequence_var([r_iks[(i, k, s)] for i in tasks],
                                       name="resource_" + str(k) + "_scenario_" + str(s))
              for s in range(scenarios_num) for k in resources}

    # CONSTRAINTS:
    # end before start:
    for i, j in edge_list:
        for s in range(scenarios_num):
            mdl.add(mdl.end_before_start(x_is[(i, s)], x_is[(j, s)]))

    # alternative:
    for i in tasks:
        for s in range(scenarios_num):
            mdl.add(mdl.alternative(x_is[(i, s)], [r_iks[(i, k, s)] for k in resources]))

    # no overlap:
    for k in resources:
        for s in range(scenarios_num):
            mdl.add(mdl.no_overlap(seq_ks[(k, s)]))

    # same sequences:
    for k in resources:
        for s in range(1, scenarios_num):
            mdl.add(mdl.same_sequence(seq_ks[(k, 0)], seq_ks[(k, s)]))

    # OBJECTIVE:
    obj_s = {}
    for s in range(scenarios_num):
        max_delta = mdl.max([(mdl.start_of(x_is[(i, s)])
                              - mdl.start_of(x_is[(i, 0)])) for i in tasks])
        obj_s[s] = max_delta
    agg_obj = mdl.max([obj_s[s] for s in range(scenarios_num)])

    mdl.add(mdl.minimize_static_lex([mdl.end_of(x_is[(last_task, 0)]),
                                     agg_obj]))

    # Solve the model
    msol = mdl.solve(TimeLimit=time_limit, log_output=log_output)
    gap = msol.get_objective_gap()

    return gap, make_schedule_from_cplex_stochastic(pg, t_to_res, r_num, msol, r_iks, fr_scenario=0)
