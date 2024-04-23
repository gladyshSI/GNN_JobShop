from docplex.cp.model import *

from schedule import Schedule


def make_schedule_from_cplex(pg, t_to_res, r_num, msol, rik):
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


def cplex_optimize(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, log_output=True):
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
    msol = mdl.solve(TimeLimit=2, log_output=log_output)
    gap = msol.get_objective_gap()

    # print("makespan =", msol.get_objective_values()[0])
    return gap, make_schedule_from_cplex(pg, t_to_res, r_num, msol, rik)
