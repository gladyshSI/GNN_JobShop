import gurobipy as gp
from gurobipy import GRB

from schedule import Schedule
from Keys.gurobi_keys import GUROBI_OPTIONS


def make_schedule_from_qp_simple(pg, t_to_res, xit, rik):
    sch = Schedule(pg, t_to_res)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for index, res in xit.items():
        if abs(res - 1.0) < 0.001:
            starting_times[index[0]] = index[1]
    for index, res in rik.items():
        if abs(res - 1.0) < 0.001:
            chosen_resources[index[0]] = index[1]

    tasks = sorted(list(range(len(pg._vertices))), key=lambda x: starting_times[x])
    for t in tasks:
        sch.schedule_task(chosen_resources[t], t, starting_times[t])

    return sch


def gurobi_qp_simple(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                     log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        xit = model.addVars(task_num, makespan, vtype=GRB.BINARY, name='xit')
        rik = model.addVars(task_num, r_num, vtype=GRB.BINARY, name='rik')

        # Precedence Constraints (1):
        model.addConstrs((gp.quicksum(t * xit[edge[1], t] for t in H) -
                          gp.quicksum(t * xit[edge[0], t] for t in H) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # Resource Quadratic Constarints (2):
        model.addConstrs((gp.quicksum(rik[i, k] * xit[i, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for k in resources for t in H), name='resource')

        # One Start (3):
        model.addConstrs((1 == gp.quicksum(xit[i, t] for t in H)
                          for i in tasks), name='onestart')

        # One Resource (4):
        model.addConstrs((1 == gp.quicksum(rik[i, k] for k in resources)
                          for i in tasks), name='oneresource')

        # Boundaries (5):
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(makespan - rl_passes_list[i][1] + 1, makespan)),
                         name='bright')

        # Objective:
        model.setObjective(gp.quicksum(t * xit[last_task, t] for t in H), GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        xit, rik = {(i, t): xit[i, t].getAttr('X') for i in tasks for t in H}, \
            {(i, k): rik[i, k].getAttr('X') for i in tasks for k in resources}
        return gap, make_schedule_from_qp_simple(pg, t_to_res, xit, rik)


def make_schedule_from_milp_simple(pg, t_to_res, x_imt):
    sch = Schedule(pg, t_to_res)
    starting_times = dict()  # task_id -> st
    chosen_resources = dict()  # task_id -> res
    for (i, m, t), res in x_imt.items():
        if abs(res - 1.0) < 0.001:
            starting_times[i] = t
            chosen_resources[i] = m

    tasks = sorted(list(range(len(pg._vertices))), key=lambda x: starting_times[x])
    for t in tasks:
        sch.schedule_task(chosen_resources[t], t, starting_times[t])

    return sch


def gurobi_milp_simple(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                       log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p == None:
        p = []
        for v in range(task_num):
            p.append(pg.get_duration(v))

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        x_imt = model.addVars(task_num, r_num, makespan, vtype=GRB.BINARY, name='x_imt')

        # Precedence Constraints:
        model.addConstrs((gp.quicksum(t * x_imt[edge[1], m, t] for t in H for m in resources) -
                          gp.quicksum(t * x_imt[edge[0], m, t] for t in H for m in resources) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # No Overlap Constraints:
        model.addConstrs((gp.quicksum(x_imt[i, m, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for m in resources for t in H), name='resource')

        # One Start & One Resource Constraints:
        model.addConstrs((1 == gp.quicksum(x_imt[i, m, t]
                                           for t in H
                                           for m in resources)
                          for i in tasks), name='onestart')
        # Boundaries:
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(makespan - rl_passes_list[i][1] + 1, makespan)), name='bright')

        # OBJECTIVE:
        model.setObjective(gp.quicksum(t * x_imt[last_task, m, t] for t in H for m in resources), GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        x_imt = {(i, m, t): x_imt[i, m, t].getAttr('X')
                 for i in tasks
                 for m in resources
                 for t in H}

        return gap, make_schedule_from_milp_simple(pg, t_to_res, x_imt)


def gurobi_milp_weights(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                        log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
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

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        x_imt = model.addVars(task_num, r_num, makespan, vtype=GRB.BINARY, name='x_imt')

        # Precedence Constraints:
        model.addConstrs((gp.quicksum(t * x_imt[edge[1], m, t] for t in H for m in resources) -
                          gp.quicksum(t * x_imt[edge[0], m, t] for t in H for m in resources) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # No Overlap Constraints:
        model.addConstrs((gp.quicksum(x_imt[i, m, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for m in resources for t in H), name='resource')

        # One Start & One Resource Constraints:
        model.addConstrs((1 == gp.quicksum(x_imt[i, m, t]
                                           for t in H
                                           for m in resources)
                          for i in tasks), name='onestart')
        # Boundaries:
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(makespan - rl_passes_list[i][1] + 1, makespan)), name='bright')

        # OBJECTIVE:
        model.setObjective(c * gp.quicksum(t * x_imt[last_task, m, t] for t in H for m in resources) +
                           gp.quicksum(weights[i] * t * x_imt[i, m, t] for t in H for m in resources for i in tasks),
                           GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        x_imt = {(i, m, t): x_imt[i, m, t].getAttr('X')
                 for i in tasks
                 for m in resources
                 for t in H}

        return gap, make_schedule_from_milp_simple(pg, t_to_res, x_imt)


def gurobi_qp_weights(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                      log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
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

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        xit = model.addVars(task_num, makespan, vtype=GRB.BINARY, name='xit')
        rik = model.addVars(task_num, r_num, vtype=GRB.BINARY, name='rik')

        # Precedence Constraints (1):
        model.addConstrs((gp.quicksum(t * xit[edge[1], t] for t in H) -
                          gp.quicksum(t * xit[edge[0], t] for t in H) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # Resource Quadratic Constarints (2):
        model.addConstrs((gp.quicksum(rik[i, k] * xit[i, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for k in resources for t in H), name='resource')

        # One Start (3):
        model.addConstrs((1 == gp.quicksum(xit[i, t] for t in H)
                          for i in tasks), name='onestart')

        # One Resource (4):
        model.addConstrs((1 == gp.quicksum(rik[i, k] for k in resources)
                          for i in tasks), name='oneresource')

        # Boundaries (5):
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(makespan - rl_passes_list[i][1] + 1, makespan)),
                         name='bright')

        # Objective:
        model.setObjective(c * gp.quicksum(t * xit[last_task, t] for t in H) +
                           gp.quicksum(weights[i] * t * xit[i, t] for t in H for i in tasks),
                           GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        xit, rik = {(i, t): xit[i, t].getAttr('X') for i in tasks for t in H}, \
            {(i, k): rik[i, k].getAttr('X') for i in tasks for k in resources}
        return gap, make_schedule_from_qp_simple(pg, t_to_res, xit, rik)


def gurobi_milp_durations(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                          log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p is None:
        p = []
        for v in range(task_num):
            duration = pg.get_duration(v)
            duration = max([5, duration])
            p.append(duration)

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        x_imt = model.addVars(task_num, r_num, makespan, vtype=GRB.BINARY, name='x_imt')

        # Precedence Constraints:
        model.addConstrs((gp.quicksum(t * x_imt[edge[1], m, t] for t in H for m in resources) -
                          gp.quicksum(t * x_imt[edge[0], m, t] for t in H for m in resources) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # No Overlap Constraints:
        model.addConstrs((gp.quicksum(x_imt[i, m, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for m in resources for t in H), name='resource')

        # One Start & One Resource Constraints:
        model.addConstrs((1 == gp.quicksum(x_imt[i, m, t]
                                           for t in H
                                           for m in resources)
                          for i in tasks), name='onestart')
        # Boundaries:
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == x_imt[i, m, t]
                          for i in tasks
                          for m in resources
                          for t in range(makespan - rl_passes_list[i][1] + 1, makespan)), name='bright')

        # OBJECTIVE:
        model.setObjective(gp.quicksum(t * x_imt[last_task, m, t] for t in H for m in resources), GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        x_imt = {(i, m, t): x_imt[i, m, t].getAttr('X')
                 for i in tasks
                 for m in resources
                 for t in H}

        return gap, make_schedule_from_milp_simple(pg, t_to_res, x_imt)


def gurobi_qp_durations(pg, t_to_res, r_num, rl_passes_list, makespan=1000, p=None, time_limit=24 * 60 * 60,
                        log_output=True):
    task_num = len(pg._vertices)
    tasks = list(range(task_num))
    last_task = task_num - 1
    resources = list(range(r_num))
    H = list(range(makespan))
    edge_list = []
    for i, js in pg._edges.items():
        for j in js:
            edge_list.append((i, j))

    if p is None:
        p = []
        for v in range(task_num):
            duration = pg.get_duration(v)
            duration = max([5, duration])
            p.append(duration)

    with gp.Env(params=GUROBI_OPTIONS) as env, gp.Model(env=env) as model:
        # MODEL:
        model.setParam("OutputFlag", 1 if log_output else 0)
        model.setParam('TimeLimit', time_limit)

        # VARIABLES:
        xit = model.addVars(task_num, makespan, vtype=GRB.BINARY, name='xit')
        rik = model.addVars(task_num, r_num, vtype=GRB.BINARY, name='rik')

        # Precedence Constraints (1):
        model.addConstrs((gp.quicksum(t * xit[edge[1], t] for t in H) -
                          gp.quicksum(t * xit[edge[0], t] for t in H) >= p[edge[0]]
                          for edge in edge_list), name='precedence')

        # Resource Quadratic Constarints (2):
        model.addConstrs((gp.quicksum(rik[i, k] * xit[i, tau]
                                      for i in tasks
                                      for tau in list(range(max(0, t - p[i] + 1), t + 1))) <= 1
                          for k in resources for t in H), name='resource')

        # One Start (3):
        model.addConstrs((1 == gp.quicksum(xit[i, t] for t in H)
                          for i in tasks), name='onestart')

        # One Resource (4):
        model.addConstrs((1 == gp.quicksum(rik[i, k] for k in resources)
                          for i in tasks), name='oneresource')

        # Boundaries (5):
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(rl_passes_list[i][0])), name='bleft')
        model.addConstrs((0 == xit[i, t] for i in tasks for t in range(makespan - rl_passes_list[i][1] + 1, makespan)),
                         name='bright')

        # Objective:
        model.setObjective(gp.quicksum(t * xit[last_task, t] for t in H), GRB.MINIMIZE)

        model.optimize()
        gap = model.MIPGap

        xit, rik = {(i, t): xit[i, t].getAttr('X') for i in tasks for t in H}, \
            {(i, k): rik[i, k].getAttr('X') for i in tasks for k in resources}
        return gap, make_schedule_from_qp_simple(pg, t_to_res, xit, rik)
