# import numpy as np
# from sklearn.metrics import r2_score
#
# from class_problem import Problem
# from class_schedule import Schedule
# from opt_experiments_new import print_sch_with_deltas
# from writers_readers import read_graph, read_tasks_to_dict
from docplex.cp.model import *
#
#
# def test(model, test_loader, device):
#     pred = []
#     real = []
#     for batch_idx, data in enumerate(test_loader):
#         data.to(device)
#         out = model(data)
#
#         pred.extend(out.detach().numpy().tolist())
#         real.extend(data.y.detach().numpy().tolist())
#
#     return np.mean((np.array(real) - np.array(pred)) ** 2), \
#         r2_score(y_pred=pred, y_true=real)


def cplex_simple(time_limit: int = 30, log_output=True):
    # MODEL
    mdl = CpoModel()

    tasks = [0, 1, 2, 3]
    resources = [0, 1]
    last_task = 3
    # VARIABLES:
    xi = {}
    rik = {}
    for i in tasks:
        xi[i] = mdl.interval_var(start=[0, 10], size=1)
        for k in resources:
            rik[(i, k)] = mdl.interval_var(optional=True)

    # CONSTRAINTS:
    # end before start:
    for i, j in [(0, 1), (1, 2), (2, 3)]:
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

    return gap




if __name__ == '__main__':
   cplex_simple()

