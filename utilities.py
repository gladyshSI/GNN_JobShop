import numpy as np
import torch

from schedule import Schedule, SchAlgorithms


# Calculate new task durations n times from rand_f distribution
# Calculate overlaps
# Take agg_f of the overlaps for each node
def get_avg_deltas(scha, n, rand_f=lambda lb, ub: np.random.randint(lb, ub + 1), agg_f=lambda x: np.mean(x)):
    deltas = dict()
    vertices = scha._sch._pg._vertices
    for i in range(n):
        new_durations = dict()
        for t in range(len(vertices)):
            lb = vertices[t]._d_min
            ub = vertices[t]._d_max
            new_durations[t] = rand_f(lb, ub)
        deltas_i = scha.calc_deltas(new_durations)
        for t, d in deltas_i.items():
            if t not in deltas.keys():
                deltas[t] = []
            deltas[t].append(d)
    for t, d in deltas.items():
        deltas[t] = agg_f(d)
    return deltas


# Random functions:
def rand_f_uniform(lb, ub):
    return np.random.randint(lb, ub + 1)


def rand_f_normal(lb, ub):
    p = np.round(np.random.normal(loc=(lb + ub) / 2, scale=(ub - lb) / 6)).astype(int)
    p = max(lb, p)
    p = min(ub, p)
    return p


def rand_f_geom(lb, ub):
    p = lb + np.random.geometric(0.6) - 1
    p = min(ub, p)
    return p
