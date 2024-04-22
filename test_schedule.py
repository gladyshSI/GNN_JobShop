import numpy as np

from precedence_graph import PrecedenceGraph
from schedule import Schedule, print_schedule, SchAlgorithms

# Create Precedence Graph with 50 vertices
pg = PrecedenceGraph()
pg.random_v(50)
pg.random_network(start_num_diap=(2, 4), end_num_diap=(2, 4), seed=1)

# Create list with resources ids
res = [0, 1, 2, 3, 4, 5]
t_to_res = dict()
for t in range(len(pg._vertices)):
    t_to_res[t] = res

# Generate random schedule with random SGS procedure
sch = Schedule(pg, t_to_res)
sch.rand_sgs()

# Edges in Schedule contains not only precedence relations, but also resource edges.
# All edges contain time lag as value in the dictionary.
print(sch._edges)

# Draw schedule
print_schedule(sch, sch.get_task_to_r_map())

# We can calculate overlaps in the schedule if the task durations will be changed
new_durations = dict()
for t in range(len(sch._pg._vertices)):
    lb = sch._pg._vertices[t]._d_min
    ub = sch._pg._vertices[t]._d_max
    new_durations[t] = np.random.randint(lb, ub + 1)

scha = SchAlgorithms(sch)
deltas = scha.calc_deltas(new_durations)
print_schedule(sch, colors=deltas)


