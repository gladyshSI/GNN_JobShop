import numpy as np

from dadaset_generation import generate_graph, generate_complete_t_to_res
from DiscreteOpt.cplex_models import cplex_simple, cplex_pg_time_lags_max
from schedule import SchAlgorithms, print_schedule
from utilities import get_avg_deltas, rand_f_geom
from DiscreteOpt.run_opt_models import get_longest_ps_dict

pg = generate_graph()
v_num = len(pg._vertices)
t_to_res = generate_complete_t_to_res(v_num, 6)
longest_ps_dict = get_longest_ps_dict(pg)
longest_ps_list = [longest_ps_dict[i] for i in range(len(pg._vertices))]

gap, sch_cplex = cplex_pg_time_lags_max(pg, t_to_res, 6, longest_ps_list, 100, log_output=True)
print("GAP =", gap)

scha_cplex = SchAlgorithms(sch_cplex)
deltas = get_avg_deltas(scha_cplex, 1000, rand_f=rand_f_geom, agg_f = lambda x: np.mean(x))
print("AVG color = ", np.mean([c for t, c in deltas.items()]))
print_schedule(sch_cplex, colors=deltas)
