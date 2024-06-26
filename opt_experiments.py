import numpy as np
from matplotlib import pyplot as plt

from DiscreteOpt.run_opt_models import run_cp_simp, get_metrics, run_qp_simp, run_cp_precedence_max, \
    run_milp_simp, run_milp_weights, run_qp_weights, run_cp_weights, run_cp_transitions, run_milp_durations, \
    run_qp_durations, add_metrics_to_file
from dadaset_generation import generate_graph, generate_complete_t_to_res
from precedence_graph import PrecedenceGraph
from schedule import SchAlgorithms, print_schedule
from utilities import get_avg_deltas, rand_f_geom

CREATE_GRAPHS = False
V_NUM = 50
R_NUM = 6
GRAPH_NUM = 100
TIME_LIMIT = 25
GRAPH_DIR = "./Data/PrecedenceGraphs/ForOptExperiments/"

if CREATE_GRAPHS:
    t_to_res = generate_complete_t_to_res(V_NUM, R_NUM)
    for i in range(GRAPH_NUM):
        path_to_the_graph = GRAPH_DIR + "gr_" + str(V_NUM) + "_" + str(i) + '.txt'
        pg = generate_graph(V_NUM)
        pg.write_to_file(path_to_the_graph)

# EXPERIMENTS:
graph_paths = [GRAPH_DIR + "gr_" + str(V_NUM) + "_" + str(i) + '.txt' for i in range(GRAPH_NUM)]

# Save schedules:
milp_simp_sch_list = []
qp_simp_sch_list = []
cp_simp_sch_list = []
milp_weights_sch_list = []
qp_weights_sch_list = []
cp_weights_sch_list = []
milp_durations_sch_list = []
qp_durations_sch_list = []
cp_transitions_sch_list = []
cp_max_sch_list = []

# Save metrics:
milp_simp_metrics = []
qp_simp_metrics = []
cp_simp_metrics = []
milp_weights_metrics = []
qp_weights_metrics = []
cp_weights_metrics = []
milp_durations_metrics = []
qp_durations_metrics = []
cp_transitions_metrics = []
cp_max_precedence_metrics = []

metrics_file_paths = {"milp_simp": './Output/metrics_milp_simp.txt',
                      "qp_simp": './Output/metrics_qp_simp.txt',
                      "cp_simp": './Output/metrics_cp_simp.txt',
                      "milp_weights": './Output/metrics_milp_weights.txt',
                      "qp_weights": './Output/metrics_qp_weights.txt',
                      "cp_weights": './Output/metrics_cp_weights.txt',
                      "milp_durations": './Output/metrics_milp_duration.txt',
                      "qp_durations": './Output/metrics_qp_duration.txt',
                      "cp_transitions": './Output/metrics_cp_transitions.txt',
                      "cp_max": './Output/metrics_cp_max.txt'}
for metrics_file_path in metrics_file_paths.values():
    open(metrics_file_path, 'w').close()

for graph_path in graph_paths:
    pg = PrecedenceGraph()
    pg.read_from_file(graph_path)

    # Create map from task_id to list of available resources
    v_num = len(pg._vertices)
    t_to_res = generate_complete_t_to_res(v_num, R_NUM)

    # GUROBI MILP SIMPLE:
    print("### " + graph_path + " ### MILP SIMPLE")
    gap, sch, time = run_milp_simp(pg, t_to_res, TIME_LIMIT)
    milp_simp_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    milp_simp_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["milp_simp"], additional="from " + graph_path)

    # GUROBI QP SIMPLE:
    print("### " + graph_path + " ### QP SIMPLE")
    gap, sch, time = run_qp_simp(pg, t_to_res, TIME_LIMIT)
    qp_simp_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    qp_simp_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["qp_simp"], additional="from " + graph_path)

    # CPLEX SIMPLE:
    print("### " + graph_path + " ### CPLEX SIMPLE")
    gap, sch, time = run_cp_simp(pg, t_to_res, TIME_LIMIT)
    cp_simp_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    cp_simp_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["cp_simp"], additional="from " + graph_path)

    # GUROBI MILP WEIGHTS
    print("### " + graph_path + " ### MILP WEIGHTS")
    gap, sch, time = run_milp_weights(pg, t_to_res, TIME_LIMIT)
    milp_weights_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    milp_weights_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["milp_weights"], additional="from " + graph_path)

    # GUROBI QP WEIGHTS
    print("### " + graph_path + " ### QP WEIGHTS")
    gap, sch, time = run_qp_weights(pg, t_to_res, TIME_LIMIT)
    qp_weights_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    qp_weights_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["qp_weights"], additional="from " + graph_path)

    # CPLEX WEIGHTS
    print("### " + graph_path + " ### CPLEX WEIGHTS")
    gap, sch, time = run_cp_weights(pg, t_to_res, TIME_LIMIT)
    cp_weights_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    cp_weights_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["cp_weights"], additional="from " + graph_path)

    # GUROBI MILP DURATIONS
    print("### " + graph_path + " ### MILP DURATIONS")
    gap, sch, time = run_milp_durations(pg, t_to_res, TIME_LIMIT)
    milp_durations_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    milp_durations_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["milp_durations"], additional="from " + graph_path)

    # GUROBI QP DURATIONS
    print("### " + graph_path + " ### QP DURATIONS")
    gap, sch, time = run_qp_durations(pg, t_to_res, TIME_LIMIT)
    qp_durations_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    qp_durations_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["qp_durations"], additional="from " + graph_path)

    # CPLEX TRANSITIONS
    print("### " + graph_path + " ### CPLEX TRANSITIONS")
    gap, sch, time = run_cp_transitions(pg, t_to_res, TIME_LIMIT)
    cp_transitions_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    cp_transitions_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["cp_transitions"], additional="from " + graph_path)

    # CPLEX MAXIMIZE PRECEDENCE TIME LAGS
    print("### " + graph_path + " ### CPLEX MAXIMIZE PRECEDENCE")
    gap, sch, time = run_cp_precedence_max(pg, t_to_res, TIME_LIMIT)
    cp_max_sch_list.append(sch)
    metrics = get_metrics(sch)
    metrics['gap'] = gap
    metrics['time'] = time
    cp_max_precedence_metrics.append(metrics)
    add_metrics_to_file(metrics, metrics_file_paths["cp_max"], additional="from " + graph_path)


# PRINT ONE SCHEDULE:
def print_sch_with_deltas(sch):
    scha = SchAlgorithms(sch)
    deltas = get_avg_deltas(scha, 1000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
    print_schedule(sch, deltas)


problem_id = 0
schedules = [milp_simp_sch_list[problem_id],
             qp_simp_sch_list[problem_id],
             cp_simp_sch_list[problem_id],
             milp_weights_sch_list[problem_id],
             qp_weights_sch_list[problem_id],
             cp_weights_sch_list[problem_id],
             milp_durations_sch_list[problem_id],
             qp_durations_sch_list[problem_id],
             cp_transitions_sch_list[problem_id],
             cp_max_sch_list[problem_id]]
for sch in schedules:
    print_sch_with_deltas(sch)


# PLOT CREATION
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 15), sharey=False)
all_metrics = [milp_simp_metrics,
               qp_simp_metrics,
               cp_simp_metrics,
               milp_weights_metrics,
               qp_weights_metrics,
               cp_weights_metrics,
               milp_durations_metrics,
               qp_durations_metrics,
               cp_transitions_metrics,
               cp_max_precedence_metrics]
labels = ['MILP Simp',
          'QP Simp',
          'CP Simp',
          'MILP Weights',
          'QP Weights',
          'CP Weights',
          'MILP Durations',
          'QP Durations',
          'CP Transitions',
          'CP Max Pr']
x = range(1, len(all_metrics)+1)
data = [[m['time'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[0, 0].scatter(x, point)
axs[0, 0].boxplot(data, labels=labels)
axs[0, 0].set_title('Times')

data = [[m['makespan'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[0, 1].scatter(x, point)
axs[0, 1].boxplot(data, labels=labels)
axs[0, 1].set_title('Makespan')

data = [[m['avg_delta'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[1, 0].scatter(x, point)
axs[1, 0].boxplot(data, labels=labels)
axs[1, 0].set_title('Avg Delta')

data = [[m['max_delta'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[1, 1].scatter(x, point)
axs[1, 1].boxplot(data, labels=labels)
axs[1, 1].set_title('Max Delta')

data = [[m['gap'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[2, 0].scatter(x, point)
axs[2, 0].boxplot(data, labels=labels)
axs[2, 0].set_title('Gap')

data = [[m['last_delta'] for m in metrics_list] for metrics_list in all_metrics]
point = [d[problem_id] for d in data]
axs[2, 1].scatter(x, point)
axs[2, 1].boxplot(data, labels=labels)
axs[2, 1].set_title('Last Delta')

plt.tight_layout()
fig_dir = "./Output/"
plt.savefig(fig_dir + 'opt_exp.png')
plt.show()
