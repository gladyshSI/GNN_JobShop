import copy
import time
from typing import Callable

from matplotlib import pyplot as plt

from DiscreteOpt.cplex_models import cplex_stochastic_avg_d_makespan_bound, cplex_stochastic_multi_mode_buf
from DiscreteOpt.print_results import read_metrics_from_file
from DiscreteOpt.run_opt_models import run_cp_simp, add_metrics_to_file, get_longest_ps_dict, get_metrics, \
    get_makespan_ub
from class_problem import Problem
from class_schedule import Schedule
from opt_experiments_new import print_sch_with_deltas
from writers_readers import read_graph, read_tasks_to_dict


def pareto_run_cp_stochastic_avg_delta(problem: Problem,
                                       max_makespan: int,
                                       time_limit: int) -> tuple[Schedule, int, float]:
    start_time = time.time()
    sch, gap = cplex_stochastic_avg_d_makespan_bound(problem, makespan=max_makespan,
                                                     scenarios_num=50,
                                                     time_limit=time_limit,
                                                     log_output=True)
    end_time = time.time()
    return sch, gap, (end_time - start_time)


def pareto_run_cp_stochastic_multi_mode_buf(problem: Problem,
                                            max_buf_size: int,
                                            time_limit: int,
                                            scenarios_num=50) -> (Schedule, float, float):
    start_time = time.time()
    sch, gap = cplex_stochastic_multi_mode_buf(problem,
                                               sum_of_buf=max_buf_size,
                                               scenarios_num=scenarios_num,
                                               time_limit=time_limit,
                                               log_output=True)
    end_time = time.time()

    return sch, gap, (end_time - start_time)


def find_min_makespan(problem: Problem, makespan_opt_runner: Callable[[Problem, int], tuple[Schedule, float, float]],
                      TIME_LIMIT: int) -> int:
    sch, gap, solv_time = makespan_opt_runner(problem, TIME_LIMIT)
    return sch.get_makespan()


if __name__ == '__main__':
    RUN = False
    NOT_DUMMY_V_NUM = 50
    DISTRIBUTION_TYPE = 'normal'
    MACHINES_NUM = 6

    PROBLEM_ID = 10
    TIME_LIMIT = 60
    GRAPH_DIR = './Data/PrecedenceGraphs/FasterGeneratedGraphs/'
    TASKS_DIR = './Data/Tasks/'

    MAX_MAKESPAN_COEF = 1.5
    STEPS_NUM = 20

    # EXPERIMENTS:
    graph_path = GRAPH_DIR + f'{NOT_DUMMY_V_NUM}_notDummyVertices/graph_{NOT_DUMMY_V_NUM + 2}_{PROBLEM_ID}.txt'
    tasks_path = TASKS_DIR + f'{DISTRIBUTION_TYPE}/tasks_{DISTRIBUTION_TYPE}_{NOT_DUMMY_V_NUM + 2}_{PROBLEM_ID}.txt'
    makespan_opt_runner = run_cp_simp
    # name, runner, output_f = 'cp_stochastic_avg_delta', pareto_runner_cp_stochastic_avg_delta, './Output/opt_experiment_metrics/pareto_metrics_cp_stochastic_avg_delta.txt'
    name, runner, output_f = 'cp_stochastic_multimode_buf', pareto_run_cp_stochastic_multi_mode_buf, './Output/opt_experiment_metrics/pareto_metrics_cp_stochastic_multimode_buf.txt'
    if RUN:
        open(output_f, 'w').close()
        problem = Problem(read_graph(graph_path), read_tasks_to_dict(tasks_path), MACHINES_NUM)

        print(f'#######\nName: {name} \ngraph_f = {graph_path}\ntasks_file = {tasks_path}\n#######')
        # min_makespan = find_min_makespan(problem, makespan_opt_runner, TIME_LIMIT)
        min_makespan = 69
        max_makespan = int(min_makespan * MAX_MAKESPAN_COEF)
        step = int((max_makespan - min_makespan) / STEPS_NUM)
        print(f'min makespan: {min_makespan}; max makespan: {max_makespan}; step: {step}')

        schedule_map = {}
        metrics_map = {}
        buf = 2  # it is hard to obtain any schedule in that time limit and min makespan
        for makespan_bound in range(min_makespan + buf, max_makespan + buf, step):
            print(f'makespan bound: {makespan_bound}')
            # schedule, gap, solving_time = runner(problem=problem, max_makespan=makespan_bound, time_limit=TIME_LIMIT)
            schedule, gap, solving_time = runner(problem=problem, max_buf_size=min(NOT_DUMMY_V_NUM, (makespan_bound - min_makespan) * MACHINES_NUM), time_limit=TIME_LIMIT)
            metrics = get_metrics(schedule)
            schedule_map[makespan_bound] = schedule
            metrics_map[makespan_bound] = metrics
            add_metrics_to_file(metrics, output_f, additional="from " + graph_path)

        # PRINT SCHEDULE FOR each STEP:
        for makespan_bound, schedule in schedule_map.items():
            sch = schedule_map[makespan_bound]
            print_sch_with_deltas(sch)

    # PLOT CREATION
    metrics = read_metrics_from_file(output_f)
    points = [(m['makespan'], m['avg_delta']) for m in metrics]
    xs, ys = zip(*points)

    # Create the plot
    plt.plot(xs, ys, label="y = avg_delta(makespan)", color="b", linestyle="-", linewidth=2)

    # Labels and title
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Plot of ys vs. xs")
    plt.legend()

    # Show the plot
    plt.show()
    print(metrics)
