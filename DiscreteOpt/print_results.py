import csv
import math
import os

from matplotlib import pyplot as plt


def read_metrics_from_file(filename):
    print(filename)
    experiment_name = filename.split('/')[-1].split('_')[-1].split('.')[0]
    distribution_type = filename.split('/')[-1].split('_')[1]
    workers_num = int(filename.split('/')[-1].split('_')[2])
    jobs_num = int(filename.split('/')[-1].split('_')[0]) + 2

    read_metrics = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        graph_f = ''
        graph_id = 0
        for line in lines:
            if line[0] == '{' and line[-2] == '}':
                metric = eval(line)
                metric['description'] = graph_id
                metric['name'] = experiment_name
                metric['graph_f'] = graph_f
                metric['distribution'] = distribution_type
                metric['workers_num'] = workers_num
                metric['jobs_num'] = jobs_num
                read_metrics.append(metric)
            else:
                graph_f = line.split(' ')[-1]
                graph_id = graph_f.split('/')[-1].split('.')[0].split('_')[-1]
    return read_metrics


def read_metrics_from_files(filenames):
    united_metrics = []
    for filename in filenames:
        united_metrics += read_metrics_from_file(filename)
    return united_metrics


def dump_all_metrics_to_csv(metrics: list[dict], output_file):
    fieldnames = ['description', 'name', 'graph_f', 'jobs_f', 'distribution', 'jobs_num', 'workers_num', 'time_limit',
                  'schedule', 'gap', 'time', 'makespan', 'avg_delta',
                  'max_delta', 'last_delta', 'all_params']
    # Clean files with metrics:
    if not os.path.isfile(output_file):
        with open(output_file, 'w', newline='') as of:
            writer = csv.DictWriter(of, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

    with open(output_file, 'a', newline='') as of:
        writer = csv.DictWriter(of, fieldnames=fieldnames, delimiter=';')
        for metric in metrics:
            writer.writerow({'description': metric['description'],'name': metric['name'],
                             'graph_f': metric['graph_f'],
                             'jobs_num': metric['jobs_num'], 'workers_num': metric['workers_num'],
                             'distribution': metric['distribution'],
                             'time_limit': 60,
                             'gap': metric['gap'], 'time': metric['time'], 'makespan': metric['makespan'],
                             'avg_delta': metric['avg_delta'], 'max_delta': metric['max_delta'],
                             'last_delta': metric['last_delta']})


def draw_comparing_boxplot(all_metrics, keys, labels, draw_problem_id=None, filename=None):
    n_cols = 2
    n_rows = math.ceil(len(keys) / 2)
    x_size = (25/8) * len(labels)
    y_size = (3/5) * x_size

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(x_size, y_size), sharey=False)
    x = []
    if draw_problem_id is not None:
        x = range(1, len(all_metrics) + 1)

    i = 0
    for key in keys:
        row_id = i // 2
        column_id = i % 2
        data = [[m[key] for m in metrics_list] for metrics_list in all_metrics]

        axs[row_id, column_id].boxplot(data, labels=labels)
        axs[row_id, column_id].set_title(key)
        if draw_problem_id is not None:
            point = [d[draw_problem_id] for d in data]
            axs[row_id, column_id].scatter(x, point)
        i += 1

    plt.tight_layout()
    if filename is None:
        filename = "../Output/comparing_boxplot.png"
    plt.savefig(filename)
    plt.show()


def combine_metrics_in_one_csv():
    files = ['../Output/opt_experiment_metrics/q10-60exp/' + f'60_{dist}_5_{name}.txt'
             for dist in ['exponential', 'uniform', 'normal']
             for name in ['BT10', 'BT20', 'BT30', 'BT35', 'BT40',
                          'TR30', 'TR45', 'TR50', 'TR55', 'TR60']]
    files += ['../Output/opt_experiment_metrics/difObjExp/' + f'60_{dist}_5_{name}.txt'
              for dist in ['exponential', 'uniform', 'normal']
              for name in ['STrm', 'STsm1', 'STsm2', 'STmaxRm', 'STmaxSm',
                           'MBr', 'MBs1', 'MBs2', 'MBmR', 'MBmS',
                           'BBr', 'BBs1', 'BBs2', 'BBmR', 'BBmS',]]
    files += ['../Output/opt_experiment_metrics/maxB1-5exp/' + f'60_{dist}_5_BB{name}.txt'
              for dist in ['exponential', 'uniform', 'normal']
              for name in [1, 2, 3, 4, 5]]
    files += ['../Output/opt_experiment_metrics/N2-150exp/' + f'60_{dist}_5_STrm{name}.txt'
              for dist in ['exponential', 'uniform', 'normal']
              for name in [2, 10, 30, 50, 150]]
    files += ['../Output/opt_experiment_metrics/60PSPLib/' + f'60_{dist}_5_{name}.txt'
             for dist in ['exponential', 'uniform', 'normal']
             for name in ['DET']]

    metrics = read_metrics_from_files(files)
    dump_all_metrics_to_csv(metrics, '../Output/opt_experiment_metrics/combined.csv')


if __name__ == '__main__':
    combine_metrics_in_one_csv()
    # DATADIR = '../Output/'
    # labels = [
    #     # 'milp_simp',
    #     # 'qp_simp',
    #     'cp_simp',
    #     # 'milp_weights',
    #     # 'qp_weights',
    #     # 'cp_weights',
    #     # 'milp_duration',
    #     # 'qp_duration',
    #     # 'cp_transitions',
    #     # 'cp_max',
    #     'cp_buffer_times',
    #     'cp_transitions',
    #     'cp_stochastic',
    #     'cp_stochastic_avg_delta',
    #     'cp_stochastic_max_delta',
    # ]
    #
    # all_metrics = []
    # for label in labels:
    #     files = [DATADIR + 'metrics_' + label + '.txt']
    #     metrics = read_metrics_from_files(files)
    #     all_metrics.append(metrics)
    #
    # keys = ['time',
    #         'makespan',
    #         'avg_delta',
    #         'max_delta',
    #         'gap',
    #         'last_delta']
    # # draw_comparing_boxplot(all_metrics, keys, labels, draw_problem_id=0, filename='../Output/comparing_boxplot.png')
