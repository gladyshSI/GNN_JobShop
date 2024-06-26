import math

from matplotlib import pyplot as plt


def read_metrics_from_file(filename):
    read_metrics = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '{' and line[-2] == '}':
                metric = eval(line)
                read_metrics.append(metric)
    return read_metrics


def read_metrics_from_files(filenames):
    united_metrics = []
    for filename in filenames:
        united_metrics += read_metrics_from_file(filename)
    return united_metrics


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


if __name__ == '__main__':
    DATADIR = '../Output/'
    labels = [
        # 'milp_simp',
        # 'qp_simp',
        'cp_simp',
        # 'milp_weights',
        # 'qp_weights',
        # 'cp_weights',
        # 'milp_duration',
        # 'qp_duration',
        # 'cp_transitions',
        # 'cp_max',
        'cp_buffer_times',
        'cp_transitions',
        'cp_stochastic',
        'cp_stochastic_avg_delta',
        'cp_stochastic_max_delta',
    ]

    all_metrics = []
    for label in labels:
        files = [DATADIR + 'metrics_' + label + '.txt']
        metrics = read_metrics_from_files(files)
        all_metrics.append(metrics)

    keys = ['time',
            'makespan',
            'avg_delta',
            'max_delta',
            'gap',
            'last_delta']
    draw_comparing_boxplot(all_metrics, keys, labels, draw_problem_id=0, filename='../Output/comparing_boxplot.png')
