from matplotlib import pyplot as plt

from DiscreteOpt.print_results import read_metrics_from_file


def make_all_metrics_from_files(metrics_files):
    all_metrics = []
    for metrics_file in metrics_files:
        all_metrics.append(read_metrics_from_file(metrics_file))
    return all_metrics


def make_box_plot(all_metrics, labels, problem_id=0):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 15), sharey=False)

    x = range(1, len(all_metrics) + 1)
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


def make_pair_comparison(label1, metrics1, label2, metrics2, metrix_name):
    fig = plt.figure()
    fig.suptitle(metrix_name + ': (' + label1 + ' - ' + label2 + ')')
    deltas = [(m1 - m2) for (m1, m2) in zip(metrics1, metrics2)]
    plt.hist(deltas, len(deltas))
    plt.show()


if __name__ == '__main__':
    labels = [
        # 'cp_simp',
        'cp_transitions',
        # 'cp_buffer_times',
        # 'cp_stochastic',
        # 'cp_stochastic_avg_delta',
        'cp_stochastic_max_delta',
        # 'cp_precedence_max',
        # 'cp_combined'
    ]

    metrics_files = ['./Output/metrics_' + label + '.txt' for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    # metrix_name = 'makespan'
    # metrix_name = 'avg_delta'
    metrix_name = 'max_delta'
    # metrix_name = 'last_delta'
    data = [[m[metrix_name] for m in metrics_list] for metrics_list in all_metrics]
    make_pair_comparison(labels[0], data[0], labels[1], data[1], metrix_name)