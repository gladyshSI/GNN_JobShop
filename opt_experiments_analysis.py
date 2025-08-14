from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


from DiscreteOpt.print_results import read_metrics_from_file


def make_all_metrics_from_files(metrics_files):
    all_metrics = []
    for metrics_file in metrics_files:
        all_metrics.append(read_metrics_from_file(metrics_file))
    return all_metrics


def calculate_best_metrics(all_metrics: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    problems_num = len(all_metrics[0])
    metrics_keys = all_metrics[0][0].keys()
    best_metrics_for_each_problem = []
    for problem_id in range(problems_num):
        best_metrics = dict()
        for metric_key in metrics_keys:
            min_metric_value = all_metrics[0][problem_id][metric_key]
            for model_id in range(len(all_metrics)):
                min_metric_value = min(min_metric_value, all_metrics[model_id][problem_id][metric_key])
            best_metrics[metric_key] = min_metric_value
        best_metrics_for_each_problem.append(best_metrics)
    return best_metrics_for_each_problem


def make_box_plot(all_metrics: list[list[dict[str, float]]], labels: list[str], problem_id=0, title='', plot_file_name='plot.svg') -> None:
    # Draw the boxplot of differences with the best model for each solved problem
    # Best model can be different for each metric and for each problem
    models_num = len(all_metrics)
    problems_num = len(all_metrics[0])
    best_metrics_for_each_problem = calculate_best_metrics(all_metrics)

    labelsize = 22
    fontsize = 27
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 15), sharey=False)
    fig.suptitle(title, fontsize=30, fontweight='bold', y=0.999)

    # x = range(1, len(all_metrics) + 1)
    data = [[m['time'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['time'] for j in range(problems_num)] for i in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[0, 0].scatter(x, point)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[0, 0].boxplot(delta_with_min, labels=labels)
    axs[0, 0].set_title('Solution Time (Δ with min.)', fontsize=fontsize)

    data = [[m['gap'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['gap'] for j in range(problems_num)] for i in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[0, 1].scatter(x, point)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[0, 1].boxplot(delta_with_min, labels=labels)
    axs[0, 1].set_title('Gap (Δ with min.)', fontsize=fontsize)

    data = [[m['makespan'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['makespan'] for j in range(problems_num)] for i in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[1, 0].scatter(x, point)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[1, 0].boxplot(delta_with_min, labels=labels)
    axs[1, 0].set_title('Initial Makespan (Δ with min.)', fontsize=fontsize)

    data = [[m['last_delta'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['last_delta'] for j in range(problems_num)] for i
                      in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[1, 1].scatter(x, point)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[1, 1].boxplot(delta_with_min, labels=labels)
    axs[1, 1].set_title('RM (Δ with min.)', fontsize=fontsize)

    data = [[m['avg_delta'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['avg_delta'] for j in range(problems_num)] for i in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[2, 0].scatter(x, point)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[2, 0].boxplot(delta_with_min, labels=labels)
    axs[2, 0].set_title('SM1 (Δ with min.)', fontsize=fontsize)

    data = [[m['max_delta'] for m in metrics_list] for metrics_list in all_metrics]
    delta_with_min = [[data[i][j] - best_metrics_for_each_problem[j]['max_delta'] for j in range(problems_num)] for i in
                      range(models_num)]
    # point = [d[problem_id] for d in data]
    # axs[2, 1].scatter(x, point)
    axs[2, 1].tick_params(axis='both', which='major', labelsize=labelsize)
    axs[2, 1].boxplot(delta_with_min, labels=labels)
    axs[2, 1].set_title('SM2  (Δ with min.)', fontsize=fontsize)

    plt.tight_layout()
    fig_dir = "./Output/plots/"
    plt.savefig(fig_dir + plot_file_name)
    plt.show()


def calculate_place_distributions(all_metrics: list[list[dict[str, float]]], labels: list[str], step: float) -> dict[str, dict]:
    place_distributions = dict()  # model_name -> metric -> dict(place -> num)
    models_num = len(all_metrics)
    problem_num = len(all_metrics[0])
    metric_keys = all_metrics[0][0].keys()

    # Create dict:
    for model_id in range(models_num):
        model_name = labels[model_id]
        if model_name not in place_distributions.keys():
            place_distributions[model_name] = dict()
        for metric_key in metric_keys:
            place_distributions[model_name][metric_key] = dict()

    for problem_id in range(problem_num):
        for metric_key in metric_keys:
            values = sorted([(model_id, all_metrics[model_id][problem_id][metric_key])
                             for model_id in range(models_num)], key=lambda x: x[1])
            place, place_value = 1, values[0][1]
            for model_id, value in values:
                model_name = labels[model_id]
                if value - place_value > step:  # If the value differ much with the previous values
                    place += 1
                    place_value = value
                if place not in place_distributions[model_name][metric_key].keys():
                    place_distributions[model_name][metric_key][place] = 0
                place_distributions[model_name][metric_key][place] += 1
    return place_distributions


def make_radar_charts(all_metrics_list_dif_exp: list[list[list[dict[str, float]]]], labels: list[list[str]],
                      experiments: list[(int, str)], title='', plot_file_name='radar.svg') -> None:
    num_vars = len(all_metrics_list_dif_exp[0][0][0].keys())
    metrics = all_metrics_list_dif_exp[0][0][0].keys()
    print(metrics)
    # Determine the number of plots needed (one for each set of all_metrics)
    num_plots = len(all_metrics_list_dif_exp)
    # TODO: fix hardcode
    if len(labels) == 1:
        labels = [labels[0] for _ in range(num_plots)]
    metrics_to_print = ['init. makespan', 'SM1', 'SM2', 'RM', 'Gap', 'solution time']

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Initialize the figure with subplots in a single row
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(8 * num_plots, 8), subplot_kw=dict(polar=True))

    # If there's only one subplot, axs won't be an array, so make it iterable
    if num_plots == 1:
        axs = [axs]

    # Plot each radar chart in its own subplot
    colors = [
        'teal', 'coral', 'purple', 'gold', 'blue', 'lime', 'orange',
        'magenta', 'cyan', 'brown', 'pink', 'red'
    ]

    for idx, all_metrics in enumerate(all_metrics_list_dif_exp):
        place_distributions = calculate_place_distributions(all_metrics, labels[idx], step=0.1)
        avg_place = dict()  # model_name -> metric -> avg_place
        for model_name, metric_to_dist in place_distributions.items():
            avg_place[model_name] = dict()
            for metric_key, dist in metric_to_dist.items():
                avg_place[model_name][metric_key] = sum([k * v for k, v in dist.items()]) / sum([v for v in dist.values()])

        df = pd.DataFrame(avg_place).T
        # Rename the index
        df.index.name = 'Model name'
        print(df)
        # Plotting radar chart
        for i, name in enumerate(df.index):
            values = df.loc[name].values
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            axs[idx].tick_params(axis='both', which='major', labelsize=20)
            axs[idx].fill(angles, values, color=colors[i % len(colors)], alpha=0.25, label=name)
            axs[idx].plot(angles, values, color=colors[i % len(colors)], linewidth=2)

        radial_ticks = [0., 1., 2., 3.]
        axs[idx].set_rgrids(radial_ticks, labels=[str(tick) for tick in radial_ticks], angle=0)
        # Optional formatting for radial labels (position and size can be adjusted)
        axs[idx].set_rgrids(radial_ticks, labels=[f'{tick:.2f}' for tick in radial_ticks], fontsize=12, angle=45)
        axs[idx].set_yticklabels([])  # Optionally remove radial grid lines
        axs[idx].set_xticks(angles[:-1])
        axs[idx].set_xticklabels(metrics_to_print)
        axs[idx].legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=23)  # bbox_to_anchor=(-0.15, 0.9)
        axs[idx].set_title(f'Distribution type: {experiments[idx][1]}', size=25, y=1.05)

    # Add a main title for the entire figure
    fig.suptitle(title, fontsize=30, fontweight='bold', y=0.99)

    # Adjust layout and spacing
    plt.tight_layout()
    fig_dir = "./Output/plots/"
    plt.savefig(fig_dir + plot_file_name, dpi=330)
    plt.show()


# def make_radar_chart(all_metrics: list[list[dict[str, float]]], labels: list[str]) -> None:
#     place_distributions = calculate_place_distributions(all_metrics, labels, step=0.1)
#     avg_place = dict()  # model_name -> metric -> avg_place
#     for model_name, metric_to_dist in place_distributions.items():
#         avg_place[model_name] = dict()
#         for metric_key, dist in metric_to_dist.items():
#             avg_place[model_name][metric_key] = sum([k * v for k, v in dist.items()]) / sum([v for v in dist.values()])
#
#     print(place_distributions)
#     df = pd.DataFrame(
#         avg_place).T
#
#     # Rename the index
#     df.index.name = 'Model name'
#     print(df)
#
#     labels = df.columns
#     num_vars = len(labels)
#
#     # Compute angles for each axis
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]  # Complete the circle
#
#     # Initialize the radar chart
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#
#     # Plot each row (name) in the DataFrame
#     colors = [
#         'teal', 'coral', 'purple', 'gold', 'blue', 'lime', 'orange',
#         'magenta', 'cyan', 'brown', 'pink', 'red'
#     ]
#     for i, name in enumerate(df.index):
#         values = df.loc[name].values
#         values = np.concatenate((values, [values[0]]))  # Complete the circle
#         ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25, label=name)
#         ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2)
#
#     # Configure the radar chart
#     ax.set_yticklabels([])  # Optionally remove radial grid lines
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels)
#     ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15))
#
#     plt.title('Radar Chart for All Models', size=15, y=1.1)
#
#     plt.tight_layout()
#     fig_dir = "./Output/plots/"
#     plt.savefig(fig_dir + 'opt_exp_radar_chart_avg_place.png')
#     plt.show()

def make_pair_comparison(label1, metrics1, label2, metrics2, metrix_name):
    fig = plt.figure()
    fig.suptitle(metrix_name + ': (' + label1 + ' - ' + label2 + ')')
    deltas = [(m1 - m2) for (m1, m2) in zip(metrics1, metrics2)]
    plt.hist(deltas, len(deltas))
    plt.show()


if __name__ == '__main__':
    MODE = 1
    if MODE == 1:
        labels = [
            'STrm2',
            'STrm10',
            'STrm30',
            'STrm50',
            'STrm100',
            'STrm150',
        ]
        no_dummy_tasks_num = 60
        distribution_type = 'exponential'
        problems_num = 50
        # metrics_files = [f'./Output/opt_experiment_metrics/{no_dummy_tasks_num}_tasks_{distribution_type}/metrics_{label}.txt' for label in labels]
        folder = 'N2-150exp'
        machines_num = 5
        metrics_files = [f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt' for label in labels]
        all_metrics = make_all_metrics_from_files(metrics_files)

        labels_to_print = [
            'STrm2',
            'STrm10',
            'STrm30',
            'STrm50',
            'STrm100',
            'STrm150',
        ]

        make_box_plot(all_metrics, labels_to_print, title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution')
    elif MODE == 2:
        # RADAR CHART
        labels = [
            'DET',
            'STrm30',
            'BT25',
            'TR35',
            'MBmS',
            'BBr',
        ]

        folder = '60PSPLib'
        machines_num = 5
        experiments = [(60, 'exponential')]# , (60, 'normal'), (60, 'exponential')]
        all_metrics_list_dif_exp = []
        for no_dummy_tasks_num, distribution_type in experiments:
            metrics_files = [
                f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
                for label in labels]
            all_metrics_one_exp = make_all_metrics_from_files(metrics_files)
            all_metrics_list_dif_exp.append(all_metrics_one_exp)

        labels_to_print_radar = [
            'DET',
            'STrm30',
            'BT25',
            'TR35',
            'MBmS',
            'BBr',
        ]
        title = "Average models' rank"
        make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title)
    else:
        print(f'Invalid MODE: {MODE}')
