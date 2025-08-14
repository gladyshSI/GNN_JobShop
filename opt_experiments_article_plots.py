from opt_experiments_analysis import make_all_metrics_from_files, make_box_plot, make_radar_charts


def draw_q10_60(distribution_type):
    labels = [
        'BT10',
        'BT20',
        'BT30',
        'BT35',
        'BT40',
        'TR30',
        'TR45',
        'TR50',
        'TR55',
        'TR60'
    ]
    no_dummy_tasks_num = 60
    problems_num = 100
    folder = 'q10-60exp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'BT10',
        'BT20',
        'BT30',
        'BT35',
        'BT40',
        'TR30',
        'TR45',
        'TR50',
        'TR55',
        'TR60'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'q10-60{distribution_type}.eps')


def draw_difObjExp(distribution_type):
    labels = [
        'STrm',
        'STsm1',
        'STsm2',
        'STmaxRm',
        'STmaxSm'
    ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = 'difObjExp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'STr',
        'STs1',
        'STs2',
        'STmR',
        'STmS'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'difObjExp_{distribution_type}.eps')


def draw_difObjExp_radar():
    labels = [
        'STrm',
        'STsm1',
        'STsm2'
    ]
    folder = 'difObjExp'
    machines_num = 5
    experiments = [(60, 'uniform'), (60, 'normal'), (60, 'exponential')]
    all_metrics_list_dif_exp = []
    for no_dummy_tasks_num, distribution_type in experiments:
        metrics_files = [
            f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
            for label in labels]
        all_metrics_one_exp = make_all_metrics_from_files(metrics_files)
        all_metrics_list_dif_exp.append(all_metrics_one_exp)

    labels_to_print_radar = [[
        'STr',
        'STs1',
        'STs2'
    ]for _ in range(3)]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title, plot_file_name='difObjExp_radar.png')


def draw_difObjExp5(distribution_type):
    labels = [
        'MBr',
        'MBs1',
        'MBs2',
        'MBmR',
        'MBmS'
    ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = 'difObjExp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'MBr',
        'MBs1',
        'MBs2',
        'MBmR',
        'MBmS'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'difObjExp5_{distribution_type}.eps')


def draw_difObjExp5_radar():
    labels = [
        'MBr',
        'MBs1',
        'MBs2',
        'MBmR',
        'MBmS'
    ]
    folder = 'difObjExp'
    machines_num = 5
    experiments = [(60, 'uniform'), (60, 'normal'), (60, 'exponential')]
    all_metrics_list_dif_exp = []
    for no_dummy_tasks_num, distribution_type in experiments:
        metrics_files = [
            f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
            for label in labels]
        all_metrics_one_exp = make_all_metrics_from_files(metrics_files)
        all_metrics_list_dif_exp.append(all_metrics_one_exp)

    labels_to_print_radar = [[
        'MBr',
        'MBs1',
        'MBs2',
        'MBmR',
        'MBmS'
    ] for _ in range(3)]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title, plot_file_name='difObjExp5.png')


def draw_difObjExp6(distribution_type):
    labels = [
        'BBr',
        'BBs1',
        'BBs2',
        'BBmR',
        'BBmS'
    ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = 'difObjExp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'BBr',
        'BBs1',
        'BBs2',
        'BBmR',
        'BBmS'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'difObjExp6_{distribution_type}.eps')

def draw_difObjExp6_radar():
    labels = [
        'BBr',
        'BBs1',
        'BBs2',
        'BBmR',
        'BBmS'
    ]
    folder = 'difObjExp'
    machines_num = 5
    experiments = [(60, 'uniform'), (60, 'normal'), (60, 'exponential')]
    all_metrics_list_dif_exp = []
    for no_dummy_tasks_num, distribution_type in experiments:
        metrics_files = [
            f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
            for label in labels]
        all_metrics_one_exp = make_all_metrics_from_files(metrics_files)
        all_metrics_list_dif_exp.append(all_metrics_one_exp)

    labels_to_print_radar = [[
        'BBr',
        'BBs1',
        'BBs2',
        'BBmR',
        'BBmS'
    ]for _ in range(3)]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title, plot_file_name='difObjExp6.png')


def draw_N2_150exp(distribution_type):
    labels = [
        'STrm2',
        'STrm10',
        'STrm30',
        'STrm50',
        'STrm100',
        'STrm150'
    ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = 'N2-150exp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'STrm2',
        'STrm10',
        'STrm30',
        'STrm50',
        'STrm100',
        'STrm150'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'N2-150exp_{distribution_type}.eps')


def draw_BBb1_5(distribution_type):
    labels = [
        'BB1',
        'BB2',
        'BB3',
        'BB4',
        'BB5'
    ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = 'maxB1-5exp'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = [
        'BB1',
        'BB2',
        'BB3',
        'BB4',
        'BB5'
    ]

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'BBb1-5{distribution_type}.eps')


def draw_60PSPLib(distribution_type):
    labels = []
    if distribution_type == 'uniform':
        labels = [
            'DET',
            'STrm30',
            'BT40',
            'TR53',
            'MBmS',
            'BBr'
        ]
    elif distribution_type == 'normal':
        labels = [
            'DET',
            'STrm30',
            'BT35',
            'TR40',
            'MBmS',
            'BBr'
        ]
    elif distribution_type == 'exponential':
        labels = [
            'DET',
            'STrm30',
            'BT25',
            'TR35',
            'MBmS',
            'BBr'
        ]
    no_dummy_tasks_num = 60
    problems_num = 50
    folder = '60PSPLib'
    machines_num = 5
    metrics_files = [
        f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
        for label in labels]
    all_metrics = make_all_metrics_from_files(metrics_files)

    labels_to_print = labels

    make_box_plot(all_metrics, labels_to_print,
                  title=f'{problems_num} Problems with {no_dummy_tasks_num} no dummy jobs and {distribution_type} duration distribution',
                  plot_file_name=f'60PSPLib_{distribution_type}.eps')


def draw_60PSPLib_radar():
    labels = []
    folder = '60PSPLib'
    machines_num = 5
    experiments = [(60, 'uniform'), (60, 'normal'), (60, 'exponential')]
    all_metrics_list_dif_exp = []
    for no_dummy_tasks_num, distribution_type in experiments:
        if distribution_type == 'uniform':
            labels = [
                'DET',
                'STrm30',
                'BT40',
                'TR53',
                'MBmS',
                'BBr'
            ]
        elif distribution_type == 'normal':
            labels = [
                'DET',
                'STrm30',
                'BT35',
                'TR40',
                'MBmS',
                'BBr'
            ]
        elif distribution_type == 'exponential':
            labels = [
                'DET',
                'STrm30',
                'BT25',
                'TR35',
                'MBmS',
                'BBr'
            ]
        metrics_files = [
            f'./Output/opt_experiment_metrics/{folder}/{no_dummy_tasks_num}_{distribution_type}_{machines_num}_{label}.txt'
            for label in labels]
        all_metrics_one_exp = make_all_metrics_from_files(metrics_files)
        all_metrics_list_dif_exp.append(all_metrics_one_exp)

    labels_to_print_radar = [[
        'DET',
        'STrm30',
        'BT40',
        'TR53',
        'MBmS',
        'BBr'
    ], [
        'DET',
        'STrm30',
        'BT35',
        'TR40',
        'MBmS',
        'BBr'
    ], [
        'DET',
        'STrm30',
        'BT25',
        'TR35',
        'MBmS',
        'BBr'
    ]]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title,
                      plot_file_name='60PSPLib_radar.png')


if __name__ == '__main__':
    distributions = ['uniform', 'normal', 'exponential']
    for distribution_type in distributions:
        draw_q10_60(distribution_type)
    draw_difObjExp_radar()
    draw_difObjExp5_radar()
    draw_difObjExp6_radar()
    for distribution_type in distributions:
        draw_N2_150exp(distribution_type)
        draw_BBb1_5(distribution_type)
        draw_60PSPLib(distribution_type)
    draw_60PSPLib_radar()
    for distribution_type in distributions:
        draw_difObjExp(distribution_type)
        draw_difObjExp5(distribution_type)
        draw_difObjExp6(distribution_type)
