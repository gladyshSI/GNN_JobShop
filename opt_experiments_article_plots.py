import pandas as pd

from opt_experiments_analysis import make_all_metrics_from_files, make_box_plot, make_radar_charts, \
    make_df_from_all_csv_files, filter_dataframe, df_in_latex, make_pivot_table_from_df, make_radar_charts_from_df, \
    make_box_plot_from_df


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
    ] for _ in range(3)]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title,
                      plot_file_name='difObjExp_radar.png')


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
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title,
                      plot_file_name='difObjExp5.png')


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
    ] for _ in range(3)]
    title = "Average models' rank"
    make_radar_charts(all_metrics_list_dif_exp, labels_to_print_radar, experiments, title,
                      plot_file_name='difObjExp6.png')


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


def main():
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


def make_pivot_tables(df: pd.DataFrame):
    # We build 3 pivot tables: for 194 jobs, 122 jobs, and 62 jobs:
    j_nums = [194, 122, 62]
    time_limits = [1800, 1200, 600]
    include_list = [{'jobs_num': [j_num], 'time_limit': [t_limit]} for j_num, t_limit in zip(j_nums, time_limits)]
    exclude = {}
    df_filtered_list = [filter_dataframe(df, include, exclude) for include in include_list]

    folder = '.\\Output\\latex_scripts\\'
    file_name_list = [f'latex_pivot_table_{j_num}.txt' for j_num in j_nums]
    for i, file_name in enumerate(file_name_list):
        with open(folder + file_name, 'w') as f:
            f.write(df_in_latex(make_pivot_table_from_df(df_filtered_list[i])))


def make_box_plots_for_article(df: pd.DataFrame,
                               j_nums=None,
                               time_limits=None,
                               distributions=None,
                               names: list[str] = None,
                               fig_dir='.\\Output\\plots\\occidata\\',
                               fig_name_suf='suf',
                               fig_name_ext='.svg'):
    if distributions is None:
        distributions = ['uniform', 'normal', 'exponential']
    if time_limits is None:
        time_limits = [1800, 1200, 600]
    if j_nums is None:
        j_nums = [194, 122, 62]

    include_list = [{'jobs_num': [j_num], 'time_limit': [t_limit], 'distribution': [dist], 'name': names}
                    for j_num, t_limit in zip(j_nums, time_limits)
                    for dist in distributions]
    metrics = ['gap', 'time', 'makespan', 'avg_delta', 'max_delta', 'last_delta']
    for include in include_list:
        df_filtered = filter_dataframe(df, include, {})

        df_filtered['name'] = pd.Categorical(df_filtered['name'], categories=names, ordered=True)
        df_filtered = df_filtered.sort_values('name')

        instances_num = df_filtered[df_filtered['name'] == names[0]].shape[0]
        plot_name = fig_name_suf + "-" + str(include['jobs_num'][0]) + include['distribution'][0] + fig_name_ext
        # title = (f"{instances_num} Instance{'s' if instances_num > 1 else ''}"
        #          f" with {str(include['jobs_num'][0] - 2)} no dummy jobs "
        #          f"and {include['distribution'][0]} duration distribution")
        title = (f"{instances_num} runs on the instance"
                 f" with {str(include['jobs_num'][0] - 2)} no dummy jobs "
                 f"and {include['distribution'][0]} duration distribution")
        make_box_plot_from_df(df_filtered, metrics, title=title, fig_dir=fig_dir, plot_file_name=plot_name)


def make_radar_charts_for_article(df: pd.DataFrame,
                                  j_nums=None,
                                  time_limits=None,
                                  distributions=None,
                                  names: list[str] = None,
                                  fig_dir='.\\Output\\plots\\occidata\\',
                                  fig_name_suf='suf',
                                  fig_name_ext='.png'):
    if distributions is None:
        distributions = ['uniform', 'normal', 'exponential']
    if time_limits is None:
        time_limits = [1800, 1200, 600]
    if j_nums is None:
        j_nums = [194, 122, 62]

    include_list_of_lists = [[{'jobs_num': [j_num], 'time_limit': [t_limit], 'distribution': [dist], 'name': names}
                              for dist in distributions]
                             for j_num, t_limit in zip(j_nums, time_limits)]
    metrics = ['gap', 'time', 'makespan', 'avg_delta', 'max_delta', 'last_delta']
    metrics_labels = ['Gap', 'solution time', 'init. makespan', 'SM1', 'SM2', 'RM']

    for include_list in include_list_of_lists:
        df_filtered_list = [filter_dataframe(df, include, {}) for include in include_list]
        plot_name = fig_name_suf + "_radar" + str(include_list[0]['jobs_num'][0]) + fig_name_ext
        j_num = include_list[0]['jobs_num'][0]
        title = f"Average models' rank for instances with {j_num-2} no dummy jobs"
        make_radar_charts_from_df(df_filtered_list, metrics, metrics_labels, title=title, fig_dir=fig_dir,
                                  plot_file_name=plot_name)








if __name__ == '__main__':
    df = make_df_from_all_csv_files('.\\Output\\occidata_2\\')
    # df = df.drop_duplicates(subset=['name', 'graph_f', 'jobs_f'], keep='last')
    ## PIVOT TABLES
    make_pivot_tables(df)
    ## Q10-60
    # names_q10_60 = ['BT10', 'BT20', 'BT30', 'BT40', 'BT50', 'TR40', 'TR45', 'TR50', 'TR55', 'TR60']
    # make_box_plots_for_article(df, names=names_q10_60, fig_name_suf='q10-60')
    ## N2-150EXP
    # names_N = ['STr2', 'STr10', 'STr30', 'STr50', 'STr100', 'STr150']
    # make_box_plots_for_article(df, names=names_N, fig_name_suf='N2-150exp')
    ## ObjST radars
    # names_ObjST = ['STr', 'STs1', 'STs2', 'STmR', 'STmS']
    # make_radar_charts_for_article(df, names=names_ObjST, fig_name_suf='ObjST', fig_name_ext='.svg')
    ## ObjMB radars
    # names_ObjMB = ['MBr', 'MBs1', 'MBs2', 'MBmR', 'MBmS']
    # make_radar_charts_for_article(df, names=names_ObjMB, fig_name_suf='ObjMB', fig_name_ext='.svg')
    ## ObjBB radars
    # names_ObjBB = ['BBr', 'BBs1', 'BBs2', 'BBmR', 'BBmS']
    # make_radar_charts_for_article(df, names=names_ObjBB, fig_name_suf='ObjBB', fig_name_ext='.svg')
    ## r Dimensionality
    names_BBrDim = ['BBr1', 'BBr2', 'BBr3', 'BBr4', 'BBr5']
    make_box_plots_for_article(df, names=names_BBrDim, fig_name_suf='BBr1-5')
    ## FINAL
    # names_Final = ['DET', 'BT40', 'TR40', 'STs2', 'MBs2', 'BBs2']
    # make_box_plots_for_article(df, names=names_Final, fig_name_suf='Final')
    # make_radar_charts_for_article(df, names=names_Final, fig_name_suf='Final', fig_name_ext='.svg')

    # names_N = ['BBr1', 'BBr2', 'BBr3', 'BBr4', 'BBr5']
    # make_box_plots_for_article(df, names=names_N, fig_name_suf='BBr1-5')

    # include_f = lambda d: {'jobs_num': [62],
    #                        'time_limit': [600],
    #                        'distribution': [d],
    #                        'name': ['MBr', 'MBs1', 'MBs2', 'MBmR', 'MBmS']}
    # exclude = {}
    # df_filtered_list = [filter_dataframe(df, include=include_f('uniform'), exclude=exclude),
    #                     filter_dataframe(df, include=include_f('normal'), exclude=exclude),
    #                     filter_dataframe(df, include=include_f('exponential'), exclude=exclude)]
    # metrics = ['gap', 'time', 'makespan', 'avg_delta', 'max_delta', 'last_delta']
    # metrics_labels = ['Gap', 'solution time', 'init. makespan', 'SM1', 'SM2', 'RM']
    # print("UNIFORM")
    # df_in_latex(make_pivot_table_from_df(df_filtered_list[0]))
    # print("NORMAL")
    # df_in_latex(make_pivot_table_from_df(df_filtered_list[1]))
    # print("EXPONENTIAL")
    # df_in_latex(make_pivot_table_from_df(df_filtered_list[2]))
    #
    # make_radar_charts_from_df(df_filtered_list, metrics, metrics_labels)
