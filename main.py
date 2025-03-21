import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from DiscreteOpt.run_opt_models import find_best_transition_time

print('Hello World!')
# look in training_experiments.py

from class_graph import PrecedenceGraph
from class_graph_algs import PGAlgorithms, print_networkx_graph, is_graph_disjunctive
from writers_readers import *

if __name__ == '__main__':
    a = 1
    if a:
        size = 10**8
        b_1, b_2, b_3 = 1, 2, 3
        task_1 = np.random.uniform(low=-b_1, high=b_1, size=size)
        task_2 = np.random.uniform(low=-b_2, high=b_2, size=size)
        task_3 = np.random.uniform(low=-b_3, high=b_3, size=size)
        # task_1 = np.random.normal(0, 3, size=size)
        # task_2 = np.random.normal(0, 2, size=size)
        seq = [task_2, task_1, task_3]
        overlaps_2 = np.maximum(0, seq[0])
        overlaps_3 = np.maximum(0, np.add(overlaps_2, seq[1]))
        overlaps_4 = np.maximum(0, np.add(overlaps_3, seq[2]))
        print(f'avg o_2 = {np.mean(overlaps_2)}')
        print(f'avg o_3 = {np.mean(overlaps_3)}')
        print(f'avg o_4 = {np.mean(overlaps_4)}')
        # bin_width = 0.01
        # bins = np.arange(bin_width, b_1 + b_2, bin_width)  # Create bins with specified width
        # plt.hist(overlaps, bins=bins, edgecolor='black')
        # plt.xlabel(f'b_1 {b_1}, b_2 {b_2}')
        # plt.ylabel('Frequency')
        # plt.title('new_sts')
        # plt.show()

    else:
        size = 10**6
        N = 15
        # tasks = [np.random.uniform(low=-i-1, high=i+1, size=size) for i in tqdm(range(N))]
        # tasks = [np.random.normal(loc=0, scale=i+1, size=size) for i in tqdm(range(N))]
        # tasks = [np.random.uniform(low=-(i + 1)**0.1, high=(i + 1)**0.1, size=size) for i in tqdm(range(N))]
        bs = np.random.uniform(low=2, high=4, size=N)
        tasks = [np.random.uniform(low=-bs[i], high=bs[i], size=size) for i in tqdm(range(N))]
        right_expected_value = [np.average(np.maximum(0, tasks[i])) for i in tqdm(range(N))]
        print(f'Right expected values: {right_expected_value}')

        # print(f'Tasks: {tasks}')
        sigmas = [[N - i - 1 for i in range(N)]]
        # BUBBLE SORT:
        last_sigma = sigmas[-1]
        for i in range(N - 1):
            for j in range(N - 1 - i):
                if right_expected_value[last_sigma[j]] > right_expected_value[last_sigma[j+1]]:
                    # print("0:", j)
                    # print("1:", last_sigma[j], last_sigma[j+1])
                    # print("2:", right_expected_value[last_sigma[j]], right_expected_value[last_sigma[j+1]])
                    new_sigma = copy.deepcopy(last_sigma)
                    new_sigma[j], new_sigma[j+1] = last_sigma[j+1], last_sigma[j]
                    # print("3:", new_sigma)
                    sigmas.append(new_sigma)
                    last_sigma = new_sigma
        print(f'Sigmas: {sigmas}')
        print(f'len = {len(sigmas)}')

        task_overlaps = []
        last_overlaps = []
        total_avg_overlaps = []
        for sigma in sigmas:
            new_sts = [np.zeros(shape=size)]
            task_overlaps_i = []
            for i in tqdm(range(N)):
                new_sts.append(np.maximum(0, np.add(new_sts[i], tasks[sigma[i]])))
                # print(f'New sts: {new_sts[-1]}')
                # print(f'Avg sts (sigma: {sigma} i: {i}): {np.average(new_sts[-1])}')
                task_overlaps_i.append(np.average(new_sts[-1]))
                # Define bin width
                bin_width = 0.01
                # bins = np.arange(0, 0.1, bin_width)  # Create bins with specified width
                # plt.hist(new_sts, bins=bins, edgecolor='black')
                # plt.xlabel(f'Sequence {sigmas[-1]}, i: {i}')
                # plt.ylabel('Frequency')
                # plt.title('new_sts')
                # plt.show()

            task_overlaps.append(task_overlaps_i)
            # print(f'last avg overlap for sigma {sigma} = {np.average(new_sts[-1])}')
            last_overlaps.append(np.average(new_sts[-1]))
            # print(f'avg of expected O for sigma {sigma} = {np.average(new_sts)}')
            total_avg_overlaps.append(np.average(new_sts))

        print(f'Task overlaps: {task_overlaps}')
        print(f'Last overlaps: {last_overlaps}')
        print(f'Total average overlaps: {total_avg_overlaps}')



        fig, ax = plt.subplots()
        ax.set_xlabel('iterations num')
        ax.set_ylabel('avg. overlap')
        ys = total_avg_overlaps
        xs = [x for x in range(len(ys))]

        plt.plot(xs, ys)
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_xlabel('iterations num')
        ax.set_ylabel('last. overlap')
        ys = last_overlaps
        xs = [x for x in range(len(ys))]

        plt.plot(xs, ys)
        plt.show()
        plt.close()
    # left_b = 1
    # right_b = 2
    #
    # left = np.random.exponential(scale=left_b, size=size)
    # # left[left < 0] /= 2
    # # bin_width = 0.01
    # # bins = np.arange(-left_b-1, left_b, bin_width)  # Create bins with specified width
    # # # Show the plot
    # # plt.hist(left, bins=bins, edgecolor='black')
    # # plt.show()
    # left_avg = np.average(left)
    # print(f'left_avg: {left_avg}')
    #
    # right = np.random.exponential(scale=right_b, size=size)
    # # right[right < 0] /= 2
    # # bin_width = 0.01
    # # bins = np.arange(-right_b - 1, right_b, bin_width)  # Create bins with specified width
    # # # Show the plot
    # # plt.hist(right, bins=bins, edgecolor='black')
    # # plt.show()
    # right_avg = np.average(right)
    # print(f'right_avg: {right_avg}')
    #
    # # print(f'right: {right}')
    # left_plus = np.maximum(left_avg, left)
    #
    #
    # right_plus = np.maximum(right_avg, right)
    #
    #
    # # print(f'left_plus: {left_plus}')
    # # print(f'right_plus: {right_plus}')
    # delta_1 = np.add(left_plus, right)
    # delta_2 = np.add(right_plus, left)
    # # print(f'delta_1: {delta_1}')
    # # print(f'delta_2: {delta_2}')
    # overlaps_1 = np.maximum(0, delta_1 - left_avg - right_avg)
    # # bin_width = 0.01
    # # bins = np.arange(0, left_b + right_b, bin_width)  # Create bins with specified width
    # # # Show the plot
    # # plt.hist(overlaps_1, bins=bins, edgecolor='black')
    # # plt.show()
    #
    # overlaps_2 = np.maximum(0, delta_2 - left_avg - right_avg)
    # # bin_width = 0.01
    # # bins = np.arange(0, left_b + right_b, bin_width)  # Create bins with specified width
    # # # Show the plot
    # # plt.hist(overlaps_2, bins=bins, edgecolor='black')
    # # plt.show()
    #
    # # print(f'overlaps_1: {overlaps_1}')
    # # print(f'overlaps_2: {overlaps_2}')
    # avg_ov_left = np.average(overlaps_1)
    # avg_ov_right = np.average(overlaps_2)
    #
    # print(f'Expected overlap 1: {avg_ov_left}\nExpected overlap 2: {avg_ov_right}')
    # if avg_ov_left < avg_ov_right:
    #     print(f'It is better to place left job with {left_b} FIRST')
    # else:
    #     print(f'It is better to place right job with {right_b} FIRST')

    # graph = PrecedenceGraph()
    # graph.random_network(20)
    #
    # alg = PGAlgorithms(graph)
    # print_networkx_graph(alg.make_networkx_graph())
    # print(is_graph_disjunctive(graph))
    #
    # write_graph(graph, "Data/PrecedenceGraphs/FasterGeneratedGraphs/gr.txt")
    # graph_new = read_graph("Data/PrecedenceGraphs/FasterGeneratedGraphs/gr.txt")
    # alg_1 = PGAlgorithms(graph_new)
    # print_networkx_graph(alg_1.make_networkx_graph())
    # print(is_graph_disjunctive(graph_new))
