from DiscreteOpt.run_opt_models import find_best_transition_time

print('Hello World!')
# look in training_experiments.py

from class_graph import PrecedenceGraph
from class_graph_algs import PGAlgorithms, print_networkx_graph, is_graph_disjunctive
from writers_readers import *

if __name__ == '__main__':
    a = {1: 3, 2: 2, 3: 1}
    b = [1, 2, 3]
    f = lambda l: min(l, key=lambda x: a[x])
    print(f(b))

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
