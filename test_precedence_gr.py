from precedence_graph import PrecedenceGraph, PGAlgorithms, print_networkx_graph

# Create Precedence Graph with 50 vertices
pg = PrecedenceGraph()
pg.random_v(50)
pg.random_network(start_num_diap=(2, 4), end_num_diap=(2, 4), seed=1)

# Create algorithms class for this graph
pga = PGAlgorithms(pg)
ranks = pga.ranking()
print("rank to vertices:\n", pga.get_rank_to_vs(ranks))

# Print statistics and draw precedence graph
print(pga.get_statistics())
print_networkx_graph(pga.make_networkx_graph())

# Save generated graph to the file
path_to_file = 'Data/PrecedenceGraphs/pg.txt'
pg.write_to_file(path_to_file)

# Read graph from file
pg_new = PrecedenceGraph()
pg_new.read_from_file(path_to_file)
pga_new = PGAlgorithms(pg_new)
print(pga_new.get_statistics())
print_networkx_graph(pga_new.make_networkx_graph())