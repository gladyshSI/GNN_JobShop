import csv
import networkx as nx
import matplotlib.pyplot as plt


def parce_PSPLib_graph(file_from: str, file_to: str) -> None:
    str_to_print = ""
    with open(file_from, 'r') as f:
        lines = f.readlines()
        state = 'search'
        for line in lines:
            if line == 'PRECEDENCE RELATIONS:\n':
                state = 'header'
                continue
            if state == 'header':
                state = 'data'
                continue
            if state == 'data' and line[0] == '*':
                break
            if state == 'data':
                data = line.split()
                if len(data) > 3:
                    fr_id = str(int(data[0]) - 1)
                    to_ids = [str(int(data[i]) - 1) for i in range(3, len(data))]
                    str_to_print += fr_id + ':' + ','.join(to_ids) + '\n'
    with open(file_to, 'w') as f:
        f.write(str_to_print)
        f.close()


def parce_csv(jobs_file: str, edges_file: str, edges_file_to: str) -> None:
    delimiter = ';'
    job_type_column: str = 'aircraft'
    job_id_column: str = 'id'
    job_dur_column: str = 'duration'
    fr_column: str = 'from_id'
    to_column: str = 'to_id'
    aircraft_id = '1'

    jobs_set = set()
    with open(jobs_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            if row[job_type_column] == aircraft_id:
                jobs_set.add(row[job_id_column])

    dg = nx.DiGraph()
    with open(edges_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            fr_id = row[fr_column]
            to_id = row[to_column]
            if fr_id in jobs_set and to_id in jobs_set:
                dg.add_edge(fr_id, to_id)

    num_nodes = len(dg.nodes)
    print(num_nodes)

    first_nodes = [n for n in list(dg.nodes) if dg.in_degree(n) == 0]
    last_nodes = [n for n in list(dg.nodes) if dg.out_degree(n) == 0]
    print(first_nodes)
    print(last_nodes)
    dg.add_edges_from([('first_dummy_node', fn) for fn in first_nodes])
    dg.add_edges_from([(ln, 'last_dummy_node') for ln in last_nodes])
    top_sort_list = list(nx.topological_sort(dg))
    print(top_sort_list)

    job_to_id = dict()
    id_to_job = dict()
    max_id = 0
    for node in top_sort_list:
        if node not in job_to_id.keys():
            job_to_id[node] = max_id
            id_to_job[max_id] = node
            max_id += 1

    print(job_to_id['last_dummy_node'])

    str_to_print = ""
    for id in range(max_id):
        fr_id = str(id)
        to_ids = [str(job_to_id[s]) for s in dg.successors(id_to_job[id])]
        if len(to_ids) == 0:
            continue
        str_to_print += fr_id + ':' + ','.join(to_ids) + '\n'

    with open(edges_file_to, 'w') as f:
        f.write(str_to_print)
        f.close()


if __name__ == '__main__':
    # Ns = [60, 120]
    # ks = [48, 60]
    # ds = [10, 10]
    # for round in range(len(Ns)):
    #     N = Ns[round]
    #     k = ks[round]
    #     d = ds[round]
    #     for i in range(1, k + 1):
    #         for j in range(1, d + 1):
    #             file_from = f'./Data/PSPLib/j{N}.sm/j{N}{i}_{j}.sm'
    #             file_to = f'./Data/PrecedenceGraphs/parsedPSPLib/graph_{N + 2}_{(i - 1) * d + j - 1}.txt'
    #             parce_PSPLib_graph(file_from, file_to)

    jobs_file = './Data/DataSets/LOVATO_taches.csv'
    edges_file = './Data/DataSets/LOVATO_precedences.csv'
    file_to = './Data/PrecedenceGraphs/LOVATO_gr_aircraft1.txt'
    parce_csv(jobs_file, edges_file, file_to)
