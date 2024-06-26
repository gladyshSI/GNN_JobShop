import plotly.graph_objects as go
import networkx as nx
import numpy as np
from collections import deque
import random
from statistics import mean


class Task:
    def __init__(self, id, duration, d_min, d_max):
        self._id = id
        self._duration = duration
        self._d_min = d_min
        self._d_max = d_max


class PrecedenceGraph:
    def __init__(self):
        self._vertices = []  # [Task]
        self._edges = dict()  # t_id -> [successor_ids]
        self._reverse_edges = dict()  # t_id -> [predecessor_ids]

    def get_duration(self, task_id):
        return self._vertices[task_id]._duration

    def random_v(self, v_num, ds_min=None, ds_max=None, seed=1):
        if ds_max is None:
            ds_max = []
        if ds_min is None:
            ds_min = []
        np.random.seed(seed=seed)
        if len(ds_min) < v_num:
            ds_min = [3 for _ in range(v_num)]
        if len(ds_max) < v_num:
            ds_max = [10 for _ in range(v_num)]

        for i in range(v_num):
            duration = np.round(np.random.uniform(ds_min[i], ds_max[i])).astype(int)
            self._vertices.append(Task(i, duration, ds_min[i], ds_max[i]))

    def random_network(self, start_num_diap=(3, 5), end_num_diap=(3, 5), seed=1):
        random.seed = seed
        v_num = len(self._vertices)
        start_num = np.round(np.random.uniform(start_num_diap[0], start_num_diap[1])).astype(int)
        end_num = min(v_num - 2 - start_num,
                      np.round(np.random.uniform(end_num_diap[0], end_num_diap[1])).astype(int))

        # Step 1: Connect two dummy tasks with start and end tasks
        dummy_st_id = 0
        self._edges[dummy_st_id] = []
        for i in range(start_num):
            to_id = i + 1
            self._edges[dummy_st_id].append(to_id)
            self._reverse_edges[to_id] = [dummy_st_id]

        dummy_end_id = v_num - 1
        self._reverse_edges[dummy_end_id] = []
        for i in range(end_num):
            fr_id = v_num - 2 - i
            self._edges[fr_id] = [dummy_end_id]
            self._reverse_edges[dummy_end_id].append(fr_id)
        print("Step 1: edges: ", self._edges)
        print("Step 1: rev:   ", self._reverse_edges)

        # Step 2: Find random predecessor:
        predecessors = list(range(1, start_num + 1))
        for to_id in list(range(start_num + 1, dummy_end_id)):
            fr_id = random.choice(predecessors)
            if fr_id not in self._edges.keys():
                self._edges[fr_id] = []
            self._edges[fr_id].append(to_id)
            self._reverse_edges[to_id] = [fr_id]
            # end tasks can't be predecessors
            if to_id < dummy_end_id - end_num:
                predecessors.append(to_id)
        print("Step 2: edges: ", self._edges)
        print("Step 2: rev:   ", self._reverse_edges)

        # Step 3: Find random successor:
        no_out_ids = [i for i in range(dummy_end_id)
                      if i not in self._edges.keys()]
        for fr_id in no_out_ids:
            predss = self.get_all_predecessors(fr_id)
            forbidden_set = set()
            for pred_id in predss:
                forbidden_set.update(self._edges[pred_id])

            to_id_set = [i for i in range(fr_id + 1, dummy_end_id) if i not in forbidden_set]
            if len(to_id_set) == 0:
                to_id_set = [dummy_end_id]
            to_id = random.choice(to_id_set)
            self._edges[fr_id] = [to_id]
            self._reverse_edges[to_id].append(fr_id)
        print("Step 3: edges: ", self._edges)
        print("Step 3: rev:   ", self._reverse_edges)

    def get_all_successors(self, v_id):
        successors = set()
        q = deque({v_id})
        while q:
            id = q.pop()
            for next in [] if id not in self._edges.keys() else self._edges[id]:
                if next not in successors:
                    successors.add(next)
                    q.append(next)
        return list(successors)

    def get_all_predecessors(self, v_id):
        predecessors = set()
        q = deque({v_id})
        while q:
            id = q.pop()
            for next in [] if id not in self._reverse_edges.keys() else self._reverse_edges[id]:
                if next not in predecessors:
                    predecessors.add(next)
                    q.append(next)
        return list(predecessors)

    def random_tree(self, avg_out_deg=2, out_degs=[], seed=1):
        np.random.seed(seed=seed)
        v_num = len(self._vertices)
        if sum(out_degs) != v_num - 1:
            # create random out_degs:
            out_deg_sum_left = v_num - 1
            out_degs = []
            while out_deg_sum_left > 0:
                deg = min(out_deg_sum_left, np.ceil(np.random.exponential(2)).astype(int))
                out_degs.append(deg)
                out_deg_sum_left -= deg

        print(out_degs, sum(out_degs) / len(self._vertices))
        max_id = 1
        for id in range(len(out_degs)):
            id_out = out_degs[id]
            self._edges[id] = list(range(max_id, max_id + id_out, 1))
            max_id += id_out

    def add_edge(self, fr_id, to_id):
        if fr_id not in self._edges.keys():
            self._edges[fr_id] = []
        self._edges[fr_id].append(to_id)
        if to_id not in self._reverse_edges.keys():
            self._reverse_edges[to_id] = []
        self._reverse_edges[to_id].append(fr_id)

    def write_to_file(self, path_to_file):
        with open(path_to_file, 'w') as f:
            f.write(str(len(self._vertices)) + '\n')
            for v in self._vertices:
                f.write(str(v._id) + ',' +
                        str(v._duration) + ',' +
                        str(v._d_min) + ',' +
                        str(v._d_max) + '\n')
            f.write(str(len(self._edges)) + '\n')
            for fr, to_ids in self._edges.items():
                f.write(str(fr) + ',')
                for i in range(len(to_ids)):
                    f.write(str(to_ids[i]))
                    if i != len(to_ids) - 1:
                        f.write(',')
                f.write('\n')

    def read_from_file(self, path_to_file):
        self._vertices = []  # [Task]
        self._edges = dict()  # t_id -> [successor_ids]
        self._reverse_edges = dict()  # t_id -> [predecessor_ids]

        with open(path_to_file, 'r') as f:
            line_id = 0
            v_num = 0
            e_num = 0
            for line in f:
                if line_id == 0:
                    v_num = int(line.rstrip())
                elif line_id <= v_num:
                    id, duration, d_min, d_max = line.split(',')
                    self._vertices.append(Task(int(id), int(duration), int(d_min), int(d_max)))
                elif line_id == v_num + 1:
                    e_num = int(line.rstrip())
                else:
                    ids = line.split(',')
                    fr = int(ids[0])
                    to_ids = [int(to_id) for to_id in ids[1:]]
                    for to_id in to_ids:
                        self.add_edge(fr, to_id)
                line_id += 1


class PGAlgorithms:
    def __init__(self, precedence_gr):
        self._pg = precedence_gr

    def get_statistics(self):
        stat = dict()
        stat['v_num'] = len(self._pg._vertices)
        stat['e_num'] = sum([len(e_out_list) for e_out_list in self._pg._edges.values()])
        stat['in_avg'] = mean([len(e_out_list) for e_out_list in self._pg._edges.values()])
        stat['out_avg'] = mean([len(e_in_list) for e_in_list in self._pg._reverse_edges.values()])
        stat['out_max'] = max([(v, len(e_out_list)) for v, e_out_list in self._pg._edges.items()], key=lambda x: x[1])
        stat['in_max'] = max([(v, len(e_in_list)) for v, e_in_list in self._pg._reverse_edges.items()],
                             key=lambda x: x[1])
        return stat

    def get_first_vertices(self):
        v_set = set()
        for v in self._pg._vertices:
            v_set.add(v._id)
        for _, to in self._pg._edges.items():
            for v in to:
                if v in v_set:
                    v_set.remove(v)
        return list(v_set)

    def get_last_vertices(self):
        v_set = set()
        for v in self._pg._vertices:
            v_set.add(v._id)
        for _, to in self._pg._reverse_edges.items():
            for v in to:
                if v in v_set:
                    v_set.remove(v)
        return list(v_set)

    def ranking(self):
        ranks = dict()
        first_vs = self.get_first_vertices()
        q = deque()
        for v in first_vs:
            q.append((v, 0))
            ranks[v] = 0
        while q:
            v, rank = q.pop()
            if v in self._pg._edges.keys():
                for next_v in self._pg._edges[v]:
                    if (next_v not in ranks.keys()) or (rank + 1 > ranks[next_v]):
                        q.append((next_v, rank + 1))
                        ranks[next_v] = rank + 1
        return ranks

    def get_rank_to_vs(self, ranks):
        rank_to_vs = dict()
        max_rank = 0
        for v, rank in ranks.items():
            max_rank = max(max_rank, rank)
            if rank not in rank_to_vs.keys():
                rank_to_vs[rank] = []
            rank_to_vs[rank].append(v)
        return rank_to_vs

    def left_longest_passes(self):
        llps = dict()
        # wasinq = set()
        first_vs = self.get_first_vertices()
        q = deque()
        for v in first_vs:
            q.appendleft(v)
            # wasinq.add(v)
        while q:
            v = q.pop()
            maxleft = 0 if v not in llps.keys() else llps[v]
            ps = [] if v not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[v]
            for pred in ps:
                llps_res = 0 if pred not in llps.keys() else llps[pred]
                maxleft = max(maxleft, llps_res + self._pg.get_duration(pred))
            llps[v] = maxleft

            ss = [] if v not in self._pg._edges.keys() else self._pg._edges[v]
            for s in ss:
                q.appendleft(s)
        return llps

    def right_longest_passes(self):
        rlps = dict()
        last_vs = self.get_last_vertices()
        q = deque()
        for v in last_vs:
            q.appendleft(v)
        while q:
            v = q.pop()
            v_dur = self._pg.get_duration(v)
            maxright = v_dur if v not in rlps.keys() else rlps[v]
            ss = [] if v not in self._pg._edges.keys() else self._pg._edges[v]
            for s in ss:
                rlps_res = 0 if s not in rlps.keys() else rlps[s]
                maxright = max(maxright, rlps_res + v_dur)
            rlps[v] = maxright

            ps = [] if v not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[v]
            for p in ps:
                q.appendleft(p)
        return rlps

    def get_longest_passes(self):
        llps = self.left_longest_passes()
        rlps = self.right_longest_passes()
        lps = {id: (llps[id], rlps[id]) for id in list(range(len(self._pg._vertices)))}
        return lps

    def positioning(self, x_coef, y_coef):
        positions = dict()
        ranks = self.ranking()
        rank_to_vs = self.get_rank_to_vs(ranks)

        for rank_id in range(len(rank_to_vs)):
            rank_len = len(rank_to_vs[rank_id])
            prev_y_avgs = dict()
            for v in rank_to_vs[rank_id]:
                ps = [] if v not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[v]
                prev_y_sum = sum([positions[prev_v][1] for prev_v in ps])
                prev_y_avg = 0 if len(ps) == 0 else prev_y_sum / len(ps)
                prev_y_avgs[v] = prev_y_avg
            vs = sorted(rank_to_vs[rank_id], key=lambda x: prev_y_avgs[x])
            y_num = 0
            for v in vs:
                positions[v] = (rank_id * x_coef, y_num * y_coef + np.random.uniform(-y_coef / 8, y_coef / 8))
                y_num += 1
        return positions

    def make_networkx_graph(self, x_coef=40, y_coef=10):
        G = nx.DiGraph()
        for v in self._pg._vertices:
            G.add_node(v._id)
        for l, rights in self._pg._edges.items():
            for r in rights:
                G.add_edge(l, r)
        positions = self.positioning(x_coef, y_coef)
        for node in G.nodes():
            G.nodes[node]['pos'] = positions[node]
        return G


def print_networkx_graph(G, colors_dict={}):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='RdBu',  # 'YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_colors = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        color = len(adjacencies[1])
        node_adjacencies.append(color)
        t = adjacencies[0]
        if len(colors_dict) == len(G.nodes()):
            color = colors_dict[t]
            node_colors.append(color)
        t_name = str(t)
        successors = ','.join(str(x) for x in adjacencies[1])
        node_text.append('task: ' + t_name + ' -> ' + successors + ' color: ' + str(color))

    node_trace.marker.color = node_adjacencies if len(node_colors) == 0 else node_colors
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()
