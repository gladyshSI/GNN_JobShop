import networkx as nx
from numpy import mean
import plotly.graph_objects as go
from class_graph import *


# TODO: Doesn't work correctly
def is_redundancy_check(edges: dict, start_ids: list) -> bool:
    # Check is there such vertex:
    if all(v_id not in edges.keys() for v_id in start_ids):
        print("All vertices are unknown")
        return False

    """
    DFS with the colors:
    0 -> We added this vertex to the steck
        if we want to go to the vertex with the color 0 => there is a redundant edge
    1 -> We opened the vertex : starting process its successors
        if we want to go to the vertex with color 1 => there is a cycle
    2 -> We've closed this vertex : have processed all its successors, but there is an opened predecessor.
        if we want to go to the vertex with color 2 => there is a redundant edge
    3 -> The vertex that opened it is closed (color 2 or 3)
        We can want to go to the vertex with color 3 and changed it color to 2.
    """
    color = {start_id: 0 for start_id in start_ids}
    q = deque(start_ids)
    while q:
        v = q[-1]
        # Open vertex:
        if color[v] == 0:
            color[v] = 1
            for next_v in [] if v not in edges.keys() else edges[v]:
                if next_v not in color.keys():
                    color[next_v] = 0
                    q.append(next_v)
                elif color[next_v] < 3:
                    print("Redundancy detected!")
                    return True
                elif color[next_v] == 3:
                    # We need to check this input edge
                    color[next_v] = 2
        # Close vertex
        elif color[v] == 1:
            color[v] = 2
            # Color the right points of the checked edges:
            for next_v in [] if v not in edges.keys() else edges[v]:
                if next_v in color.keys() and color[next_v] == 2:
                    color[next_v] = 3
            q.pop()
        # Case 3: we do not expect other cases
        else:
            raise ValueError("Some error in is_redundancy_check algorithm. We don't expect color != (0 or 1) in steck")
    return False


# TODO: Doesn't work correctly
def is_graph_disjunctive(graph: PrecedenceGraph) -> bool:
    edges = graph.get_copy_of_all_edges()
    start_vertices = graph.get_start_ids()

    # Check that there is no redundant edges:
    is_redundant = is_redundancy_check(edges, list(start_vertices))
    return not is_redundant


# TODO: extract and make functions instead of class. Do this class is really necessary?
def get_rank_to_vs(ranks):
    rank_to_vs = dict()
    max_rank = 0
    for v, rank in ranks.items():
        max_rank = max(max_rank, rank)
        if rank not in rank_to_vs.keys():
            rank_to_vs[rank] = []
        rank_to_vs[rank].append(v)
    return rank_to_vs


class PGAlgorithms:
    def __init__(self, precedence_gr):
        self._pg = copy.deepcopy(precedence_gr)
        self._vertices = self._pg.get_all_ids()

    def get_statistics(self):
        stat = dict()
        stat['v_num'] = len(self._vertices)
        stat['e_num'] = sum([len(e_out_list) for e_out_list in self._pg._edges.values()])
        stat['in_avg'] = mean([len(e_out_list) for e_out_list in self._pg._edges.values()])
        stat['out_avg'] = mean([len(e_in_list) for e_in_list in self._pg._reverse_edges.values()])
        stat['out_max'] = max([(v, len(e_out_list)) for v, e_out_list in self._pg._edges.items()], key=lambda x: x[1])
        stat['in_max'] = max([(v, len(e_in_list)) for v, e_in_list in self._pg._reverse_edges.items()],
                             key=lambda x: x[1])
        return stat

    def ranking(self):
        ranks = dict()
        first_vs = self._pg.get_start_ids()
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

    # # TODO: Transfer to class_problem
    # def left_longest_passes(self):
    #     llps = dict()
    #     # wasinq = set()
    #     first_vs = self._pg.get_start_ids()
    #     q = deque()
    #     for v in first_vs:
    #         q.appendleft(v)
    #         # wasinq.add(v)
    #     while q:
    #         v = q.pop()
    #         maxleft = 0 if v not in llps.keys() else llps[v]
    #         ps = [] if v not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[v]
    #         for pred in ps:
    #             llps_res = 0 if pred not in llps.keys() else llps[pred]
    #             maxleft = max(maxleft, llps_res + self._pg.get_duration(pred))
    #         llps[v] = maxleft
    #
    #         ss = [] if v not in self._pg._edges.keys() else self._pg._edges[v]
    #         for s in ss:
    #             q.appendleft(s)
    #     return llps
    #
    # # TODO: Transfer to class_problem
    # def right_longest_passes(self):
    #     rlps = dict()
    #     last_vs = self._pg.get_end_ids()
    #     q = deque()
    #     for v in last_vs:
    #         q.appendleft(v)
    #     while q:
    #         v = q.pop()
    #         v_dur = self._pg.get_duration(v)
    #         maxright = v_dur if v not in rlps.keys() else rlps[v]
    #         ss = [] if v not in self._pg._edges.keys() else self._pg._edges[v]
    #         for s in ss:
    #             rlps_res = 0 if s not in rlps.keys() else rlps[s]
    #             maxright = max(maxright, rlps_res + v_dur)
    #         rlps[v] = maxright
    #
    #         ps = [] if v not in self._pg._reverse_edges.keys() else self._pg._reverse_edges[v]
    #         for p in ps:
    #             q.appendleft(p)
    #     return rlps

    # # TODO: Transfer to class_problem
    # def get_longest_passes(self):
    #     llps = self.left_longest_passes()
    #     rlps = self.right_longest_passes()
    #     lps = {id: (llps[id], rlps[id]) for id in list(range(len(self._vertices)))}
    #     return lps

    def positioning(self, x_coef, y_coef):
        positions = dict()
        ranks = self.ranking()
        rank_to_vs = get_rank_to_vs(ranks)

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
        for v in self._vertices:
            G.add_node(v)
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
