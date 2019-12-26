import cplex
import numpy as np
import networkx as nx
from tqdm import tqdm
from datetime import datetime
import time
num_presicion = 1e-4

def greedy_coloring_heuristic(graph):
    '''
    Greedy graph coloring heuristic with degree order rule
    '''
    color_num = iter(range(0, len(graph)))
    color_map = {}
    used_colors = set()
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    color_map[nodes.pop(0)] = next(color_num)  # color node with color code
    used_colors = {i for i in color_map.values()}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in
                            list(filter(lambda x: x in color_map, graph.neighbors(node)))}
        if len(neighbors_colors) == len(used_colors):
            color = next(color_num)
            used_colors.add(color)
            color_map[node] = color
        else:
            color_map[node] = next(iter(used_colors - neighbors_colors))
    return len(used_colors), color_map

def greedy_coloring_heuristic_multiple(graph):
    '''
    Greedy graph coloring heuristic with degree order rule
    '''
    color_num, color_map = greedy_coloring_heuristic(graph)
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    color_map2 = {}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in
                            list(filter(lambda x: x in color_map, graph.neighbors(node)))}
        color_map2[node] = [i for i in range(0, color_num) if not i in list(neighbors_colors)]
    return color_num, color_map2


def read_networkx_graph(file_path):
    '''
        Parse .col file and return graph object
    '''
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges), int(vertices_num)

def get_neighbours_graph(graph, vertex):
    neighbours = [vertex]
    for neighbour in graph.neighbors(str(vertex)):
        neighbours.append(neighbour)
    return graph.subgraph(neighbours)

def get_nonneighbours_graph(graph, vertex):
    nonneighbours = [x for x in graph.nodes if not x in graph.neighbors(str(vertex)) and not x == vertex]
    return graph.subgraph(nonneighbours)

class Node():
    def __init__(self, parent_node, clique, graph,rows, rownames, rhs,senses):
        self.parent = parent_node
        self.clique = clique
        self.children = []
        self.graph = graph
        self.rows = rows
        self.rownames = rownames
        self.rhs = rhs
        self.senses = senses
        self.checked = False

    def solve(self, current_best_integer_solution):
        prob = cplex.Cplex()
        prob.set_results_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)
        prob.variables.add(obj = obj, names = colnames)
        prob.linear_constraints.add(lin_expr=self.rows, senses=self.senses,
                                    rhs=self.rhs, names=self.rownames)
        self.prob = prob
        prob.solve()
        solution = prob.solution.get_objective_value()
        values = np.array(prob.solution.get_values())
        self.ub = solution
        if solution <= current_best_integer_solution:
            print('Bad branch')
            return 0, 0
        self.color_num, self.color_map = greedy_coloring_heuristic_multiple(self.graph)
        color_counter = 0
        violation = np.zeros(self.color_num)
        for i in range(self.color_num):
            vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
            violation[i] = np.sum(values[vertexes])
        add_row = []
        if np.max(violation) > 1:
            for i in range(len(violation)):
                if violation[i] != max(violation):
                    continue
                tmp = np.zeros(n)
                vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
                tmp[vertexes] = 1
                add_row.append([colnames, tmp.tolist()])
            self.children.append(Node(self, self.clique, self.graph, self.rows + add_row, self.rownames + ['c'+str(len(self.rownames) + i + 1) for i in range(len(add_row))],
                            self.rhs + [1]*len(add_row),self.senses+'L'*len(add_row)))
        else:
            self.check_clique(values)
        return solution, values

    def check_clique(self, values):
        nodes = np.argwhere(values==1).squeeze()+1
        new_graph = self.graph.copy()
        for node in nodes:
            if str(int(node)) in new_graph:
                new_graph = get_neighbours_graph(new_graph, str(int(node)))
        if len(new_graph.nodes) == len(self.graph.nodes):
            self.checked = True
        else:
            for i in np.argwhere(values==1).squeeze():
                tmp = np.zeros(n)
                tmp[i] = 1
                pruned_graph = get_neighbours_graph(self.graph, str(i+1))
                self.children.append(Node(self, self.clique + [str(i+1)], pruned_graph, self.rows + [[colnames, tmp.tolist()]], self.rownames + ['c'+str(len(self.rownames) + 1)],
                                self.rhs + [1],self.senses+'G'))
        return len(new_graph.nodes) == len(self.graph.nodes)

if __name__ == '__main__':
    original_graph, n = read_networkx_graph('../DIMACS_all_ascii/MANN_a9.clq')
    obj = np.ones(n)
    colnames = [x for x in original_graph.nodes]
    rhs = np.ones(n).tolist()
    senses = 'L' * n
    rownames = ['b'+str(i+1) for i in range(n)]
    rows = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        rows.append([colnames, a.tolist()])

    parent_node = Node(None,[], original_graph, rows, rownames, rhs, senses)
    node = parent_node
    node.solve(0)
    # parent_node.solve(0)
    current_best_integer_solution = 0
    first_layer_size = len(parent_node.children)
    max_tree_depth = 0
    current_depth_position = 0
    current_progress = 1 - len(parent_node.children)/first_layer_size
    print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)
    i = 0
    while len(parent_node.children) != 0:
        while(len(node.children) != 0): #go down one branch
            node = node.children[0]
            current_depth_position+=1
            solution, values = node.solve(current_best_integer_solution)
            if len(node.children) == 0 and not node.checked:
                raise AssertionError('Not correct result!')
        exit()
        if current_depth_position > max_tree_depth:
            max_tree_depth = current_depth_position
        if solution > current_best_integer_solution:
            current_best_integer_solution = solution
            current_best_values = values
        while (node.ub <= current_best_integer_solution or len(node.children) == 0) and node != parent_node:
            node.children = []
            node = node.parent
            current_depth_position-=1
        node.children.pop(0)

        while len(node.children) == 0 and node!=parent_node: # return up the branch until a good possibility exists
            node = node.parent
            current_depth_position-=1
            node.children.pop(0) # have just been there

        if 1 - len(parent_node.children)/first_layer_size != current_progress:
            current_progress = 1 - len(parent_node.children)/first_layer_size
            print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)
            max_tree_depth = 0

        if current_depth_position == 0 and len(node.children)==0:
            print('The search is finished!')
            print('Final solution: ', current_best_integer_solution)
            print('Optimal values: ', dict(zip(colnames, current_best_values)))
            exit()
