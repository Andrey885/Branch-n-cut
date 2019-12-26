import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages/')
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
    if len(graph.nodes) == 0:
        return 0, {}
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
    def __init__(self, parent_node, clique,nonclique, graph,rows, rownames, rhs,senses, bnb_mode):
        self.parent = parent_node
        self.clique = clique
        self.nonclique = nonclique
        self.children = []
        self.graph = graph
        self.rows = rows
        self.rownames = rownames
        self.rhs = rhs
        self.senses = senses
        self.bnb_mode = bnb_mode

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
        if solution <= current_best_integer_solution or len(self.graph.nodes)==0:
            self.ub = 0
            return 0, 0
        # print(self.graph.nodes)
        self.color_num, self.color_map = greedy_coloring_heuristic(self.graph)
        self.ub = min(solution, self.color_num)
        if self.ub <= current_best_integer_solution:
            return 0, 0
        color_counter = 0
        violation = np.zeros(self.color_num)
        color_map_augmented = []
        for i in range(self.color_num):
            # vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i == v])
            vertexes = [k for k, v in self.color_map.items() if v == i]
            nodes = [x for x in self.graph.nodes]
            for vertex in vertexes:
                for neighbor in self.graph.neighbors(vertex):
                    if neighbor in nodes:
                        nodes.remove(neighbor)
            vertexes = nodes
            vertexes = np.array([int(v)-1 for v in vertexes])
            color_map_augmented.append(vertexes)
            violation[i] = np.sum(values[vertexes])
        color_map_augmented = np.array(color_map_augmented)
        # print(values, self.clique, self.nonclique)
        if np.max(violation) > 1:
            for vertexes in color_map_augmented[violation > 1]:
                add_row = []
                tmp = np.zeros(n)
                tmp[vertexes] = 1
                add_row.append([colnames, tmp.tolist()])
                self.children.append(Node(self, self.clique, self.nonclique, self.graph, self.rows + add_row, self.rownames + ['c'+str(len(self.rownames) + i + 1) for i in range(len(add_row))],
                            self.rhs + [1]*len(add_row),self.senses+'L'*len(add_row), bnb_mode = self.bnb_mode))
        # elif not self.bnb_mode:
        #     add_row = self.add_exponential_constraints()
        #     self.children.append(Node(self, self.clique, self.nonclique, self.graph, self.rows + add_row, self.rownames + ['c'+str(len(self.rownames) + i + 1) for i in range(len(add_row))],
        #                     self.rhs + [1]*len(add_row),self.senses+'L'*len(add_row), bnb_mode = self.bnb_mode))
        elif np.sum(abs(values - values.astype(int))) <  num_presicion*len(values):
            for i in np.argwhere(values==1).squeeze():
                if str(i+1) in self.clique:
                    continue
                if not str(i+1) in self.graph and not str(i+1) in self.nonclique:
                    tmp = np.zeros(n)
                    tmp[i] = 1
                    self.children.append(Node(self, self.clique, self.nonclique+[str(i+1)],self.graph, self.rows + [[colnames, tmp]],
                                    self.rownames + ['c'+str(len(self.rownames) + 1)],
                                    self.rhs + [0],self.senses+'L', bnb_mode = self.bnb_mode))
                    continue
                tmp = np.zeros(n)
                tmp[i] = 1
                self.children.append(Node(self, self.clique+[str(i+1)], self.nonclique, get_neighbours_graph(self.graph, str(i+1)), self.rows + [[colnames, tmp]],
                                self.rownames + ['c'+str(len(self.rownames) + 1)],
                                self.rhs + [1],self.senses+'G', bnb_mode = self.bnb_mode))
        elif np.sum(abs(values - values.astype(int))) > num_presicion*len(values):
            for i in range(len(values)):
                if abs(values[i] - int(values[i])) > num_presicion:
                    tmp = np.zeros(n)
                    tmp[i] = 1
                    if not str(i+1) in self.clique and str(i+1) in self.graph.nodes:
                        self.children.append(Node(self, self.clique+[str(i+1)], self.nonclique, get_neighbours_graph(self.graph, str(i+1)), self.rows + [[colnames, tmp]],
                                        self.rownames + ['c'+str(len(self.rownames) + 1)],
                                        self.rhs + [1],self.senses+'G', bnb_mode = self.bnb_mode))
                    if not str(i+1) in self.nonclique and str(i+1) in self.graph.nodes:
                        self.children.append(Node(self, self.clique, self.nonclique+[str(i+1)], get_nonneighbours_graph(self.graph, str(i+1)), self.rows + [[colnames, tmp]],
                                        self.rownames + ['c'+str(len(self.rownames) + 1)],
                                        self.rhs + [0],self.senses+'L', bnb_mode = self.bnb_mode))
        # else:
            # if self.check_clique(values):
                # self.clique = [str(int(i+1)) for i in np.argwhere(values == 1).squeeze()]
            # else:
            #     print(self.bnb_mode)
            #     raise AssertionError()
        return len(self.clique), values

    def check_clique(self, values):
        clique = original_graph.subgraph([str(i+1) for i in np.argwhere(values==1).squeeze()])
        for node in clique:
            clique = get_neighbours_graph(clique, str(int(node)))
        if len(clique.nodes) == len(values[values==1]):
            self.checked = True
        return self.checked

    def add_exponential_constraints(self):
        complement = nx.complement(original_graph)
        self.bnb_mode = True
        add_row = []
        for edge in complement.edges:
            tmp = np.zeros(n)
            tmp[int(edge[0]) - 1] = 1
            tmp[int(edge[1]) - 1] = 1
            add_row.append([colnames, tmp])
        return add_row

if __name__ == '__main__':
    # original_graph, n = read_networkx_graph('../DIMACS_all_ascii/playground.clq')
    # original_graph, n = read_networkx_graph('../DIMACS_all_ascii/johnson8-2-4.clq')
    # original_graph, n = read_networkx_graph('../DIMACS_all_ascii/MANN_a9.clq')
    original_graph, n = read_networkx_graph('../DIMACS_all_ascii/hamming6-2.clq')
    # original_graph, n = read_networkx_graph('../DIMACS_all_ascii/brock200_2.clq')
    # original_graph, n = read_networkx_graph('../DIMACS_all_ascii/C125.9.clq')
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

    parent_node = Node(None,[],[], original_graph, rows, rownames, rhs, senses, bnb_mode = False)
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
            # print(current_depth_position, len(node.children), node.bnb_mode, node.ub)
        if current_depth_position > max_tree_depth:
            max_tree_depth = current_depth_position
        if len(node.clique) > current_best_integer_solution:
            print('Found better', len(node.clique))
            current_best_integer_solution = len(node.clique)
            best_clique = node.clique
            current_best_values = values
        while (node.ub <= current_best_integer_solution or len(node.children) == 0) and node != parent_node:
            node.children = []
            node = node.parent
            current_depth_position-=1
        if node == parent_node and len(node.children)!= 0:
            for child in node.children:
                if len(node.children[0].clique) != 0:
                    child.nonclique.append(node.children[0].clique[0])
                    if node.children[0].clique[0] in child.graph.nodes:
                        child.graph = get_nonneighbours_graph(child.graph, node.children[0].clique[0])
                    tmp = np.zeros(n)
                    tmp[int(node.children[0].clique[0])-1] = 1
                    child.rows += [[colnames, tmp]]
                    child.rownames += ['c'+str(len(child.rownames) + 1)]
                    child.rhs += [0]
                    child.senses+='L'
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
            print('Best clique: ', best_clique)
            exit()
