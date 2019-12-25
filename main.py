import cplex
import numpy as np
from numba import *
from tqdm import tqdm
from datetime import datetime
import time
import networkx as nx
# import utils_for_cplex as utils
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
            # if line.startswith('c'):  # graph description
            #     print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            if line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
                # print('{0} {1} {2}'.format(name, vertices_num, edges_num))
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

class Node():
    def __init__(self, parent, original_graph, clique, ub,graph, prob, fix_vertex, add_constraints):
        self.original_graph = original_graph
        self.n = np.max(np.array([x for x in original_graph.nodes], dtype = int))
        self.parent = parent
        self.ub = ub
        self.graph = graph
        self.fix_vertex = fix_vertex
        self.add_row = add_constraints
        self.clique = clique
        self.children = []
        self.prob = prob

    def solve(self, current_best_integer_solution):
        colnames = [x for x in self.original_graph.nodes]
        if self.fix_vertex:
            self.clique.append(self.fix_vertex)
            new_graph = get_neighbours_graph(self.graph, str(int(self.fix_vertex)))
            pruned_vertexes = [x for x in new_graph.nodes if not x in self.graph.nodes]
            self.graph = new_graph
            rhs = np.zeros(len(pruned_vertexes)).tolist()
            self.add_row = []
            for i in pruned_vertexes:
                tmp = np.zeros(len(colnames))
                tmp[int(i)-1] = 1
                self.add_row.append([colnames, tmp.tolist()])
            self.prob.linear_constraints.add(lin_expr=self.add_row, senses='L'*len(self.add_row), rhs=[0]*len(self.add_row),
                                    names=['c'+str(self.prob.linear_constraints.get_num()+i+1) for i in range(len(self.add_row))])
            add_row2 = []
            tmp = np.zeros(self.n)
            tmp[int(self.fix_vertex)-1] = 1
            add_row2.append([colnames, tmp.tolist()])
            self.prob.linear_constraints.add(lin_expr=add_row2, senses='G'*len(add_row2), rhs=[1]*len(add_row2),
                                    names=['c'+str(self.prob.linear_constraints.get_num()+i+1) for i in range(len(add_row2))])

        else:
            self.prob.linear_constraints.add(lin_expr=self.add_row, senses='L'*len(self.add_row), rhs=[1]*len(self.add_row),
                                    names=['c'+str(self.prob.linear_constraints.get_num()+i+1) for i in range(len(self.add_row))])
        # exit()
        self.prob.solve()
        solution = self.prob.solution.get_objective_value()
        values = np.array(self.prob.solution.get_values())
        self.color_num, self.color_map = greedy_coloring_heuristic_multiple(self.graph)
        color_counter = 0
        violation = np.zeros(self.color_num)
        for i in range(self.color_num):
            vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
            violation[i] = np.sum(values[vertexes])
        # print('v',violation, '\n', self.color_map, values)
        if np.max(violation) == 1:
            if self.check_clique(np.argwhere(values==1).squeeze()+1):
                print('Found better', solution)
                return solution, values
            # print(values, self.clique)
            for i in range(len(values)):

                if abs(values[i]) < num_presicion or str(i+1) in self.clique or not str(i+1) in self.graph.nodes:## WARNING: last is not ok
                    continue
                else:
                    self.children.append(Node(self, self.original_graph, self.clique, self.ub, self.graph, self.prob, str(i+1), []))
        add_row = []
        for i in range(len(violation)):
            if violation[i] != max(violation):
                continue
            tmp = np.zeros(self.n)
            vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
            tmp[vertexes] = 1
            add_row.append([colnames, tmp.tolist()])
        # print('v', violation, '\n',values,'\n',self.ub)
        if solution <= self.ub or solution <= current_best_integer_solution:
            return 0, 0
        # self.ub = solution
        # for fix_vertex in np.array([str(x) for x in np.sort(np.array(self.graph.nodes).astype(int))])[values==1]:
        if not self.fix_vertex:
            self.children.append(Node(self,self.original_graph, self.clique, self.ub, self.graph, self.prob, None, add_row))
        return solution, values

    def check_clique(self, values):
        new_graph = self.graph.copy()
        for node in values:
            if str(int(node)) in new_graph:
                new_graph = get_neighbours_graph(new_graph, str(int(node)))
        return len(new_graph.nodes) == len(self.graph.nodes)

def main():
    graph, n = read_networkx_graph('../DIMACS_all_ascii/MANN_a9.clq')
    n = len(graph.nodes)
    obj = np.ones(n)
    colnames = [x for x in graph.nodes]
    rhs = np.ones(n).tolist()
    senses = 'L' * n
    rownames = ['b'+str(i+1) for i in range(n)]
    rows = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        rows.append([colnames, a.tolist()])
    prob = cplex.Cplex()
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)
    prob.variables.add(obj=obj, names=colnames)
    prob.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rownames)
    parent_node = Node(None, graph, [], 0, graph, prob, None, [])
    node = parent_node
    # parent_node.solve(0)
    current_best_integer_solution = 0
    max_possible_solution = 0
    solution, values = parent_node.solve(current_best_integer_solution)
    max_possible_solution = solution
    first_layer_size = len(parent_node.children)
    max_tree_depth = 0
    current_depth_position = 0
    current_progress = 1 - len(parent_node.children)/first_layer_size
    print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)
    i = 0
    values_prev = values_prev_prev = np.zeros(n)
    while len(parent_node.children) != 0:
        while(len(node.children) != 0): #go down one branch
            node = node.children[0]
            current_depth_position+=1
            solution, values = node.solve(current_best_integer_solution)
            # print(node.clique)
            # if np.sum(abs(values - values_prev))==0 and np.sum(abs(values_prev_prev-values)) == 0:
            #     raise AssertionError('Not updating!')
            values_prev_prev = values_prev
            values_prev = values
            # print(values)
            # i+=1
            # node.prob.write('./problems/'+str(i)+'.lp')
            # if i==21:
            #     exit()
        # exit()
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
if __name__ == '__main__':
    main()
