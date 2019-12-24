import cplex
import numpy as np
from numba import *
from tqdm import tqdm
from datetime import datetime
import time
import networkx as nx
import utils_for_cplex as utils

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

class Node():
    def __init__(self, parent, ub,graph, prob, color_map,color_num, values, colnames):
        self.parent = parent
        self.ub = ub
        self.graph = graph
        self.prob = prob
        self.color_map = color_map
        self.color_num = color_num
        self.prev_values = values
        self.colnames = colnames
        self.children = []

    def solve(self, current_best_integer_solution):
        color_counter = 0
        violation = np.zeros(self.color_num)
        for i in range(color_num):
            vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
            violation[i] = np.sum(self.prev_values[vertexes])
        if np.max(violation) == 1:
            print('Found solution', np.sum(violation))
            return np.sum(violation), self.prev_values
        add_row = []
        for i in range(len(violation)):
            if violation[i] != max(violation):
                continue
            tmp = np.zeros(n)
            vertexes = np.array([int(k)-1 for k, v in self.color_map.items() if i in v])
            tmp[vertexes] = 1
            add_row.append([self.colnames, tmp.tolist()])
        self.prob.linear_constraints.add(lin_expr=add_row, senses='L'*len(add_row), rhs=[1]*len(add_row),
                                    names=['c'+str(self.prob.linear_constraints.get_num()+i+1) for i in range(len(add_row))])

        self.prob.solve()
        if self.prob.solution.status[self.prob.solution.get_status()] != 'optimal':
            raise AssertionError('The solution is not optimal!')
        solution = self.prob.solution.get_objective_value()
        self.values = self.prob.solution.get_values()
        # print(solution, values, color_map)
        if solution <= self.ub:
            return 0, 0
        self.ub = solution
        self.children.append(Node(self, self.ub, self.graph, self.prob, self.color_map, self.color_num, self.values, self.colnames))
        return solution, values


if __name__ == '__main__':
    graph, n = read_networkx_graph('../DIMACS_all_ascii/MANN_a9.clq')
    prob = cplex.Cplex()
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)
    obj = np.ones(n)
    colnames = ['x'+str(i) for i in range(n)]
    prob.variables.add(obj=obj, names=colnames)
    rhs = np.ones(n).tolist() #right parts of constraints
    senses = 'L' * n #all constraints are '<='
    rownames = ['b'+str(i+1) for i in range(n)]#names of constraints
    rows = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        rows.append([colnames, a.tolist()])
    color_num, color_map = greedy_coloring_heuristic_multiple(graph) ## WARNING: modify it! Not best solution
    color_counter = 0
    rows2 = []
    for i in range(color_num):
        tmp = np.zeros(n)
        vertexes = [int(k)-1 for k, v in color_map.items() if i in v]
        tmp[np.array(vertexes)] = 1
        rows2.append([colnames, tmp.tolist()])
    rows += rows2
    rhs += np.ones(len(rows2)).tolist()
    rownames += ['c'+str(i+1) for i in range(len(rownames), len(rownames)+len(rows2))]#names of constraints
    senses += 'L'*len(rows2)
    prob = cplex.Cplex()
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)
    prob.variables.add(obj=obj, names=colnames)
    prob.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=rownames)
    prob.solve()
    solution = prob.solution.get_objective_value()
    values = prob.solution.get_values()
    parent_node = Node(None, n, graph,prob, color_map,color_num, np.array(values), colnames)
    current_best_integer_solution = 0
    max_possible_solution = 0
    solution, values = parent_node.solve(current_best_integer_solution)
    max_possible_solution = solution
    if np.sum(abs(values - np.array(values).astype(int))) == 0:
        print('Already integer solution for such data!')
        print("Solution value = ", solution)
        print('Optimal variables: ', dict(zip(parent_node.colnames, values)))
        exit()
    print('Float solution: ', solution)
    node = parent_node
    first_layer_size = len(parent_node.children)
    max_tree_depth = 0
    current_depth_position = 0
    current_progress = 1 - len(parent_node.children)/first_layer_size
    print(datetime.now(), 'Current progress: ', current_progress, ', best solution', current_best_integer_solution, ', max tree depth on last stage: ', max_tree_depth)

    while len(parent_node.children) != 0:
        while(len(node.children) != 0): #go down one branch
            node = node.children[0]
            current_depth_position+=1
            solution, values = node.solve(current_best_integer_solution)
        if current_depth_position > max_tree_depth:
            max_tree_depth = current_depth_position
        if solution > current_best_integer_solution:
            current_best_integer_solution = solution
            current_best_values = values
        while node.ub <= current_best_integer_solution or len(node.children) == 0:
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
