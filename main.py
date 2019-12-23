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
    def __init__(self, parent, ub,graph, obj, colnames, rows1, senses1, rhs1, rownames1):
        self.parent = parent
        self.ub = ub
        self.graph = graph
        self.obj = obj
        self.colnames = colnames # names of variables, always constant
        self.rows = rows1 # left parts of constraints
        self.senses = senses1 # types of constraints
        self.rhs = rhs1 # right parts of constraints
        self.rownames = rownames1 # names of constraints

    def solve(self, current_best_integer_solution):
        color_num, color_map = greedy_coloring_heuristic(self.graph) ## WARNING: modify it! Not best solution
        color_counter = 0
        rows2 = []
        tmp = np.zeros(n)
        for k, v in sorted(color_map.items(), key=lambda item: item[1]):
            if v == color_counter:
                tmp[int(k) -1] = 1
            else:
                color_counter+=1
                rows2.append([colnames, tmp.tolist()])
                tmp = np.zeros(n)
                tmp[int(k)-1] = 1
        rows2.append([colnames, tmp.tolist()])
        self.rows += rows2
        self.rhs += np.ones(len(rows2)).tolist()
        self.rownames += ['c'+str(i+1) for i in range(len(self.rownames), len(self.rownames)+len(rows2))]#names of constraints
        self.senses += 'L'*len(rows2)
        prob = cplex.Cplex()
        prob.set_results_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)
        prob.variables.add(obj=self.obj, names=self.colnames)
        prob.linear_constraints.add(lin_expr=self.rows, senses=self.senses,
                                    rhs=self.rhs, names=self.rownames)
        self.prob = prob
        prob.solve()
        if prob.solution.status[prob.solution.get_status()] != 'optimal':
            print('Not opt')
            raise AssertionError('The solution is not optimal!')
        solution = prob.solution.get_objective_value()
        values = prob.solution.get_values()
        self.ub = solution


if __name__ == '__main__':
    graph, n = read_networkx_graph('../DIMACS_all_ascii/playground.clq')
    prob = cplex.Cplex()
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.maximize)
    obj = np.ones(n)
    colnames = ['x'+str(i) for i in range(n)]
    prob.variables.add(obj=obj, names=colnames)
    rhs1 = np.ones(n).tolist() #right parts of constraints
    senses1 = 'L' * n #all constraints are '<='
    rownames1 = ['b'+str(i+1) for i in range(n)]#names of constraints
    rows1 = []
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1
        rows1.append([colnames, a.tolist()])


    parent_node = Node(None, n, graph, obj, colnames, rows1, senses1, rhs1, rownames1)

    current_best_integer_solution = 0
    max_possible_solution = 0
    solution, values = parent_node.solve(current_best_integer_solution)
    exit()
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
        while node.any_solution <= current_best_integer_solution or node.ub <= current_best_integer_solution or len(node.children) == 0:
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
