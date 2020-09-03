'''
Created on 24 Jan 2020

@author: ja9508

Code adapted from https://www.dcc.fc.up.pt/~jpp/code/py_metaheur/tsp.py
'''


#import math
import  sys
#import numpy
import random
import externaltsp as tool


class DTSP():

#     Initialisation for the class TSP given the number of cities and the distance matrix.
#     It is possible to use either symmetric or asymmetric TSP instances. 
    def __init__(self, cities=None, distance_matrix=None, rotation_list = None):
        self.size = cities
        self.matrix = distance_matrix
        if rotation_list == None:
            self.masks = [list(range(self.size))]
        else:
            self.masks = rotation_list
    
    def read_rotation_mask(self, filename):
        try:
            with open(filename) as f:
                self.masks = []
                for i, line in enumerate(f):
                    if "," in line:
                        self.masks.append([int(n) for n in line.strip().split(',')] )
        except:
            print("Dynamic environment unknown. Running static version...") 
    
    def evaluate(self, genes, change_index = 0):
        try:
            """Calculate the length of a tour according to distance matrix 'D'."""
            z = self.matrix[self.masks[change_index][genes[-1]], self.masks[change_index][genes[0]]]  # edge from last to first city of the tour
            for i in range(0, len(genes) -1 ):
                z += self.matrix[self.masks[change_index][genes[i]], self.masks[change_index][genes[i+1]]]  # add length of edge from city i-1 to i
            return z
        except:
            change_index = 0
            z = self.matrix[genes[self.masks[change_index][-1]], genes[self.masks[change_index][0]]]  # edge from last to first city of the tour
            for i in range(1, len(genes)):
                z += self.matrix[genes[self.masks[change_index][i]], genes[self.masks[change_index][i - 1]]]  # add length of edge from city i-1 to i
            return z

def generate_random_sTSP(problem_size):
    """Random creation of a symmetric TSP adjacency matrix.
    
    The number of cities needs to be specified before the use of the method.
    The generated values belongs to the values above the main diagonal.
    """
    matrix = {}
    for i in range(0, problem_size):
        for j in range(i + 1, problem_size):
            matrix[i,j] = matrix[j,i] = random.randint(1,99)
            
#            matrix[i,j] = numpy.random.randint(100)
            matrix[j,i] = matrix[i,j]
    return DTSP(problem_size, matrix)

def generate_random_aTSP(problem_size):
    """Random creation of an asymmetric TSP matrix.
    
    The number of cities needs to be specified before the use of the method.
    The generated values belongs to the values above the main diagonal.
    """
    matrix = {}
    for i in range(0, problem_size):
        for j in range(i + 1, problem_size):
            matrix[i,j] = random.randint(1,99)
            matrix[j,i] = random.randint(1,99)
#            matrix[i,j] = numpy.random.randint(100)
#            matrix[j,i] = numpy.random.randint(100)
    return DTSP(problem_size, matrix)

def read_tsplib(filename):
    "basic function for reading a TSP problem on the TSPLIB format"
    "NOTE: only works for 2D euclidean or manhattan distances"
    f = open(filename, 'r');

    line = f.readline()
    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        dist = tool.distL2
    elif line.find("MAN_2D") != -1:
        dist = tool.distL1
    else:
        print("cannot deal with non-euclidean or non-manhattan distances")
        raise Exception

    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()

    xy_positions = []
    while 1:
        line = f.readline()
        if line.find("EOF") != -1: break
        (i, x, y) = line.split()
        x = float(x)
        y = float(y)
        xy_positions.append((x, y))

    n, D = tool.mk_matrix(xy_positions, dist)
    return DTSP(cities = n, distance_matrix = D)

        
"""
MAIN
"""
if __name__ == "__main__":
    print("1")
    if len(sys.argv) == 1:
        print("1.1")
        problem = read_tsplib("H:\\PythonTester\\TSP\\TSPLIB\\test5.tsp")
    else:
        print("1.2")
        instance = sys.argv[1]
        problem = read_tsplib(instance)  # create the distance matrix
    print("done")
        
