'''
Created on 5 Feb 2020

@author: ja9508
'''

import random
import sys


class DKP():

#     Initialisation for the class TSP given the number of cities and the distance matrix.
#     It is possible to use either symmetric or asymmetric TSP instances. 
    def __init__(self, length = None, weights=None, profits=None, capacity = None, xor_list = None):
        self.size = length
        self.values = profits
        self.weights = weights
        self.capacity = capacity
        self.masks = xor_list
        if self.masks == None:
            self.masks = [[0] * self.size]
    
    def read_rotation_mask(self, filename = None):
        try:
            with open(filename) as f:
                self.masks = []
                for line in f:
                    if "," in line:
                        self.masks.append([int(n) for n in line.strip().split(',')] )
        except:
            print("Dynamic environment unknown. Running static version...")            
    
    def evaluate(self, binary_vector, change_index = 0):
        """Evaluates the quality of a given binary vector.
           If the sum of weights exceed the capacity --> penalty: 10^-10 [sum(weights) - sum(weights * binary)]
              from "Experimental study on PBIL algorithms for DOPs" S. Yang
        """
        try:    
            weight_sum = sum([self.weights[i] * (binary_vector[i] ^ self.masks[change_index][i]) for i in range(len(binary_vector))])
            profit_sum = sum([self.values[i] * (binary_vector[i] ^ self.masks[change_index][i]) for i in range(len(binary_vector))])
            if weight_sum > self.capacity:
                return -(pow(10, -10) * (sum(self.weights) - weight_sum))
            return -profit_sum
        except:
            change_index = 0
            weight_sum = sum([(self.weights[i] * binary_vector[i]) ^ self.masks[change_index][i] for i in range(len(binary_vector))])
            profit_sum = sum([(self.values[i] * binary_vector[i]) ^ self.masks[change_index][i] for i in range(len(binary_vector))])
            if weight_sum > self.capacity:
                return -(pow(10, -10) * (sum(self.weights) - weight_sum))
            return -profit_sum

def read_instance(filename):
    
    n = w = p = c = 0    
    
    with open(filename) as f:
#        for line in f:
        while True:
            line = f.readline()
            if "Elements" in line:
                n = int(f.readline())
            elif "Profit" in line:
                line = f.readline()
                p = [int(i) for i in line.strip().split(',')]  
            elif "Weight" in line:
                line = f.readline()
                w = [int(i) for i in line.strip().split(',')]
            elif "Capacity" in line:
                c = int(f.readline())
            elif line.find("EOF") != -1 or not line: 
                break

    return DKP(n,w,p,c)


"""Idea obtained from the article: "GAs with Memory- and Elitism-Based Immigrants in Dynamic Environments". S. Yang
    The function constructs a binary knapsack problem with the weight and profit randomly created 
    in the range of [1,30] and the capacity of the kapsack set to half of the total weight items.
    
    @param probelm_size: size of the problem wanted to generate.
"""
def generate_random_KP(problem_size):
    weights = [random.randint(1,30)  for i in range(problem_size)]
    profits = [random.randint(1,30)  for i in range(problem_size)]
    capacity = sum(weights)/2
    return DKP(problem_size,weights,profits,capacity)

"""
MAIN
"""
if __name__ == "__main__":
    if len(sys.argv) == 1:
        problem = read_instance("H:\Joan_Files\PhD\Code\PythonCode\DCOP_Spyder\Problem_instances\KP\golberg.kp")
#    else:
#        instance = sys.argv[1]
#        problem = read_tsplib(instance)  # create the distance matrix
    print("done")
        
