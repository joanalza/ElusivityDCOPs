'''
Created on 3 Feb 2020

@author: ja9508
'''

#import numpy
from random import randint

def create_new_solution(sol_size):
    """Generates a u.a.r. permutation

    @param permu_size : size of the permutation wanted to generate.
"""
    return [randint(0,1) for b in range(1,sol_size+1)]
#    return list(numpy.random.randint(2, size= sol_size))