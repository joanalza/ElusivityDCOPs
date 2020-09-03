'''
Created on 24 Jan 2020

@author: ja9508

Pseudocode obtained from https://www.researchgate.net/profile/Majdi_Mafarja/publication/329909719/figure/fig7/AS:712085495349253@1546785820536/The-pseudo-code-for-Ant-Colony-Optimization.png
                    and https://github.com/Vampboy/Ant-Colony-Optimization
                    and https://www.sciencedirect.com/topics/engineering/ant-colony-optimization
                    and https://www.youtube.com/watch?v=9KVg3gv_OcA (3:03)
                    and https://towardsdatascience.com/using-ant-colony-and-genetic-evolution-to-optimize-ride-sharing-trip-duration-56194215923f
'''

import csv
import sys
import random
import timeit
import os.path
#from collections import Counter
#from random import seed
#from random import randint

sys.path.insert(1, "src\Algorithms")
import Population
sys.path.insert(1, "src\Problems")
#from Problem import Problem
import DTSP
sys.path.insert(1, "src\Individual")
import Permutation

"""
Main
"""
def main():
    start_time = timeit.default_timer()
    random.seed(1)
    
    # Parametrization of the required variables
    output_name = None
    problem = DTSP.generate_random_sTSP(20)
    colony_size = 25
    maxGens = 300
    elitism = True
    create_new_member = Permutation.create_new_permutation
    
    # Algorithm's specific parameters
    evaporation_rate = 0.2
    alpha = 1   # Pheromone factor
    beta = 0   # Visibility factor
    
    # Dynamics initialisation
    alg_version = "ri"
    frequency = int(2.5 * problem.size)
    n_immigrants = int(0.25 * colony_size)
    
    
    
    # Update variables if there are inputs
    # instance Problem_instances/TSP/kroA100.tsp 
    # dynamic Dynamics/PPn100sH0.1_1.txt 
    # result Results/ACO-kroA100-PPn100sH0.1f10_1.csv 
    # stop 1000 
    # seed 1328921976 
    # pop 25 
    # algorithm restart 
    # freq 10
    if len(sys.argv) != 1:
        if "?" in sys.argv:
            print_usage()
            return 0
        else:
#            dict([(i,i+1) for i in range(0,5,2)])
            parameters = dict([ (sys.argv[i], sys.argv[i+1]) for i in range(1,len(sys.argv),2)])
            if 'instance' in parameters and "tsp" in parameters['instance']:
                problem = DTSP.read_tsplib(parameters['instance'])  # create the distance matrix
                frequency = int(2.5 * problem.size)
            if 'dynamic' in parameters: 
                problem.read_rotation_mask(parameters['dynamic'])
            if 'result' in parameters: 
                output_name = parameters['result']
            if 'pop' in parameters: 
                colony_size = int(parameters['pop'])
                n_immigrants = int(0.25 * colony_size)
            if 'stop' in parameters: 
                maxGens = int(parameters['stop'])
            if 'seed' in parameters: 
                random.seed(int(parameters['seed']))
            if 'elitism' in parameters: 
                elitism = bool(parameters['elitism'])
            if 'algorithm' in parameters: 
                alg_version = parameters['algorithm']
            if 'evap' in parameters: 
                evaporation_rate = float(parameters['evap'])
            if 'alpha' in parameters: 
                alpha = float(parameters['alpha'])
            if 'beta' in parameters: 
                beta = float(parameters['beta'])
            if 'freq' in parameters: 
                frequency = int(parameters['freq'])
        
        
    # Initialise all the necessary parameters
    generations = evaluations = 0
    change = 0
    changeDetected = False
    best_solution = None
    output = [["Generation","Best","PopAvg","Change","Algorithm"]]
    ant = [] * problem.size
    
    # Algorithm's specific parameters setting
    pheromone = [[1.0/problem.size] * problem.size] * problem.size
    initial_pheromone = [[1.0/problem.size] * problem.size] * problem.size
    visibility = dict([(name, 1.0/elem) for name, elem in problem.matrix.items()])
    
    # Iteration: terminate?
    while (generations < maxGens):
        population = []
        generations += 1
        count = []
        print("Generation: ",generations)
        
        # Initialise colony by placing ants in ramdom positions   
        if colony_size <= problem.size:
            colony = random.sample(range(problem.size), colony_size)
        else:
            colony = [random.randint(0,problem.size - 1) for i in range(colony_size)]
        
        # Each ant completely builds a solution and update trail
        for initial_location in colony:
            if population == list() and elitism and generations!= 1:
                population.append(best_solution)
            else:
                ant = construct_solution(initial_location, pheromone, problem, visibility, alpha, beta, change)
            
#               print(ant)
                population.append((ant, problem.evaluate(ant, change)))
                count.append(str(ant))
                evaluations +=1
                
        # Sort generated population
#        print(Counter(count))
        population = sorted(population, key = lambda individual:individual[1], reverse = False)
        pop_average = Population.population_calculate_average(population)
        best_solution = population[0]
        # print("Population: ",population)
        # print("Pheromone: ", pheromone)
        
        # Check if there has been an environmental change
        changeDetected = detect_change(population, problem.evaluate, change)
        
        # Apply dynamic approach
        if "restart" in alg_version and changeDetected:
#            print("Restarted")
            population = []
            # Reinitialise pheromone trail
            pheromone = list(initial_pheromone)
            
            # Restart the entire population and sort it       
            colony = [random.randint(0,problem.size - 1) for i in range(colony_size)]
            
            # Each ant completely builds a solution and update trail
            for initial_location in colony:
                ant = construct_solution(initial_location, pheromone, problem, visibility, alpha, beta, change)
            
#               print(ant)
                population.append((ant, problem.evaluate(ant, change)))
                count.append(str(ant))
                evaluations +=1
                    
            # Sort generated population
            population = sorted(population, key = lambda individual:individual[1], reverse = False)
            pop_average = Population.population_calculate_average(population)
            best_solution = population[0]  
        elif "restart" in alg_version and not changeDetected:
            # Replace worst individuals using immigrant schemes    
            for i_immigrant in range(1, n_immigrants + 1):
                random_immigrant = create_new_member(problem.size)
                population[-i_immigrant] = (random_immigrant, problem.evaluate(random_immigrant, change))
                evaluations += 1
        elif "ri" in alg_version:
            if changeDetected:
                # Re-evaluate the population and sort it
                population, evaluations = Population.evaluate_population(population, problem.evaluate, change, evaluations)
                population = sorted(population, key = lambda individual:individual[1], reverse = False)
                pop_average = Population.population_calculate_average(population)
                best_solution = population[0]
                
            # Replace worst individuals using immigrant schemes    
            for i_immigrant in range(1, n_immigrants + 1):
                random_immigrant = create_new_member(problem.size)
                population[-i_immigrant] = (random_immigrant, problem.evaluate(random_immigrant, change))
                evaluations += 1
        elif "standard" in alg_version:
            if changeDetected:
                # Re-evaluate the population and sort it
                population, evaluations = Population.evaluate_population(population, problem.evaluate, change, evaluations)
                population = sorted(population, key = lambda individual:individual[1], reverse = False)
                pop_average = Population.population_calculate_average(population)
                best_solution = population[0]
        
        # Store the information of each generation
        output.append([generations,best_solution[1],pop_average,change,alg_version])
        
        # Evaporate the pheromone
        pheromone = [[(1-evaporation_rate) * phr for phr in row] for row in pheromone]
        
        # Update pheromone ading more weight to the pheromones
        sol = best_solution[0]
        fit = best_solution[1]
        summation =  1.0/float(fit)
        for i in range(len(sol) - 1):
            pheromone[sol[i]][sol[i + 1]] += summation
        pheromone[sol[-1]][sol[0]] += summation
        
        # Change the environment depending on the frquency and the current generation
        if(generations % frequency == 0):
            try:
#                print("Best solution so far: ", best_solution)
                print("Rotation permutation: ", problem.masks[change + 1], "at generation ", generations)
                change += 1
            except:
                pass
#                problem.masks.append(Permutation.create_new_permutation(problem.size))
#                print("Environment changed randomly to: ", problem.masks[change])
    
    # Write the results on a csv file
    print("Execution time: ", timeit.default_timer() - start_time)
    if output_name == None:
        print_output(output)
        pass
    else:
        random.seed()
        witing_time = random.uniform(0,10)
        print("Execution paused ", witing_time, " seg. to avoid data loosing.")
        timeit.time.sleep(witing_time)
        if os.path.isfile(output_name): output.pop(0) # Remove headers
        with open(output_name, 'a+') as file:
            writer = csv.writer(file)
            writer.writerows(output)
            
            
    return output
        
        
    
"""
COMPLEMENTARY FUNCTIONS
"""
def construct_solution(initial_city, pheromone, problem, visibility, alpha, beta, change_index):
    # Copy of the visibility matrix (it is going to be changing constantly)
    aux_visibility = dict(visibility).copy()
#                print(aux_visibility == visibility)
    ant = [initial_city]
    mask = problem.masks[change_index]
    
    for gene in range(problem.size - 1):
        # Obtain initial city of the ant
        current_city = ant[gene]
        
        # Make visibility of the current city to 0
#                    aux_visibility = {name: (elem if name[1] != current_city else 0) for name, elem in aux_visibility.items()}
        for i in range(problem.size):
            if (problem.masks[change_index][i],current_city) in aux_visibility: 
                aux_visibility[mask[i],current_city] = 0
        
        # Calculate probabilities to choose next city
        probabilities = calculate_probability(current_city, pheromone[current_city],
                                              aux_visibility, alpha, beta)
        
#                    cum_probabilities = cumulative_sum(probabilities)
        
        # Find next city having probability higher than random
        r = random.uniform(0, 1)
        upto = 0
        for i in range(problem.size):
            if upto + probabilities[i] >= r:
                new_gene = i
                break
            upto += probabilities[i]
        
        # Add new gene to the constructing ant
        ant.append(new_gene)
    return ant

def detect_change(population, evaluation_function, i_change):
    best_of_population = population[0]
    if best_of_population[1] != evaluation_function(best_of_population[0], i_change):
        return True
    return False

def calculate_probability(gene, pheromones, visibility, alpha, beta):
    # Store esential information to reduce computational cost
    n = len(pheromones)
    combine_feature = probabilities = [0] * n
    trail = heuristic = total = 0
    
    for i in range(len(pheromones)):
        trail = pheromones[i] ** alpha
        
        #calculating pheromone feature 
        if i!= gene and visibility[gene, i] != 0:
            heuristic = visibility[gene, i] ** beta
        else:
            heuristic = 0
        
        combine_feature[i] = trail * heuristic
                
    total = sum(combine_feature)
    
    probabilities = [float(x) / float(total) for x in combine_feature]   #finding probability of element probs(i) = comine_feature(i)/total
    return probabilities

#def cumulative_sum(lists): 
#    return [sum(lists[0:x + 1]) for x in range(0, len(lists))] 

def print_usage():
    my_list = ["ACO.py","instance","[intance file]","dynamic","[dynamic file]","result","[results file]",
               "stop","[maximum generations]","seed","[seed]","pop","[colony size]","elitism","{0,1}",
               "algorithm","{dynamic,restart}*",
               "evap","[evaporation rate]","alpha","[pheromone factor]","beta","[visibility factor]"]
    print(" ".join(my_list))
    
def print_output(result):
    print("\n".join(map(str,result)))

if __name__ == "__main__":
#    perm = mutation_4_permutation([0,1,2,3,4], 0.1)
    result = main()
