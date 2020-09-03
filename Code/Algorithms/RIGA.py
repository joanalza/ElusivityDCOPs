'''
Created on 24 Jan 2020
@author: ja9508
Code obtained from https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
'''

import csv
import sys
import random
import timeit
import os.path

sys.path.insert(1, "src\Algorithms")
import Population
sys.path.insert(1, "src\Problems")
import DTSP
import DKP
sys.path.insert(1, "src\Individual")
import Permutation
import binary

"""
Main
"""
def main():
    start_time = timeit.default_timer()
    random.seed(1)
    
    # Parametrization of the required variables
    output_name = None
    problem = DKP.generate_random_KP(10)
#    population_size = 10 * problem.size
    population_size = 100
    create_new_member = binary.create_new_solution
    maxGens = 10000
    
    # Algorithm's specific parameters
    pool_size = int(problem.size * 0.05)
    if pool_size < 1 : pool_size = 1
    crossover_function = single_point_crossover
    mutation_function = mutate_binary
    mutation_rate = 0.01
    
    # Dynamics initialisation
    alg_version = "ri"
    frequency = int(2.5 * problem.size)
    n_immigrants = int(0.2 * population_size)
    
    # Update variables if there are inputs
    # instance Problem_instances/TSP/kroA100.tsp
    # dynamic Dynamics/PPn100sH0.1_1.txt 
    # result Results/GA-kroA100-PPn100sH0.1f10_1.csv 
    # stop 1000 
    # seed 1236329674 
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
                create_new_member = Permutation.create_new_permutation
                crossover_function = crossover_4_permutation
                mutation_function = mutation_4_permutation
                pool_size = int(problem.size * 0.05)
                frequency = int(2.5 * problem.size)
            elif 'instance' in parameters and "kp" in parameters['instance']:
                problem = DKP.read_instance(parameters['instance'])
                create_new_member = binary.create_new_solution
                crossover_function = single_point_crossover
                mutation_function = mutate_binary
                pool_size = int(problem.size * 0.05)
                frequency = int(2.5 * problem.size)
            if 'dynamic' in parameters: 
                problem.read_rotation_mask(parameters['dynamic'])
            if 'result' in parameters: 
                output_name = parameters['result']
            if 'pop' in parameters: 
                population_size = int(parameters['pop'])
                n_immigrants = int(0.2 * population_size)
            if 'stop' in parameters: 
                maxGens = int(parameters['stop'])
            if 'seed' in parameters: 
                random.seed(int(parameters['seed']))
            if 'algorithm' in parameters: 
                alg_version = parameters['algorithm']
            if 'mating' in parameters: 
                pool_size = int(parameters['mating'])
            if 'mutation' in parameters: 
                mutation_rate = float(parameters['mutation'])
            if 'freq' in parameters: 
                frequency = int(parameters['freq'])
        
        
    # Initialise all the necessary parameters
    generations = evaluations = 0
    change = 0
    changeDetected = False
    output = [["Generation","Best","PopAvg","Change","Algorithm"]]
    
    # Save best solution before change
#    f_offline = []
    
        
    # Initialise population generating a random population (the solutions are already evaluated and the average is calculated too)      
    population, pop_average, evaluations = Population.create_random_population(population_size, problem.size, create_new_member, problem.evaluate, evaluations, change)
    
    population = sorted(population, key = lambda individual:individual[1])#, reverse = False)
    best_solution = population[0]
    
    # Iteration: terminate?
    while (generations < maxGens):
        generations += 1
        children = new_population = []
        print("Generation: ", generations)
        
        # Check if there has been an environmental change
        changeDetected = detect_change(population, problem.evaluate, change)
        
        # Apply dynamic approach
        if "restart" in alg_version and changeDetected:
            # Restart the entire population and sort it
#            print("Restart")
            population, pop_average, evaluations = Population.create_random_population(population_size, problem.size, create_new_member, problem.evaluate, evaluations, change)
            population = sorted(population, key = lambda individual:individual[1])#, reverse = False)
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
                population = sorted(population, key = lambda individual:individual[1])#, reverse = False)
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
                population = sorted(population, key = lambda individual:individual[1])#, reverse = False)
                pop_average = Population.population_calculate_average(population)
                best_solution = population[0]
        
        
        # Store the information of each generation
        output.append([generations,best_solution[1],pop_average,change,alg_version])
        
        # Generation of the new offspring using selection, crossover and mutation
        for i in range(len(population)):
            parent1, parent2 = selection(population, pool_size)
            offspring = crossover_function(parent1[0], parent2[0])
            offspring = mutation_function(offspring, mutation_rate)
            evaluations += 1
            children.append((offspring, problem.evaluate(offspring, change)))
        
        # Merge previous population and new individuals and select the best ones
        new_population = population + children
        population = sorted(new_population, key = lambda individual:individual[1])[0:len(population)]        
        pop_average = Population.population_calculate_average(population)
        best_solution = population[0]
#        print("Generation ",generations ," time: ", timeit.default_timer() - start_time)
        
        # Change the environment depending on the frquency and the current generation
        if(generations % frequency == 0):
            try:
#                print("Best solution so far: ", best_solution)
                print("Rotation permutation: ", problem.masks[change + 1])
                change += 1
            except:
                pass
#                problem.masks.append(create_new_member(problem.size))
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
def detect_change(population, evaluation_function, i_change):
    best_of_population = population[0]
    if best_of_population[1] != evaluation_function(best_of_population[0], i_change):
        return True
    return False

def selection(population, selection_size):
    """ Tournament selection. A se of randomly selected individuals are chosen from the population
            and the one with the best fitness value in the group is considered as the first parent. The second one is selected
            following the same criteria.
    """
#    if len(set([str(item[0]) for item in population])) >= selection_size:
#        parent1 = sorted(random.sample(population, k = selection_size), key = lambda individual:individual[1])[0]
#        while True:
#            parent2 = sorted(random.sample(population, k = selection_size), key = lambda individual:individual[1])[0]
#            if parent1 != parent2:
#                break
#    else:
    parent1 = sorted(random.sample(population, k = selection_size), key = lambda individual:individual[1])[0]
    parent2 = sorted(random.sample(population, k = selection_size), key = lambda individual:individual[1])[0]
    return parent1,parent2


def crossover_4_permutation(individual1, individual2):
    """ Ordered crossover. Randomly select a subset of the first parent and fill the remainder from the genes of the second
            parent in the same order in which they appear (without duplicating any genes). 
    """
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(individual1))
    geneB = int(random.random() * len(individual2))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(individual1[i])
        
    childP2 = [item for item in individual2 if item not in childP1]

    child = childP1 + childP2
    return child

def single_point_crossover(individual1, individual2):
    """ Ordered crossover. Randomly select a subset of the first parent and fill the remainder from the genes of the second
            parent in the same order in which they appear (without duplicating any genes). 
    """

    child = list(individual1)
    
    # Generate the crsover point randomly
    k = int(random.random() * len(individual1))

    # Interchange the genes of the parents
    for i in range(k, len(individual1)):
        child[i] = individual2[i]

    return child
 
def mutate_binary(individual, mut_prob):
    return [abs(1-bit) if random.random() < mut_prob else bit for bit in individual]

def mutation_4_permutation(individual, mutationProbability):
    for swapped in range(len(individual)):
        if(random.random() < mutationProbability):
            swapWith = int(random.random() * len(individual))
            
            gene1 = individual[swapped]
            gene2 = individual[swapWith]
            
            individual[swapped] = gene2
            individual[swapWith] = gene1
    return individual


def print_usage():
    my_list = ["GA.py","instance","[intance file]","dynamic","[dynamic file]","result","[results file]",
               "stop","[maximum generations]","seed","[seed]","pop","[population size]", "mating", "[pool size]",
               "algorithm","{standard, ri, restart}*","mutation","[float]"]
#    print("GA.py instance [intance file] dynamic [dynamic file] result [results file] stop [maximum generations] seed [seed] pop [population size] elitism {0,1} algorithm {dynamic,restart}* crossover [float] mutation [float]")
    print(" ".join(my_list))
    
def print_output(result):
    print("\n".join(map(str,result)))

if __name__ == "__main__":
#    perm = mutation_4_permutation([0,1,2,3,4], 0.1)
    result = main()
    # instance Problem_instances/TSP/kroA100.tsp stop 10000 pop 100