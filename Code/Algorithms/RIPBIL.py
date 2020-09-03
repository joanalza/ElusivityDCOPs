'''
Created on 3 Feb 2020

@author: ja9508

Code obtained from http://www.cleveralgorithms.com/nature-inspired/probabilistic/pbil.html
                    https://en.wikipedia.org/wiki/Population-based_incremental_learning
'''

import csv
import sys
import random
import timeit
import os.path
#from collections import Counter

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
    population_size = 3
    maxGens = 100
    elitism = True
    
    # Algorithm's specific parameters
    learning_rate = 0.25    
    mutation_probability = 0.02
    mutation_rate = 0.05
    model = [0.5] * problem.size
    # model = [0.95, 0.05, 0.95, 0.05, 0.05, 0.95, 0.05, 0.95, 0.95, 0.05]
    reinitialised_model = [0.5] * problem.size
    generate_candidate = generate_binary_candidate
    create_new_member = binary.create_new_solution
    update_model = update_binary_vector
    mutation_model = mutate_binary_vector
    
    # Dynamics initialisation
    alg_version = "ri"
    frequency = int(2.5 * problem.size)
    n_immigrants = int(0.2 * population_size)
    
    # Update variables if there are inputs
    if len(sys.argv) != 1:
        if "?" in sys.argv:
            print_usage()
            return 0
        else:
            # instance Problem_instances/KP/joanA100.kp 
            # dynamic Dynamics/BPn100sH0.1_1.txt 
            # result Results/PBIL-joanA100-BPn100sH0.1f10_1.csv 
            # stop 1000 
            # seed 1346684559 
            # pop 25 
            # algorithm restart 
            # freq 10
            parameters = dict([ (sys.argv[i], sys.argv[i+1]) for i in range(1,len(sys.argv),2)])
            if 'instance' in parameters and "tsp" in parameters['instance']:
                problem = DTSP.read_tsplib(parameters['instance'])
                model = [[1.0/problem.size] * problem.size] * problem.size
                reinitialised_model = [[1.0/problem.size] * problem.size] * problem.size
                generate_candidate = generate_permutation_candidate
                create_new_member = Permutation.create_new_permutation
                update_model = update_permutation_matrix
                mutation_model = mutate_permutation_matrix
                frequency = int(2.5 * problem.size)
            elif 'instance' in parameters and "kp" in parameters['instance']:
                problem = DKP.read_instance(parameters['instance'])
                model = [0.5] * problem.size
                reinitialised_model = [0.5] * problem.size
                generate_candidate = generate_binary_candidate
                create_new_member = binary.create_new_solution
                update_model = update_binary_vector
                mutation_model = mutate_binary_vector
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
            if 'elitism' in parameters: 
                elitism = bool(parameters['elitism'])
            if 'algorithm' in parameters: 
                alg_version = parameters['algorithm']
            if 'selection' in parameters: 
                truncation_size = parameters['selection'] 
            if 'freq' in parameters: 
                frequency = int(parameters['freq'])
        
    # Initialise all the necessary parameters
    truncation_size = int(population_size * 0.5)
    generations = evaluations = 0
    change = 0
    pop_average = 0  
    best_solution = None
    changeDetected = False 
    output = [["Generation","Best","PopAvg","Change","Algorithm"]]
    
#    print(parameters)
    
    # Iteration: terminate?
    while (generations < maxGens):
#        print("Iteration start time: ", timeit.default_timer() - generation_time)
        
        generations += 1
        population = []
        count = []
        print(generations)
        
        # Generate samples
        for i in range(population_size):
            if i == 0 and elitism and generations!= 1:
                population.append(best_solution)
            else:
                candidate = generate_candidate(model)
    #            print(Permutation.is_permutation(candidate), candidate)
                population.append((candidate, problem.evaluate(candidate, change)))
                count.append(str(candidate))
                evaluations += 1
        
        # Sort existing population and evaluate them
        population = sorted(population, key = lambda individual:individual[1], reverse = False)
        pop_average = Population.population_calculate_average(population)
        best_solution = population[0]
        
        # Check if there has been an environmental change
        changeDetected = detect_change(population, problem.evaluate, change)
        
        # Apply dynamic approach
        if "restart" in alg_version and changeDetected:
            # Restart the entire population and sort it
            population, pop_average, evaluations = Population.create_random_population(population_size, problem.size, create_new_member, problem.evaluate, evaluations, change)
            population = sorted(population, key = lambda individual:individual[1], reverse = False)
            best_solution = population[0]
            model = list(reinitialised_model)
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
        
        # Update probability vector towards best solution
        truncated_pop = population[0:truncation_size]
        # truncated_pop = population[0]
        model = update_model(model, truncated_pop, learning_rate)
        
        # Mutate probability vector
        model = mutation_model(model, mutation_probability, mutation_rate)
        
        # print("Model: ", model)
            
        # Store the information of each generation
        output.append([generations,best_solution[1],pop_average,change,alg_version])
        
        # Change the environment depending on the frquency and the current generation
#        if(generations % frequency == 0):
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
#        print_output(output)
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

def generate_binary_candidate(vector): 
    return [1 if random.uniform(0,1) < p_i else 0 for p_i in vector]

def update_binary_vector(probability_vector, population, lrate):
    if type(population) == list:
        solutions = [item[0] for item in population]
        probabilities = [float(sum(col)) / float(len(col)) for col in zip(*solutions)]
    elif type(population) == tuple:
        probabilities = population[0]
    return [probability_vector[i]*(1.0-lrate) + probabilities[i]*lrate for i in range(len(probability_vector))]

def mutate_binary_vector(vector, mut_prob, shift):
    return [bit * (1.0 - shift) + (random.uniform(0,1) * shift) if random.uniform(0,1) < mut_prob else bit for bit in vector]


def generate_permutation_candidate(matrix):
    """
    Idea obtained from https://cs.stackexchange.com/questions/31863/random-permutations-by-probability-matrix
    Extended idea with the supervisory team: work with availabilityveector and probability sum vector
    """
    # Required parameters for the efficiency of the code
    matrix_size = len(matrix)
    a = 0
    
    # Initialisation of the required elements
    new_perm = [None] * matrix_size
#    availability = [True] * matrix_size
    available_elements = list(range(matrix_size))
    aux_probability = [sum(row) for row in matrix]
    updated_probabilities = [0] * matrix_size
    
    # Generate a random permutation to do not alter the original probabilities 
    random_perm = Permutation.create_new_permutation(matrix_size)
    
    # For efficiency, reorder the matrix according to the generated random permutation to loop through
    aux_matrix = [matrix[i] for i in random_perm]
    transposed_matrix = list(zip(*aux_matrix))
    
    # By an iteration process build a solution
    for probability_row in aux_matrix:
        # Update the required parameters for the probabilities
#        available_elements = [i for i, x in enumerate(availability) if x]
#        if (sum(availability) == 1):
#            new_perm[a] = available_elements[0]
#            break
#        updated_probabilities = [probability_row[i]/aux_probability[i] for i in available_elements]
            
        if (len(available_elements) > 1):
            for i in available_elements:
                try:
                    updated_probabilities[i] = float(probability_row[i]) / float(aux_probability[a])
                except:
                    updated_probabilities[i] = 0
        else:
            new_perm[random_perm[a]] = available_elements[0]
            break
            
        
        # Generate a gene according to the probabilities and add it to the constructing solution
        new_elem = weighted_choice(available_elements, weights=updated_probabilities)
        new_perm[random_perm[a]] = new_elem
        a += 1
        
        # Update availability and probability sum vectors
        available_elements.remove(new_elem)
        matrix_column = transposed_matrix[new_elem]
        for i in range(matrix_size):
            aux_probability[i] -= matrix_column[i]
        
    return new_perm


def weighted_choice(choices, weights):
    r = random.uniform(0, 1)
    upto = 0
    for i in choices:
        if upto + weights[i] >= r:
            return i
        upto += weights[i]
    # Shouldnt get here
    return random.sample(choices,1)[0]

def update_permutation_matrix(probability_matrix, population, lrate):
    solutions = [item[0] for item in population]      
    probabilities = [[float(tuple(col).count(i)) / float(len(col)) for i in range(len(probability_matrix))] for col in zip(*solutions)]
    return [[probability_matrix[i][j]*(1.0-lrate) + probabilities[i][j]*lrate for j in range(len(probability_matrix))] for i in range(len(probability_matrix))]


def mutate_permutation_matrix(matrix, mut_prob, shift):
    """
    Mutation using the inversion operator. Two cities are randomly selected and the sub-tour between them is reversed.
    """
    return matrix
#    return [[bit * (1.0 - shift) + (random.random() * shift) if random.random() < mut_prob else bit for bit in vector] for vector in matrix]



def check_model_correctness(matrix):
    row_list = [True if sum(row)>=0.9 and sum(row)<=1 else False for row in matrix]
    #col_list = [True if sum(col)>=0.9 and sum(col)<=1 else False for col in zip(*matrix)]
    return all(row_list) #and all(col_list)

def print_usage():
    my_list = ["PBIL.py","instance","[intance file]","dynamic","[dynamic file]","result","[results file]",
               "stop","[maximum generations]","seed","[seed]","pop","[population size]","selection","[selection_size]",
               "elitism","{True, False}","algorithm","{dynamic,restart}*","learning","[small float]",
               "mut_prob","[small float]", "mut_shift","[small float]"]
#    print("GA.py instance [intance file] dynamic [dynamic file] result [results file] stop [maximum generations] seed [seed] pop [population size] elitism {0,1} algorithm {dynamic,restart}* crossover [float] mutation [float]")
    print(" ".join(my_list))
    
def print_output(result):
    print("\n".join(map(str,result)))

if __name__ == "__main__":
#    pass
    result = main()