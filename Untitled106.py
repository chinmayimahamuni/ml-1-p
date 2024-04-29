#!/usr/bin/env python
# coding: utf-8

# In[5]:


# cl3 6
import numpy as np

# Parameters
num_antibodies = 10  # smaller population
mutation_scale = 0.5  # mutation impact, how much an antibody can change during mutation.
num_generations = 20  # number of iterations

# Objective function (calculate_affinity)
# Evaluate the fitness of the population. Lower the value better the likelihood.
def objective_function(x):
    return x**2

# Initialize antibodies
# At first each option is the considered as the solution.
antibodies = np.random.uniform(-10, 10, num_antibodies)

# Run the clonal selection algorithm
for _ in range(num_generations):
    # Evaluate fitness
    fitness = objective_function(antibodies)
    
    # Select the best antibody (lowest fitness)
    best_idx = np.argmin(fitness)
    best_antibody = antibodies[best_idx]
    
    # Generate a mutant from the best antibody
    mutant = best_antibody + np.random.normal(0, mutation_scale)
    
    # Evaluate mutant fitness
    mutant_fitness = objective_function(mutant)
    
    # Replace the worst antibody if mutant is better
    worst_idx = np.argmax(fitness)
    if mutant_fitness < fitness[worst_idx]:
        antibodies[worst_idx] = mutant

# Result
best_idx = np.argmin(objective_function(antibodies))
best_antibody = antibodies[best_idx]
best_fitness = objective_function(best_antibody)

print("Best antibody:", best_antibody)
print("Best fitness:", best_fitness)


# In[2]:


# cl3 5 coconut
import numpy as np
import random

# Constants
NUM_PARAMETERS = 4  # Number of parameters (e.g., temperature, feed rate, etc.)
POPULATION_SIZE = 50  # Number of individuals in the population
MAX_GENERATIONS = 100  # Number of generations
MUTATION_RATE = 0.01  # Probability of mutation
PARAMETER_RANGES = np.array([[100, 300],  # Inlet air temperature range
                             [0.1, 1.0],  # Feed rate range
                             [200, 500],  # Air flow rate range
                             [1000, 3000]])  # Atomizer speed range

# Neural network simulation
# Fitness Function
def predict_nn(parameters):
    # Simulate NN predictions. In practice, you would replace this with actual model predictions.
    # For simplicity, let's assume better performance is when parameters are closer to the middle of their range.
    ideal = np.mean(PARAMETER_RANGES, axis=1)
    return -np.sum((parameters - ideal) ** 2, axis=1)  # Negative squared distance from the ideal point

# Genetic Algorithm Components
def initialize_population(size):
    return np.random.uniform(low=PARAMETER_RANGES[:, 0], high=PARAMETER_RANGES[:, 1], size=(size, NUM_PARAMETERS))

def evaluate_fitness(population):
    return predict_nn(population)

def select(population, fitness, num_parents):
    # Tournament selection
    parents = np.zeros((num_parents, NUM_PARAMETERS))
    for i in range(num_parents):
        random_ids = np.random.randint(0, len(population), 4)
        best_id = random_ids[np.argmax(fitness[random_ids])]
        parents[i] = population[best_id]
    return parents

def crossover(parents):
    offspring = np.zeros((POPULATION_SIZE, NUM_PARAMETERS))
    crossover_point = np.random.randint(1, NUM_PARAMETERS-1)
    for i in range(POPULATION_SIZE):
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        offspring[i, :crossover_point] = parents[parent1_index, :crossover_point]
        offspring[i, crossover_point:] = parents[parent2_index, crossover_point:]
    return offspring

def mutate(offspring):
    for idx in range(offspring.shape[0]):
        for jdx in range(NUM_PARAMETERS):
            if np.random.rand() < MUTATION_RATE:
                random_value = np.random.uniform(PARAMETER_RANGES[jdx, 0], PARAMETER_RANGES[jdx, 1])
                offspring[idx, jdx] = random_value
    return offspring

def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        fitness = evaluate_fitness(population)
        parents = select(population, fitness, POPULATION_SIZE // 2)
        offspring = crossover(parents)
        population = mutate(offspring)
    best_index = np.argmax(evaluate_fitness(population))
    return population[best_index]

# Run the genetic algorithm
best_parameters = genetic_algorithm()
print("Optimized Parameters:", best_parameters)


'''
Initialize Population
    Generates random initial solutions within the specified ranges for each parameter.
Evaluate Fitness
    Uses the predict_nn function to calculate how good each solution is.
Selection
    Chooses the best individuals from the population to be parents for the next generation. It uses a tournament selection process, where groups of individuals compete to be selected.
Crossover
    Combines parameters from two parents to create new solutions. It helps to mix good characteristics from different solutions.
Mutation
    Introduces random changes to some parameters, which helps to explore new areas of the solution space and prevents the algorithm from getting stuck in a local optimum.
'''


# In[ ]:




