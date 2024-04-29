#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cl3 8 DEAP 
import random
from deap import base, creator, tools

# Define the fitness function
#  function evaluates how good an "individual" is
def evalOneMax(individual):
    return (sum(individual),)  # Sum the number of 1s in the individual

# Set up the environment
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Objective: maximize the fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

'''
ToolBox is the first function. Used to initialize the initial population.
'''
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1) #A random attribute generator for our individuals, producing 0s or 1s.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 10)  # 10 genes per individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

'''
    Operators,
    mate operator executes a uniform crossover
    The mutation operator applies a gaussian mutation
    Then Parent Selection,
'''
toolbox.register("evaluate", evalOneMax) # Registers the fitness evaluation function.
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)  # 10% mutation probability
toolbox.register("select", tools.selTournament, tournsize=3)

# Create a population and evaluate it
population = toolbox.population(n=50)  # 50 individuals in the population
NGEN = 10  # Number of generations

for gen in range(NGEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    population = offspring

# Extracting and printing the best result
best_ind = tools.selBest(population, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


# In[ ]:




