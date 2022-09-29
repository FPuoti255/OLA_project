import numpy as np

# https://www.osti.gov/servlets/purl/1659396

# ALGORITHM:
# step1
    #   step2
    #   step3
    #   step4
    #   step5

# The process stops if the maximum value of the fitting function for the population 
# does not change after N_max consecutive generations (cycles). Components of the corresponding 
# chromosome will be hyperparameters optimal for given GP kernel and training set


# HYPERPARAMETERS
alpha = 1.0
# RBF Kernel
length_scale1 = 1.0
length_scale_bounds1 = (1e-3,1e3)
# Const Kernel
constant_value = 1.0
constant_value_bounds = (1e-5, 1e5)


def initialize_population():
    '''
    Step 1:
        Initialization. An initial population of N_pop chromosomes is created is constructed 
        by random generation. The population size in this paper was chosen to be 40
    '''
    pass


def crossover():
    '''
    Step 2a:
        Crossover: N_pop randomly selected chromosome pairs exchange randomly chosen components to create offspring.
    '''
    pass


def mutation():
    '''
    Step 2b:
        Mutation: a part of the population undergoes mutation, which can be modeled as a chromosome crossover 
        with a random chromosome.
    '''
    pass


def ranking():
    '''
    Step 3. 
        Ranking: chromosomes are ranked based on the corresponding values of the fitting function.
    '''
    pass


def perturbation():
    '''
    Step 4.
        Perturbation: adding 5-10 little random changes to components of highest-ranking chromosome to see 
        if it increases the fitting function value. If yes, that chromosome is replaced by the perturbed chromosome. 
    '''
    pass


def selection():
    '''
    Step 5.
        Selection: from the whole population, only Npop highest-ranking chromosomes are retained
        THEN --> step2
    '''
    pass


def fitness_function(regret):
    '''
    choices:
        1. Euclidian metric for closeness between the modeled and original function in the case of regression (MSE)
        2. marginal likelihood
    goal: maximize
    '''
    pass
