import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Simulation import *
from Ecommerce import *

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

N_pop = 40
hyperparameters_len = 3

# HYPERPARAMETERS
alpha = 1.0
# RBF Kernel
length_scale = 1.0                      # float       
# Const Kernel
constant_value = 1.0  

# Constraints
length_scale_bounds1 = 1e-3
length_scale_bounds2 = 1e3              
constant_value_bounds1 = 1e-5         
constant_value_bounds2 = 1e5          

kernel = C(length_scale,(length_scale_bounds1,length_scale_bounds2)) * RBF(constant_value,(constant_value_bounds1,constant_value_bounds2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9)


def initialize_population():
    '''
    Step 1:
        Initialization. An initial population of N_pop chromosomes is created is constructed by random generation.
    '''
    N_pop = 40
    alphas = np.random.uniform(low=1e-1, high=20, size=N_pop)
    length_scales = np.random.uniform(low=length_scale_bounds1, high=length_scale_bounds2, size=N_pop)
    constant_values = np.random.uniform(low=constant_value_bounds1, high=constant_value_bounds2, size=N_pop)

    return alphas, length_scales, constant_values


def crossover(population):
    '''
    Step 2a:
        Crossover: N_pop randomly selected chromosome pairs exchange randomly chosen components to create offspring.

    :param: population: tuples of hyperparameters
    '''
    random.shuffle(population)
    pairs = [population[i*2: (i+1)*2] for i in range(int(len(population)/2))]
    
    for candidate1, candidate2 in pairs:
        candidate1[1], candidate2[1] = candidate2[1], candidate1[1]

    return population
        

def mutation(population):
    '''
    Step 2b:
        Mutation: a part of the population undergoes mutation, which can be modeled as a chromosome crossover 
        with a random chromosome.
   
    :param: population: tuples of hyperparameters
    '''
    for candidate in population:
        if random.choice([0, 1]):   # comment if we want to mutate all the candidates
            rand_idx = np.random.randint(0,hyperparameters_len)
            candidate2_idx = np.random.randint(0,len(population))
            print(f"Mutation!\n candidate {candidate} swapped element at idx {rand_idx} with candidate {population[candidate2_idx]}")
            candidate[rand_idx] = population[candidate2_idx][rand_idx]

    N_pop /= 2
    return population   


def ranking(population, regrets):
    '''
    Step 3. 
        Ranking: chromosomes are ranked based on the corresponding values of the fitting function.
    '''
    mapping = zip(population,fitness_function(regrets))

    ranking = sorted(mapping, key = lambda mapping: mapping[1]) 
    population = [cand[0] for cand in ranking]
    regrets = [cand[1] for cand in ranking]

    return population, regrets


def perturbation(population, regrets):
    '''
    Step 4.
        Perturbation: adding 5-10 little random changes to components of highest-ranking chromosome to see 
        if it increases the fitting function value. If yes, that chromosome is replaced by the perturbed chromosome. 
    '''
    population, regrets = ranking(population, regrets)
    tops = int(len(population)/4)
    for top in tops:
        candidate = population[top]
        for param_idx in hyperparameters_len:
            perturbation = np.random.uniform(low=-candidate[param_idx]/5,high=candidate[param_idx]/5,size=1)
            new_param = candidate[param_idx] + perturbation
            if fitness_function(run(new_param)) > regrets[top]:
                candidate[param_idx] = new_param


def selection(population, regrets):
    '''
    Step 5.
        Selection: from the whole population, only N_pop highest-ranking chromosomes are retained
        THEN --> step2
    '''
    population, regrets = ranking(population, regrets)
    return population[:N_pop]


def fitness_function(regrets):
    '''
    choices:
        1. Euclidian metric for closeness between the modeled and original function in the case of regression 
           (Basically it's the MSE = REGRET)
        2. marginal likelihood
    goal: maximize
    '''
    return regrets


def step3_genetic_algorithm(initial_kernel_params):

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))    
    optimal_gain = np.zeros(shape=(n_experiments, T))

    for e in range(0, n_experiments):
        print('Experiment n°:', e+1)

        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
            observations_probabilities, users_poisson_parameters = setup_environment()

        env = Environment(users_reservation_prices, graph_weights, alpha_bars)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, initial_kernel_params=initial_kernel_params)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, initial_kernel_params=initial_kernel_params)

        for t in tqdm(range(0, T), position = 0, desc="n_iteration"):
            # Every day a new montecarlo simulation must be run to sample num of items sold
            num_sold_items = estimate_nodes_activation_probabilities(
                env.network.get_adjacency_matrix(),
                env.users_reservation_prices,
                users_poisson_parameters,
                product_prices,
                observations_probabilities
            )
            
            log("num_sold_items:\n")
            log(num_sold_items)
            log("\n\n")

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )     
            log("expected_reward:\n")
            log(expected_reward)
            log("\n\n")       

            optimal_allocation , optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)

            print(f'optimal_allocation: {optimal_allocation}, reward: {optimal_gain[e][t]}')

            # aggregation is needed since in this step the ecommerce
            # cannot observe the users classes features
            aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)

            arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
            # the environment returns the users_alpha and the reward for that allocation
            alpha, gpts_gains_per_experiment[e][t] = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
            ecomm3_gpts.update(arm_idxs, alpha)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)
            alpha, gpucb_gains_per_experiment[e][t] = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
            ecomm3_gpucb.update(arm_idxs, alpha)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')
            log('-------------------------------------------------')
    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def main():
    
    initial_kernal_params = initialize_population()

    step3_genetic_algorithm(initial_kernal_params)


if __name__ == "main":
    main()    