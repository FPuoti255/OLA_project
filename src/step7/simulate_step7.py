import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "step7"))

import numpy as np
from tqdm import *
import json

from Utils import *
from constants import *

from Social_influence import *
from Environment import *
from Scenario import *

from Ecommerce import *
from Ecommerce7 import *


def simulate_step7():
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T']
    n_experiments = hyperparams['simulation']['n_experiments']

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))


    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    gp_hyperparameters = hyperparams['step3']

    features = hyperparams['step7']['features']
    split_time = hyperparams['step7']['split_time']
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm7_gpts = Ecommerce7(B_cap, budgets, product_prices, 'TS', gp_hyperparameters, features, split_time)
        ecomm7_gpucb = Ecommerce7(B_cap, budgets, product_prices, 'UCB',  gp_hyperparameters, features, split_time)


        _, num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )


        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            expected_reward = env.compute_disaggregated_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_allocation_idxs, optimal_gain[e][t] = ecomm.clairvoyant_disaggregated_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')

            # ------------------- TS --------------------------------------------

            
            arm, arm_idxs = ecomm7_gpts.pull_arm()
            alpha, reward, real_sold_items = env.round_step7(arm, arm_idxs, num_sold_items)
            ecomm7_gpts.update(arm_idxs, alpha, real_sold_items)
            gpts_gains_per_experiment[e][t] += reward



                

            # ------------------------------ UCB ----------------------------


    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain
