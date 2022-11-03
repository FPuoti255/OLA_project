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
from Ecommerce3 import *
from Ecommerce4 import *
from Ecommerce5 import *
from Ecommerce6 import *
from Ecommerce7 import *


def observe_learned_functions():
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T']

    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)
    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
    ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)

    _, num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

    # aggregation is needed since in this step the ecommerce cannot observe the users classes features
    aggregated_num_sold_items = np.sum(num_sold_items, axis=0)

    for t in tqdm(range(0, T), position = 0, desc="n_iteration"):


        _ = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )


        arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
        # the environment returns the users_alpha and the reward for that allocation
        alpha, _ = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        ecomm3_gpts.update(arm_idxs, alpha)

        # arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)
        # alpha, _ = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        # ecomm3_gpucb.update(arm_idxs, alpha)

    return ecomm3_gpts, ecomm3_gpucb, env


def simulate_step3():
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T']
    n_experiments = hyperparams['simulation']['n_experiments']

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    gp_config = hyperparams['step3']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)


        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_config)

        _, num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        # aggregation is needed since in this step the ecommerce cannot observe the users classes features
        aggregated_num_sold_items = np.sum(num_sold_items, axis=0)

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, _ , optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')
            
            arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
            alpha, gpts_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm3_gpts.update(arm_idxs, alpha)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)
            alpha, gpucb_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm3_gpucb.update(arm_idxs, alpha)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step4():
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T']
    n_experiments = hyperparams['simulation']['n_experiments']

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    gp_config = hyperparams['step4']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)



        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm4_gpts = Ecommerce4('TS',B_cap, budgets, product_prices, gp_config)
        ecomm4_gpucb = Ecommerce4('UCB', B_cap, budgets, product_prices, gp_config)
        
        _, num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):
            

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_allocation_idxs, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')
            
            arm, arm_idxs = ecomm4_gpts.pull_arm()
            # the environment returns the users_alpha and the reward for that allocation
            alpha, gpts_gains_per_experiment[e][t], sold_items = env.round_step4(pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items = num_sold_items)
            ecomm4_gpts.update(arm_idxs, alpha, sold_items)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')


            arm, arm_idxs = ecomm4_gpucb.pull_arm()
            alpha, gpucb_gains_per_experiment[e][t], sold_items = env.round_step4(pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items = num_sold_items)
            ecomm4_gpucb.update(arm_idxs, alpha, sold_items)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step5():
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T']
    n_experiments = hyperparams['simulation']['n_experiments']

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    gp_config = hyperparams['step5']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)


        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices, gp_config)
        ecomm5_gpucb = Ecommerce5_GPUCB(B_cap, budgets, product_prices, gp_config)

        _, num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, _, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')
            
            arm, arm_idxs = ecomm5_gpts.pull_arm()
            reward_per_arm, gpts_gains_per_experiment[e][t] = env.round_step5(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm5_gpts.update(arm_idxs, reward_per_arm)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm5_gpucb.pull_arm()
            reward_per_arm, gpucb_gains_per_experiment[e][t] = env.round_step5(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm5_gpucb.update(arm_idxs, reward_per_arm)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step6():
    
    hyperparams = json.load(open("hyperparameters.json"))
    T = hyperparams['simulation']['T_step6']
    n_experiments = hyperparams['simulation']['n_experiments_step6']
    
    swucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    cducb_gains_per_experiment = np.zeros(shape=(n_experiments, T))    
    optimal_gain = np.zeros(shape=(n_experiments, T))
 
    non_stationary_scenario = NonStationaryScenario(T=T)

    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
            observations_probabilities, users_poisson_parameters = non_stationary_scenario.setup_environment()


    gp_config = hyperparams['step3']

    #tau = np.ceil(2.5 * np.sqrt(T)).astype(int)
    tau = np.ceil(2.5 * np.sqrt(T)).astype(int)
    M = hyperparams["step6"]["M"]
    eps = hyperparams["step6"]["eps"]
    h = hyperparams["step6"]["h"]

    print(f'tau : {tau}\nM, h, eps: {M}, {h}, {eps};\n')
    for e in range(0, n_experiments):
        print('Experiment n°', e + 1)
        

        
        env = Non_Stationary_Environment(
            users_reservation_prices, 
            graph_weights, 
            alpha_bars,
            users_poisson_parameters,
            non_stationary_scenario.get_n_phases(),
            non_stationary_scenario.get_phase_len()
        )

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, gp_config, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, gp_config, M, eps, h)

        
        current_phase = -1

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):

            new_phase = env.get_current_phase()
            if new_phase != current_phase :
                current_phase = new_phase

                _, num_sold_items = estimate_nodes_activation_probabilities(
                    env.get_network().get_adjacency_matrix(),
                    env.get_users_reservation_prices(),
                    env.get_users_poisson_parameters(),
                    product_prices,
                    observations_probabilities
                )


            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_allocation_idxs, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')

            arm, arm_idxs = ecomm6_swucb.pull_arm()
            alpha,swucb_gains_per_experiment[e][t] , sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                    num_sold_items = num_sold_items, end_round = False)
            ecomm6_swucb.update(arm_idxs, alpha, sold_items)
            
                                                                    
            arm, arm_idxs = ecomm6_cducb.pull_arm()
            alpha, cducb_gains_per_experiment[e][t], sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                    num_sold_items = num_sold_items, end_round = True)
            ecomm6_cducb.update(arm_idxs, alpha, sold_items)


    return swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain, non_stationary_scenario.get_n_phases(), non_stationary_scenario.get_phase_len()



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

    gp_config = hyperparams['step4']

    features = hyperparams['step7']['features']
    split_time = hyperparams['step7']['split_time']
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm7_gpts = Ecommerce7(B_cap, budgets, product_prices, 'TS', gp_config, features, split_time)
        ecomm7_gpucb = Ecommerce7(B_cap, budgets, product_prices, 'UCB',  gp_config, features, split_time)


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

            
            arm, arm_idxs = ecomm7_gpts.pull_arm()
            alpha, reward, real_sold_items = env.round_step7(arm, arm_idxs, num_sold_items)
            ecomm7_gpts.update(arm_idxs, alpha, real_sold_items)
            gpts_gains_per_experiment[e][t] += reward

            arm, arm_idxs = ecomm7_gpucb.pull_arm()
            alpha, reward, real_sold_items = env.round_step7(arm, arm_idxs, num_sold_items)
            ecomm7_gpucb.update(arm_idxs, alpha, real_sold_items)
            gpucb_gains_per_experiment[e][t] += reward

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


if __name__ == "__main__":
    simulate_step6()

    
