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
from step7.Ecommerce7 import *


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

    num_sold_items = estimate_nodes_activation_probabilities(
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

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)


        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)

        num_sold_items = estimate_nodes_activation_probabilities(
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

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)



        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm4_gpts = Ecommerce4('TS',B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm4_gpucb = Ecommerce4('UCB', B_cap, budgets, product_prices, gp_hyperparameters)
        
        num_sold_items = estimate_nodes_activation_probabilities(
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
            alpha, gpts_gains_per_experiment[e][t], sold_items = env.round_step4(pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items = num_sold_items, optimal_arm=optimal_allocation_idxs)
            ecomm4_gpts.update(arm_idxs, alpha, sold_items)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm4_gpucb.pull_arm()
            alpha, gpucb_gains_per_experiment[e][t], sold_items = env.round_step4(pulled_arm=arm, pulled_arm_idxs=arm_idxs, num_sold_items = num_sold_items, optimal_arm=optimal_allocation_idxs)
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

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step5']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)


        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm5_gpucb = Ecommerce5_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)

        num_sold_items = estimate_nodes_activation_probabilities(
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
            
            # arm, arm_idxs = ecomm5_gpts.pull_arm()
            # reward_per_arm, gpts_gains_per_experiment[e][t] = env.round_step5(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            # ecomm5_gpts.update(arm_idxs, reward_per_arm)
            # log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

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


    gp_hyperparameters = hyperparams['step3']

    tau = np.ceil(3.0 * np.sqrt(T)).astype(int)

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
        ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, gp_hyperparameters, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, gp_hyperparameters, M, eps, h)

        
        current_phase = -1

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):

            new_phase = env.get_current_phase()
            if new_phase != current_phase :
                print('environment changing_phase')
                current_phase = new_phase

                num_sold_items = estimate_nodes_activation_probabilities(
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
                                                                                    num_sold_items = num_sold_items, optimal_arm = optimal_allocation_idxs, end_round = True)
            ecomm6_swucb.update(arm_idxs, alpha, sold_items)
            
                                                                    
            # arm, arm_idxs = ecomm6_cducb.pull_arm()
            # alpha, cducb_gains_per_experiment[e][t], sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
            #                                                                         num_sold_items = num_sold_items, optimal_arm = optimal_allocation_idxs, end_round = True)
            # ecomm6_cducb.update(arm_idxs, alpha, sold_items)


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

    gp_hyperparameters = hyperparams['step3']

    features = hyperparams['step7']['features']
    split_time = hyperparams['step7']['split_time']
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env = Environment(users_reservation_prices, graph_weights, alpha_bars, users_poisson_parameters)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm7_gpts = Ecommerce7('TS', B_cap, budgets, product_prices, gp_hyperparameters, features)
        ecomm7_gpucb = Ecommerce7('UCB', B_cap, budgets, product_prices, gp_hyperparameters, features)


        num_sold_items = estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            env.users_poisson_parameters,
            product_prices,
            observations_probabilities
        )


        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            # ------------------- TS --------------------------------------------

            context_learners = ecomm7_gpts.get_context_tree().get_leaves()
            idxs = []
            pulled_arms = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

            if t % split_time == 0 and t != 0:
                print('------- Thompson_Sampling splitting evaluation --------')
                for learner in context_learners:
                    learner.evaluate_splitting_condition(features,
                                                         ecomm7_gpts.get_pulled_arms(
                                                         )[-split_time:],
                                                         ecomm7_gpts.get_collected_rewards(
                                                         )[-split_time:],
                                                         ecomm7_gpts.get_collected_sold_items()[-split_time:])

                context_learners = ecomm7_gpts.get_context_tree().get_leaves()

            for learner in context_learners:
                arm, arm_idxs, learner_idxs = learner.pull_arm()

                # avoid_class_overlapping
                for idx in learner_idxs:
                    if not np.array_equal(pulled_arms[idx, :], np.zeros(shape=NUM_OF_PRODUCTS)):
                        learner_idxs.remove(idx)

                idxs.append(learner_idxs)
                pulled_arms[learner_idxs, :] = arm

            reward, estimated_sold_items = env.round_step7(
                pulled_arms, B_cap, nodes_activation_probabilities, num_sold_items)

            for i in range(len(context_learners)):
                context_learners[i].update(
                    pulled_arms[idxs[i][0]],
                    np.sum(reward[idxs[i]], axis=0),
                    np.sum(estimated_sold_items[idxs[i]], axis=0)
                )

            ecomm7_gpts.update_history(pulled_arms, reward, estimated_sold_items)
            for learner in context_learners:
                _, rew = learner.algorithm.solve_optimization_problem(nodes_activation_probabilities)
                gpts_gains_per_experiment[e][t] += rew

            # ------------------------------ UCB ----------------------------

            context_learners = ecomm7_gpucb.get_context_tree().get_leaves()
            idxs = []
            pulled_arms = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

            if t % split_time == 0 and t != 0:
                print('------- UCB splitting evaluation --------')
                for learner in context_learners:
                    learner.evaluate_splitting_condition(features,
                                                         ecomm7_gpucb.get_pulled_arms(
                                                         )[-split_time:],
                                                         ecomm7_gpucb.get_collected_rewards(
                                                         )[-split_time:],
                                                         ecomm7_gpucb.get_collected_sold_items()[-split_time:])

                context_learners = ecomm7_gpucb.get_context_tree().get_leaves()

            for learner in context_learners:
                arm, learner_idxs = learner.pull_arm()

                # avoid_class_overlapping
                for idx in learner_idxs:
                    if not np.array_equal(pulled_arms[idx, :], np.zeros(shape=NUM_OF_PRODUCTS)):
                        learner_idxs.remove(idx)

                idxs.append(learner_idxs)
                pulled_arms[learner_idxs, :] = arm

            reward, estimated_sold_items = env.round_step7(
                pulled_arms, B_cap, nodes_activation_probabilities, num_sold_items)

            for i in range(len(context_learners)):
                context_learners[i].update(
                    pulled_arms[idxs[i][0]],
                    np.sum(reward[idxs[i]], axis=0),
                    np.sum(estimated_sold_items[idxs[i]], axis=0)
                )

            ecomm7_gpucb.update_history(pulled_arms, reward, estimated_sold_items)
            for learner in context_learners:
                _, rew = learner.algorithm.solve_optimization_problem(nodes_activation_probabilities)
                gpucb_gains_per_experiment[e][t] += rew

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment


    # -----------STEP 3------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, gpts_max_variance_per_experiment, gpucb_max_variance_per_experiment = simulate_step3()
    plot_regrets(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, gpts_max_variance_per_experiment,
                 gpucb_max_variance_per_experiment, ["GPTS", "GPUCB"])

    # -----------STEP 4------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = simulate_step4()
    plot_regrets(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, ["GPTS", "GPUCB"])

    # -----------STEP 5------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = simulate_step5()
    plot_regrets(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, ["GPTS", "GPUCB"])

    # -----------STEP 6------------
    swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts = simulate_step6()
    plot_regrets(swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts, ["SWUCB", "CDUCB"])

    # -----------STEP 7------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = simulate_step7()
    plot_regrets(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, ["GPTS", "GPUCB"])


if __name__ == "__main__":
    simulate_step6()

    
