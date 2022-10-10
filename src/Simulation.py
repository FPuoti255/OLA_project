import random

import numpy as np
from tqdm import *

from Utils import *
from constants import *

from Social_influence import *
from Environment import *
from Scenario import *
from Non_Stationary_Environment import *

from Ecommerce import *
from Ecommerce3 import *
from Ecommerce4 import *
from Ecommerce5 import *
from Ecommerce6 import *
from step7.Ecommerce7 import *


def setup_non_stationaty_environment():

    nodes_activation_probabilities = []
    num_sold_items = []
    users_reservation_prices = []
    product_functions_idxs = []   
    alpha_bars = [] 
    users_poisson_parameters = []
    prod_fun_idx = np.arange(NUM_OF_PRODUCTS)

    graph_weights = generate_graph_weights()
    observations_probabilities = generate_observation_probabilities(graph_weights)
    product_prices = np.round(np.random.random(size=NUM_OF_PRODUCTS) * products_price_range, 2)
    scenario = Scenario()

    for _ in range(n_phases):

        alpha_bar, single_users_poisson_parameters = generate_users_parameters(scenario.users_reservation_prices)
        users_reservation_prices.append(scenario.users_reservation_prices)
        users_poisson_parameters.append(single_users_poisson_parameters)
        alpha_bars.append(alpha_bar)

        estimation = estimate_nodes_activation_probabilities(
            graph_weights,
            scenario.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )
        nodes_activation_probabilities.append(estimation[0])
        num_sold_items.append(estimation[1])

        np.random.shuffle(prod_fun_idx)  # In place shuffling
        product_functions_idxs.append(prod_fun_idx.copy())

    env = Non_Stationary_Environment(
        users_reservation_prices, product_functions_idxs, graph_weights,
        alpha_bars, num_sold_items, nodes_activation_probabilities, users_poisson_parameters, T
    )

    return graph_weights, nodes_activation_probabilities, alpha_bars, product_prices, num_sold_items, product_functions_idxs, scenario.users_reservation_prices, users_poisson_parameters


def generate_new_non_stationary_environment():
    '''
    :return: env, observations_probabilities, graph_weights, product_prices, num_sold_items, nodes_activation_probabilities
    '''

    graph_weights = generate_graph_weights()
    observations_probabilities = generate_observation_probabilities(graph_weights)
    product_prices = np.round(np.random.random(size=NUM_OF_PRODUCTS) * products_price_range, 2)

    users_alpha = []
    users_reservation_prices = []
    nodes_activation_probabilities = []
    num_sold_items = []
    product_functions_idxs = []
    users_poisson_parameters = []
    prod_fun_idx = np.arange(NUM_OF_PRODUCTS)

    scenario = Scenario()

    for _ in range(n_phases):

        alpha_bars, _ = generate_users_parameters(scenario.users_reservation_prices)

        users_alpha.append(alpha_bars)
        users_reservation_prices.append(scenario.users_reservation_prices)
        users_poisson_parameters.append(users_poisson_parameters)

        estimation = estimate_nodes_activation_probabilities(
            graph_weights,
            scenario.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )
        nodes_activation_probabilities.append(estimation[0])
        num_sold_items.append(estimation[1])

        np.random.shuffle(prod_fun_idx)  # In place shuffling
        product_functions_idxs.append(prod_fun_idx.copy())

    env = Non_Stationary_Environment(
        users_reservation_prices, product_functions_idxs, graph_weights,
        users_alpha, num_sold_items, nodes_activation_probabilities, users_poisson_parameters, T
    )

    # Network.print_graph(G=env.network.G)

    return env, observations_probabilities, product_prices


def observe_learned_functions():

    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
        observations_probabilities, users_poisson_parameters = setup_environment()

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices)
    ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices)

    for t in tqdm(range(0, T), position = 0, desc="n_iteration"):
    # Every day a new montecarlo simulation must be run to sample num of items sold
        num_sold_items = estimate_nodes_activation_probabilities2(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        _ = env.compute_clairvoyant_reward(
            num_sold_items,
            product_prices,
            budgets
        )

        # aggregation is needed since in this step the ecommerce
        # cannot observe the users classes features
        aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)

        arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
        # the environment returns the users_alpha and the reward for that allocation
        alpha, _ = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        ecomm3_gpts.update(arm_idxs, alpha)

        arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)
        alpha, _ = env.round_step3(pulled_arm = arm, pulled_arm_idxs = arm_idxs)
        ecomm3_gpucb.update(arm_idxs, alpha)

    return ecomm3_gpts, ecomm3_gpucb, env


def simulate_step3():
    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        scenario = Scenario()
        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                    observations_probabilities, users_poisson_parameters = scenario.setup_environment()


        env = Environment(users_reservation_prices, graph_weights, alpha_bars)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices)

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            # Every day a new montecarlo simulation must be run to sample num of items sold
            num_sold_items = estimate_nodes_activation_probabilities(
                env.network.get_adjacency_matrix(),
                env.users_reservation_prices,
                users_poisson_parameters,
                product_prices,
                observations_probabilities
            )
            # print('sold items')
            # print(num_sold_items)
            # for c in range(num_sold_items.shape[0]):
            #     for p in range(num_sold_items.shape[1]):
            #         num_sold_items[c][p] = num_sold_items[c][p] * random.choice([1,2])

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )
            # print('clairvoyantt expected reward')
            # print(expected_reward)

            optimal_allocation, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)


            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')

            # aggregation is needed since in this step the ecommerce cannot observe the users classes features
            aggregated_num_sold_items = np.sum(num_sold_items, axis=0)
            # print('agg num sold items')
            # print(aggregated_num_sold_items)

            arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
            # # the environment returns the users_alpha and the reward for that allocation
            alpha, gpts_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm3_gpts.update(arm_idxs, alpha)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)

            alpha, gpucb_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm3_gpucb.update(arm_idxs, alpha)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')
            #if optimal_allocation == arm:
            #   log("OPTIMAL PULLED")
            log('-'*100)

        #print('opt - ucb')
        #print(optimal_gain - gpucb_gains_per_experiment)
        #print('opt - ts')
        #print(optimal_gain - gpts_gains_per_experiment)
    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step4():
    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        env, observations_probabilities, product_prices, users_poisson_parameters = setup_environment()

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm4_gpts = Ecommerce4_GPTS(B_cap, budgets, product_prices)
        ecomm4_gpucb = Ecommerce4_GPUCB(B_cap, budgets, product_prices)

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
            env.graph_weights,
            env.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        exp_clicks = env.estimate_num_of_clicks(budgets / B_cap)
        _, optimal_gain_per_experiment[e] = ecomm.solve_optimization_problem(
            num_sold_items,
            exp_clicks,
            nodes_activation_probabilities
        )

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):
            arm = ecomm4_gpts.pull_arm()
            reward, estimated_sold_items = env.round_step4(arm, B_cap, num_sold_items)
            ecomm4_gpts.update(arm, reward, estimated_sold_items)
            _, gpts_gains_per_experiment[e][t] = ecomm4_gpts.solve_optimization_problem(nodes_activation_probabilities)

            arm = ecomm4_gpucb.pull_arm()
            reward, estimated_sold_items = env.round_step4(arm, B_cap, num_sold_items)
            ecomm4_gpucb.update(arm, reward, estimated_sold_items)
            _, gpucb_gains_per_experiment[e][t] = ecomm4_gpucb.solve_optimization_problem(
                nodes_activation_probabilities)

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step5():
    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        env, observations_probabilities, product_prices, users_poisson_parameters = setup_environment()

        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices)
        ecomm5_gpucb = Ecommerce5_GPUCB(B_cap, budgets, product_prices)

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
            env.graph_weights,
            env.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )
        exp_clicks = env.estimate_num_of_clicks(budgets / B_cap)

        ecomm = Ecommerce(B_cap, budgets, product_prices)

        _, optimal_gain_per_experiment[e] = ecomm.solve_optimization_problem(
            num_sold_items,
            exp_clicks,
            nodes_activation_probabilities
        )

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):
            arm, arm_idx = ecomm5_gpts.pull_arm()
            reward = env.round_step5(arm, nodes_activation_probabilities)
            ecomm5_gpts.update(arm, arm_idx, reward)

            _, gain = ecomm5_gpts.solve_optimization_problem(
                num_sold_items,
                exp_clicks
            )

            gpts_gains_per_experiment[e][t] = np.minimum(gain, optimal_gain_per_experiment[e])
            # gpts_gains_per_experiment[e][t] = gain

            # ----------------------

            arm, arm_idx = ecomm5_gpucb.pull_arm()
            reward = env.round_step5(arm, nodes_activation_probabilities)
            ecomm5_gpucb.update(arm, arm_idx, reward)

            _, gain = ecomm5_gpucb.solve_optimization_problem(
                num_sold_items,
                exp_clicks
            )
            gpucb_gains_per_experiment[e][t] = np.minimum(gain, optimal_gain_per_experiment[e])
            # gpucb_gains_per_experiment[e][t] = gain

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step6():
    swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T))

    tau = np.floor(np.sqrt(T)).astype(int)

    M = np.ceil(0.033 * T)
    eps = 0.1
    h = 2 * np.log(T)

    for e in range(0, n_experiments):
        print('Experiment n°', e + 1)

        graph_weights, nodes_activation_probabilities, alpha_bars, product_prices, num_sold_items, product_functions_idxs, users_reservation_prices, users_poisson_parameters = setup_non_stationaty_environment()

        env = Non_Stationary_Environment(
            users_reservation_prices, product_functions_idxs, graph_weights,
            alpha_bars, num_sold_items, nodes_activation_probabilities, users_poisson_parameters, T
        )

        ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, M, eps, h)

        optimal_phase_gain = 0

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):

            exp_clicks = env.estimate_num_of_clicks(budgets / B_cap)
            ecomm = Ecommerce(B_cap, budgets, product_prices)

            if t % phase_len == 0:
                _, optimal_phase_gain = ecomm.solve_optimization_problem(
                    env.get_num_sold_items(),
                    exp_clicks,
                    env.get_nodes_activation_probabilities()
                )

            optimal_gain_per_experiment[e][t] = optimal_phase_gain

            arm = ecomm6_swucb.pull_arm()
            reward, sold_items = env.round_step6(arm, B_cap)
            ecomm6_swucb.update(arm, reward, sold_items)
            _, swucb_gains_per_experiment[e][t] = ecomm6_swucb.solve_optimization_problem(
                env.get_nodes_activation_probabilities())

            arm = ecomm6_cducb.pull_arm()
            reward, sold_items = env.round_step6(arm, B_cap, True)
            ecomm6_cducb.update(arm, reward, sold_items)
            _, cducb_gains_per_experiment[e][t] = ecomm6_cducb.solve_optimization_problem(
                env.get_nodes_activation_probabilities())

    return swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step7():
    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env, observations_probabilities, product_prices, users_poisson_parameters = setup_environment()

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
            env.graph_weights,
            env.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        exp_num_clicks = env.estimate_disaggregated_num_clicks(budgets / B_cap)

        ecomm7_gpts = Ecommerce7(B_cap, budgets, product_prices, features, 'TS')
        ecomm7_gpucb = Ecommerce7(B_cap, budgets, product_prices, features, 'UCB')

        _, optimal_gain_per_experiment[e] = ecomm7_gpts.clairvoyant_solve_optimization_problem(num_sold_items,
                                                                                               exp_num_clicks,
                                                                                               nodes_activation_probabilities)

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
    main()

    
