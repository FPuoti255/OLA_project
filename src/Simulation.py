import numpy as np
from tqdm import *
import json

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
#from step7.Ecommerce7 import *


def observe_learned_functions():

    scenario = Scenario()
    graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                observations_probabilities, users_poisson_parameters = scenario.setup_environment()

    env = Environment(users_reservation_prices, graph_weights, alpha_bars)
    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']
    ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
    ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)


    for t in tqdm(range(0, T), position = 0, desc="n_iteration"):
    # Every day a new montecarlo simulation must be run to sample num of items sold
        num_sold_items = np.random.randint(10,30,size=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))/10
        """estimate_nodes_activation_probabilities(
            env.network.get_adjacency_matrix(),
            env.users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )"""

        # aggregation is needed since in this step the ecommerce
        # cannot observe the users classes features
        aggregated_num_sold_items = np.sum(num_sold_items, axis = 0)

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
    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        scenario = Scenario()
        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                    observations_probabilities, users_poisson_parameters = scenario.setup_environment()

        env = Environment(users_reservation_prices, graph_weights, alpha_bars)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            # Every day a new montecarlo simulation must be run to sample num of items sold
            num_sold_items = np.maximum(
                np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )
            # num_sold_items = estimate_nodes_activation_probabilities(
            #     env.network.get_adjacency_matrix(),
            #     env.users_reservation_prices,
            #     users_poisson_parameters,
            #     product_prices,
            #     observations_probabilities
            # )

            # aggregation is needed since in this step the ecommerce cannot observe the users classes features
            aggregated_num_sold_items = np.sum(num_sold_items, axis=0)

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')
            
            arm, arm_idxs = ecomm3_gpts.pull_arm(aggregated_num_sold_items)
            alpha, gpts_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm3_gpts.update(arm_idxs, alpha)
            log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            # arm, arm_idxs = ecomm3_gpucb.pull_arm(aggregated_num_sold_items)
            # alpha, gpucb_gains_per_experiment[e][t] = env.round_step3(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            # ecomm3_gpucb.update(arm_idxs, alpha)
            # log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')


            # if optimal_allocation == arm:
            # #   log("OPTIMAL PULLED")
            # log('-'*100)

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step4():
    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step3']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        scenario = Scenario()
        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                    observations_probabilities, users_poisson_parameters = scenario.setup_environment()


        env = Environment(users_reservation_prices, graph_weights, alpha_bars)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm4_gpts = Ecommerce4('TS',B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm4_gpucb = Ecommerce4('UCB', B_cap, budgets, product_prices, gp_hyperparameters)
        
        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):
            
            num_sold_items = estimate_nodes_activation_probabilities(
                env.network.get_adjacency_matrix(),
                env.users_reservation_prices,
                users_poisson_parameters,
                product_prices,
                observations_probabilities
            )

            # num_sold_items = np.maximum(
            #     np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
            #     0
            # )
            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
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

            log('-'*100)

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step5():
    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    optimal_gain = np.zeros(shape=(n_experiments, T))

    gp_hyperparameters = json.load(open("hyperparameters.json"))['step5']

    for e in range(0, n_experiments):
        print('Experiment n°:', e + 1)

        scenario = Scenario()
        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
                    observations_probabilities, users_poisson_parameters = scenario.setup_environment()

        env = Environment(users_reservation_prices, graph_weights, alpha_bars)

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices, gp_hyperparameters)
        ecomm5_gpucb = Ecommerce5_GPUCB(B_cap, budgets, product_prices, gp_hyperparameters)

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            # Every day a new montecarlo simulation must be run to sample num of items sold
            num_sold_items = np.maximum(
                np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )
            # num_sold_items = estimate_nodes_activation_probabilities(
            #     env.network.get_adjacency_matrix(),
            #     env.users_reservation_prices,
            #     users_poisson_parameters,
            #     product_prices,
            #     observations_probabilities
            # )

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')
            
            # arm, arm_idxs = ecomm5_gpts.pull_arm()
            # reward_per_arm, gpts_gains_per_experiment[e][t] = env.round_step5(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            # ecomm5_gpts.update(arm_idxs, reward_per_arm)
            # log(f'gpts pulled_arm: {arm}, reward : {gpts_gains_per_experiment[e][t]}')

            arm, arm_idxs = ecomm5_gpucb.pull_arm()
            reward_per_arm, gpucb_gains_per_experiment[e][t] = env.round_step5(pulled_arm=arm, pulled_arm_idxs=arm_idxs)
            ecomm5_gpucb.update(arm_idxs, reward_per_arm)
            log(f'ucb pulled_arm: {arm}, reward: {gpucb_gains_per_experiment[e][t]}')


            # if optimal_allocation == arm:
            # #   log("OPTIMAL PULLED")
            # log('-'*100)

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain


def simulate_step6():
    
    swucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    cducb_gains_per_experiment = np.zeros(shape=(n_experiments, T))    
    optimal_gain = np.zeros(shape=(n_experiments, T))
    

    tau = np.floor(np.sqrt(T)).astype(int)
    M = np.ceil(0.033 * T)
    eps = 0.1
    h = 2 * np.log(T)

    for e in range(0, n_experiments):
        print('Experiment n°', e + 1)
        
        non_stationary_scenario = NonStationaryScenario()

        graph_weights, alpha_bars, product_prices, users_reservation_prices, \
             observations_probabilities, users_poisson_parameters = non_stationary_scenario.setup_environment()

        
        env = Non_Stationary_Environment(
            users_reservation_prices, 
            graph_weights, 
            alpha_bars,
            users_poisson_parameters,
            non_stationary_scenario.get_n_phases(),
            non_stationary_scenario.get_phase_len()
        )

        ecomm = Ecommerce(B_cap, budgets, product_prices)
        ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, M, eps, h)


        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=True):

            # num_sold_items = estimate_nodes_activation_probabilities(
            #     env.get_network().get_adjacency_matrix(),
            #     env.get_users_reservation_prices(),
            #     env.get_users_poisson_parameters(),
            #     product_prices,
            #     observations_probabilities[env.get_current_phase()]
            # )

            num_sold_items = np.maximum(
                np.random.normal(loc = 4, scale = 2, size = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)),
                0
            )

            expected_reward = env.compute_clairvoyant_reward(
                num_sold_items,
                product_prices,
                budgets
            )

            optimal_allocation, optimal_gain[e][t] = ecomm.clairvoyant_optimization_problem(expected_reward)
            log(f'optimal_allocation: \t{optimal_allocation}, \treward : \t{optimal_gain[e][t]}')

            arm, arm_idxs = ecomm6_swucb.pull_arm()
            alpha,swucb_gains_per_experiment[e][t] , sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                    num_sold_items = num_sold_items)
            ecomm6_swucb.update(arm_idxs, alpha, sold_items)

            arm, arm_idxs = ecomm6_cducb.pull_arm()
            alpha, cducb_gains_per_experiment[e][t], sold_items = env.round_step6(pulled_arm=arm, pulled_arm_idxs=arm_idxs, 
                                                                                    num_sold_items = num_sold_items, end_phase = True)
            ecomm6_cducb.update(arm_idxs, alpha, sold_items)


    return swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain, non_stationary_scenario.get_n_phases(), non_stationary_scenario.get_phase_len()


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
    simulate_step6()

    
