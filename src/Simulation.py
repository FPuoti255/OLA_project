import numpy as np

from Utils import *
from constants import *

from Social_influence import *
from Environment import *
from Non_Stationary_Environment import *

from Ecommerce import *
from Ecommerce3 import *
from Ecommerce4 import *
from Ecommerce5 import *
from Ecommerce6 import *
from step7.Ecommerce7 import *

def generate_click_probabilities():
    '''
    :return: matrix representing the probability of going from a node to another
    '''

    adjacency_matrix = np.random.uniform(
        low=0.1, high=1.00001, size=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
    adjacency_matrix[np.diag_indices(n=NUM_OF_PRODUCTS, ndim=2)] = 0.0

    # set some values to zero is not fully connected, otherwise it's ready
    if not fully_connected:
        graph_mask = np.random.randint(
            low=0, high=2, size=adjacency_matrix.shape)
        adjacency_matrix = np.multiply(adjacency_matrix, graph_mask)

    adjacency_matrix = np.round(adjacency_matrix, 2)
    return adjacency_matrix


def generate_observation_probabilities(click_probabilities):
    '''
    :return: a random matrix representing the probability of observing from node i, when is primary, to node j, when it's in the secondaries.
             Probability is 1 for observing the first slot of the secondary product and LAMBDA for the second slot
    '''

    obs_prob = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

    for product in range(NUM_OF_PRODUCTS):

        available_products = [
            i
            for i in range(0, NUM_OF_PRODUCTS)
            if i != product and click_probabilities[product][i] != 0.0
        ]

        if len(available_products) >= 2:
            idxs = np.random.choice(
                a=available_products,
                size=max(2, len(available_products)),
                replace=False,
            )
            obs_prob[product][idxs[0]] = 1
            obs_prob[product][idxs[1]] = LAMBDA
        elif len(available_products) == 1:
            obs_prob[product][available_products[0]] = 1
        else:
            continue

    return obs_prob


def generate_users_parameters():
    '''
    :return: 
        - alphas represents the percentage of users (for each class) landing on a specific product webpage including the competitor's
        - users_reservation prices (NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS)
        - users_poisson_parameters = NUM_OF_USERS_CLASSES x NUM_OF_PRODUCTS matrix giving, 
                        for each users class and for each product, the poisson distribution of the bought items in the montecarlo sampling
    '''

    users_concentration_parameters = [
        np.clip(a=np.random.normal(loc=50, scale=30,
                size=NUM_OF_PRODUCTS + 1), a_min=1e-3, a_max=100),
        np.clip(a=np.random.normal(loc=75, scale=15,
                size=NUM_OF_PRODUCTS + 1), a_min=1e-3, a_max=100),
        np.clip(a=np.random.normal(loc=40, scale=40,
                size=NUM_OF_PRODUCTS + 1), a_min=1e-3, a_max=100)
    ]

    # N.B. the 𝛼_0 is the one corresponding to the competitor(s) product
    alphas = np.array(
        [np.random.dirichlet(alpha=users_concentration_parameters[i])
         for i in range(len(users_concentration_parameters))]
    )

    users_reservation_prices = np.array(
        [
            np.clip(a=np.random.normal(loc=50, scale=10,
                                       size=NUM_OF_PRODUCTS), a_min=30, a_max=100),
            np.clip(a=np.random.normal(loc=75, scale=10,
                                       size=NUM_OF_PRODUCTS), a_min=50, a_max=100),
            np.clip(a=np.random.normal(loc=40, scale=10,
                                       size=NUM_OF_PRODUCTS), a_min=15, a_max=100)
        ])

    max_expected_poisson_realization = 5

    users_poisson_parameters = np.array(
        [np.full(shape=NUM_OF_PRODUCTS, fill_value=max_expected_poisson_realization) * users_reservation_prices[user_class] / 100
         for user_class in range(NUM_OF_USERS_CLASSES)]
    )  # 3x5

    return alphas / NUM_OF_USERS_CLASSES, users_reservation_prices, users_poisson_parameters


def generate_new_environment():
    '''
    :return: env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices,  users_poisson_parameters
    '''

    click_probabilities = generate_click_probabilities()
    # Secondary product set by the business unit
    observations_probabilities = generate_observation_probabilities(
        click_probabilities)

    product_prices = np.round(np.random.random(
        size=NUM_OF_PRODUCTS) * products_price_range, 2)

    users_alpha, users_reservation_prices, users_poisson_parameters = generate_users_parameters()

    env = Environment(users_reservation_prices,
                      click_probabilities, users_alpha)

    # Network.print_graph(G=env.network.G)
    return env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices, users_poisson_parameters


def generate_new_non_stationary_environment():
    '''
    :return: env, observations_probabilities, click_probabilities, product_prices, num_sold_items, nodes_activation_probabilities
    '''

    click_probabilities = generate_click_probabilities()
    observations_probabilities = generate_observation_probabilities(
        click_probabilities)
    product_prices = np.round(np.random.random(
        size=NUM_OF_PRODUCTS) * products_price_range, 2)

    users_alpha = []
    users_reservation_prices = []
    users_poisson_parameters = []
    nodes_activation_probabilities = []
    num_sold_items = []
    product_functions_idxs = []
    prod_fun_idx = np.arange(NUM_OF_PRODUCTS)

    for _ in range(n_phases):

        alphas, res_prices, poisson_par = generate_users_parameters()

        users_alpha.append(alphas)
        users_reservation_prices.append(res_prices)
        users_poisson_parameters.append(poisson_par)

        estimation = estimate_nodes_activation_probabilities(
            click_probabilities,
            res_prices,
            poisson_par,
            product_prices,
            observations_probabilities
        )
        nodes_activation_probabilities.append(estimation[0])
        num_sold_items.append(estimation[1])

        np.random.shuffle(prod_fun_idx)  # In place shuffling
        product_functions_idxs.append(prod_fun_idx.copy())

    env = Non_Stationary_Environment(
        users_reservation_prices, product_functions_idxs, click_probabilities,
        users_alpha, num_sold_items, nodes_activation_probabilities, users_poisson_parameters, T
    )

    # Network.print_graph(G=env.network.G)

    return env, observations_probabilities, click_probabilities, product_prices


def simulate_step2():

    env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices, users_poisson_parameters = generate_new_environment()

    nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
        click_probabilities,
        users_reservation_prices,
        users_poisson_parameters,
        product_prices,
        observations_probabilities
    )
    exp_clicks = env.estimate_num_of_clicks(budgets/B_cap)

    optimal_allocation, optimal_gain = env.dummy_optimization_solver(
        budgets, B_cap, product_prices, num_sold_items, nodes_activation_probabilities, exp_clicks)

    print("optimal allocation is:", "".join(str(optimal_allocation)),
          "with a reward of:", int(optimal_gain))

    ecomm = Ecommerce(B_cap, budgets, product_prices)

    estimated_opt_allocation, estimated_opt_gain = ecomm.solve_optimization_problem(
        num_sold_items,
        exp_clicks,
        nodes_activation_probabilities
    )

    print("estimated_opt_allocation is:", "".join(
        str(estimated_opt_allocation)), "with a reward of:", int(estimated_opt_gain))


def simulate_step3():

    gpts_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpts_max_variance_per_experiment = np.zeros(shape=(n_experiments, T))

    gpucb_gains_per_experiment = np.zeros(shape=(n_experiments, T))
    gpucb_max_variance_per_experiment = np.zeros(shape=(n_experiments, T))
    
    optimal_gain_per_experiment = np.zeros(shape=(n_experiments))

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices, users_poisson_parameters = generate_new_environment()

        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices)
        ecomm3_gpucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices)

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
            click_probabilities,
            users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        exp_clicks = env.estimate_num_of_clicks(budgets/B_cap)
        ecomm = Ecommerce(B_cap, budgets, product_prices)

        _, optimal_gain_per_experiment[e] = ecomm.solve_optimization_problem(
            num_sold_items,
            exp_clicks,
            nodes_activation_probabilities
        )

        for t in tqdm(range(0, T), position = 0, desc="n_iteration", leave=False):

            arm = ecomm3_gpts.pull_arm()
            reward = env.round_step3(arm, B_cap)
            ecomm3_gpts.update(arm, reward)
            _, gpts_gains_per_experiment[e][t] = ecomm3_gpts.solve_optimization_problem(num_sold_items, nodes_activation_probabilities)
            gpts_max_variance_per_experiment[e][t] = ecomm3_gpts.get_max_gp_variance()

            arm = ecomm3_gpucb.pull_arm()
            reward = env.round_step3(arm, B_cap)
            ecomm3_gpucb.update(arm, reward)
            _,  gpucb_gains_per_experiment[e][t] = ecomm3_gpucb.solve_optimization_problem(num_sold_items, nodes_activation_probabilities)
            gpucb_max_variance_per_experiment[e][t] = ecomm3_gpucb.get_max_gp_variance()

    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment, gpts_max_variance_per_experiment, gpucb_max_variance_per_experiment


def simulate_step4():

    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices, users_poisson_parameters = generate_new_environment()
        ecomm4_gpts = Ecommerce4_GPTS(B_cap, budgets, product_prices)
        ecomm4_gpucb = Ecommerce4_GPUCB(B_cap, budgets, product_prices)

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
                click_probabilities,
                users_reservation_prices,
                users_poisson_parameters,
                product_prices,
                observations_probabilities
            )

        exp_clicks = env.estimate_num_of_clicks(budgets/B_cap)
        ecomm = Ecommerce(B_cap, budgets, product_prices)
        _ , optimal_gain_per_experiment[e] = ecomm.solve_optimization_problem(
            num_sold_items,
            exp_clicks,
            nodes_activation_probabilities
        )

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):

            arm = ecomm4_gpts.pull_arm()
            reward, estimated_sold_items = env.round_step4(arm, B_cap, nodes_activation_probabilities, num_sold_items)
            ecomm4_gpts.update(arm, reward, estimated_sold_items)
            _, gpts_gains_per_experiment[e][t] = ecomm4_gpts.solve_optimization_problem(nodes_activation_probabilities)

            arm = ecomm4_gpucb.pull_arm()
            reward, estimated_sold_items = env.round_step4(arm, B_cap, nodes_activation_probabilities, num_sold_items)
            ecomm4_gpucb.update(arm, reward, estimated_sold_items)
            _, gpucb_gains_per_experiment[e][t] = ecomm4_gpucb.solve_optimization_problem(nodes_activation_probabilities)


    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step5():

    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        env, observations_probabilities, click_probabilities, product_prices,\
             users_reservation_prices, users_poisson_parameters = generate_new_environment()

        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices)
        ecomm5_gpucb = Ecommerce5_GPUCB(B_cap, budgets, product_prices)

        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
                click_probabilities,
                users_reservation_prices,
                users_poisson_parameters,
                product_prices,
                observations_probabilities
            )
        exp_clicks = env.estimate_num_of_clicks(budgets/B_cap)

        ecomm = Ecommerce(B_cap, budgets, product_prices)

        _ , optimal_gain_per_experiment[e] = ecomm.solve_optimization_problem(
            num_sold_items,
            exp_clicks,
            nodes_activation_probabilities
        )


        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):

            arm, arm_idx = ecomm5_gpts.pull_arm()
            reward = env.round_step5(arm, nodes_activation_probabilities)
            ecomm5_gpts.update(arm, arm_idx, reward)
            
            _, gpts_gains_per_experiment[e][t] = ecomm5_gpts.solve_optimization_problem(
                num_sold_items,
                exp_clicks
            )
            
            # ----------------------

            arm, arm_idx = ecomm5_gpucb.pull_arm()
            reward = env.round_step5(arm, nodes_activation_probabilities)
            ecomm5_gpucb.update(arm, arm_idx, reward)

            _, gpucb_gains_per_experiment[e][t] = ecomm5_gpucb.solve_optimization_problem(
                num_sold_items,
                exp_clicks
            )


    return gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step6():

    swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments,T))

    tau = np.floor(np.sqrt(T)).astype(int)

    M = np.ceil(0.033 * T)
    eps = 0.1
    h = 2 * np.log(T)

    for e in range(0, n_experiments):
        print('Experiment n°', e)

        env, observations_probabilities, click_probabilities,\
            product_prices = generate_new_non_stationary_environment()

        ecomm6_swucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, M, eps, h)
        
        optimal_phase_gain = 0

        for t in tqdm(range(0, T), position=0, desc="n_iteration", leave=False):
            
            exp_clicks = env.estimate_num_of_clicks(budgets/B_cap)
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
            _, swucb_gains_per_experiment[e][t] = ecomm6_swucb.solve_optimization_problem(env.get_nodes_activation_probabilities())

            arm = ecomm6_cducb.pull_arm()
            reward, sold_items = env.round_step6(arm, B_cap, True)
            ecomm6_cducb.update(arm, reward, sold_items)
            _, cducb_gains_per_experiment[e][t] = ecomm6_cducb.solve_optimization_problem(env.get_nodes_activation_probabilities())

    return swucb_gains_per_experiment, cducb_gains_per_experiment, optimal_gain_per_experiment


def simulate_step7():

    gpts_gains_per_experiment, gpucb_gains_per_experiment, optimal_gain_per_experiment = np.zeros(
        shape=(n_experiments, T)), np.zeros(shape=(n_experiments, T)), np.zeros(shape=(n_experiments))
    

    for e in range(0, n_experiments):
        print('Experiment n°:', e)

        
        env, observations_probabilities, click_probabilities, product_prices,\
            users_reservation_prices, users_poisson_parameters = generate_new_environment()


        nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
            click_probabilities,
            users_reservation_prices,
            users_poisson_parameters,
            product_prices,
            observations_probabilities
        )

        exp_num_clicks = env.estimate_disaggregated_num_clicks(budgets/B_cap)

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
                    if not np.array_equal(pulled_arms[idx, :] , np.zeros(shape = NUM_OF_PRODUCTS)):
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
                    if not np.array_equal(pulled_arms[idx, :] , np.zeros(shape = NUM_OF_PRODUCTS)):
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


if __name__ == "__main__":

    # -----------SOCIAL INFLUENCE SIMULATION + STEP2 OPTIMIZATION PROBLEM --------------
    simulate_step2()

    # -----------STEP 3------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, gpts_max_variance_per_experiment, gpucb_max_variance_per_experiment = simulate_step3()
    plot_regrets(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts, gpts_max_variance_per_experiment, gpucb_max_variance_per_experiment, ["GPTS", "GPUCB"])

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
