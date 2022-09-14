from mimetypes import init
import numpy as np

from Utils import *
from constants import *

from Environment import *
from Non_Stationary_Environment import *

from Social_influence import *
from Network import Network

from Ecommerce import *
from Ecommerce3 import *
from Ecommerce4 import *
from Ecommerce5 import *
from Ecommerce6 import *


B_cap = 200
budgets = np.arange(start=0, stop=B_cap + 1, step=B_cap / 10)

users_price_range = 100
products_price_range = 60

n_experiments = 4
T = 4
n_phases = int(ceil(T/10))
phase_len = int(ceil(T/n_phases))


def generate_click_probabilities():
    '''
    :return: matrix representing the probability of going from a node to another
    '''

    adjacency_matrix = np.random.uniform(low=1e-2, high=1+1e-5, size=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
    adjacency_matrix[np.diag_indices(n=NUM_OF_PRODUCTS, ndim=2)] = 0.0

    # set some values to zero is not fully connected, otherwise it's ready
    if not fully_connected:
        graph_mask = np.random.randint(low=0, high=1+1, size=adjacency_matrix.shape)
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


def generate_alphas(users_parameters=[82, 56, 80, 82, 42, 59]):
    '''
    :return: alphas represents the percentage of users (for each class) landing on a specific product webpage including the competitor's
    '''
    # In order to generate a matrix of 𝛼_i with sum equal to 1, we used a multinomial distribution.
    # Notice that we needed to include also the 'competitors product', and we decided to give to all the products equal probability -> [1 / (NUM_OF_PRODUCTS+1)]
    # the 𝛼_0 is the one corresponding to the competitor(s) product

    alphas = np.random.dirichlet(alpha=users_parameters, size=NUM_OF_USERS_CLASSES) / NUM_OF_USERS_CLASSES
    return alphas


def generate_new_environment():
    '''
    :return: env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices
    '''

    # ----------ogni esperimento
    click_probabilities = generate_click_probabilities()
    # Secondary product set by the business unit
    observations_probabilities = generate_observation_probabilities(click_probabilities)
    
    product_prices = np.round(np.random.random(size=NUM_OF_PRODUCTS) * products_price_range, 2)
    users_reservation_prices = np.round(np.random.random(size=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)) * users_price_range, 2)
    users_alpha = generate_alphas()

    env = Environment(users_reservation_prices, click_probabilities, users_alpha)
    
    # Network.print_graph(G=env.network.G)
    return env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices


def generate_new_non_stationary_environment():

    '''
    :return: env, observations_probabilities, click_probabilities, product_prices, num_sold_items, nodes_activation_probabilities
    '''

    click_probabilities = generate_click_probabilities()
    observations_probabilities = generate_observation_probabilities(click_probabilities)
    product_prices = np.round(np.random.random(size=NUM_OF_PRODUCTS) * products_price_range, 2)

    users_reservation_prices = []
    nodes_activation_probabilities = []
    num_sold_items = []
    product_functions_idxs = []
    users_alpha = []
    prod_fun_idx = np.arange(NUM_OF_PRODUCTS)

    for _ in range(n_phases):

        res_prices = np.round(np.random.random(size=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)) * users_price_range, 2)
        users_reservation_prices.append(res_prices)

        estimation = estimate_nodes_activation_probabilities(
            click_probabilities,
            res_prices,
            product_prices,
            observations_probabilities
        )
        nodes_activation_probabilities.append(estimation[0])
        num_sold_items.append(estimation[1])

        np.random.shuffle(prod_fun_idx)  # In place shuffling
        product_functions_idxs.append(prod_fun_idx.copy())

        users_parameters = np.random.randint(50, high=101, size=6)
        users_alpha.append(generate_alphas(users_parameters=users_parameters))

    env = Non_Stationary_Environment(
        users_reservation_prices, product_functions_idxs, click_probabilities, users_alpha, num_sold_items, T)

    # Network.print_graph(G=env.network.G)

    return env, observations_probabilities, click_probabilities, product_prices, num_sold_items, nodes_activation_probabilities



def simulate_step2():

    env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices = generate_new_environment()
    
    nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
        click_probabilities,
        users_reservation_prices,
        product_prices,
        observations_probabilities
    )
    ecomm = Ecommerce(B_cap, budgets, product_prices, observations_probabilities)

    exp_clicks = env.estimate_expected_user_alpha(ecomm.budgets / ecomm.B_cap)

    optimal_allocation = ecomm.solve_optimization_problem(
        env.get_network().get_adjacency_matrix(),
        num_sold_items,
        env.get_users_reservation_prices(),
        exp_clicks,
        nodes_activation_probabilities
    )

    return product_prices



def simulate_step3():

    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = [],[],[]

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):

        env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices = generate_new_environment()

        ecomm3_gpts = Ecommerce3_GPTS(B_cap, budgets, product_prices)
        ecomm3_ucb = Ecommerce3_GPUCB(B_cap, budgets, product_prices)


        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):

            nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
                click_probabilities,
                users_reservation_prices,
                product_prices,
                observations_probabilities
            )

            arm = ecomm3_gpts.pull_arm(num_sold_items)
            reward = env.round_step3(arm)
            ecomm3_gpts.update(arm, reward)

            arm = ecomm3_ucb.pull_arm(num_sold_items)
            reward = env.round_step3(arm)
            ecomm3_ucb.update(arm, reward)

        gpts_rewards_per_experiment.append(ecomm3_gpts.collected_rewards)
        gpucb_rewards_per_experiment.append(ecomm3_ucb.collected_rewards)
        opts.append(np.sum(env.get_users_alpha(), axis=0)[1:])

    return gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts



def simulate_step4():
    
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = [],[],[]

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):

        env, observations_probabilities, click_probabilities, product_prices, users_reservation_prices = generate_new_environment()
        ecomm4_gpts = Ecommerce4_GPTS(B_cap, budgets, product_prices)
        ecomm4_ucb = Ecommerce4_GPUCB(B_cap, budgets, product_prices)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):

            nodes_activation_probabilities, num_sold_items = estimate_nodes_activation_probabilities(
                click_probabilities,
                users_reservation_prices,
                product_prices,
                observations_probabilities
            )

            arm = ecomm4_ucb.pull_arm()
            reward, sold_items = env.round_step4(arm, num_sold_items)
            ecomm4_ucb.update(arm, reward, sold_items)

            arm = ecomm4_gpts.pull_arm()
            reward, sold_items = env.round_step4(arm, num_sold_items)
            ecomm4_gpts.update(arm, reward, sold_items)

        gpucb_rewards_per_experiment.append(ecomm4_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm4_gpts.collected_rewards)

        opts.append(np.sum(env.get_users_alpha(), axis=0)[1:])

    return gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts



def simulate_step5():

    gpucb_rewards_per_experiment, gpts_rewards_per_experiment, opts = [],[],[]

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):

        env, _, _, product_prices, _ = generate_new_environment()

        ecomm5_gpts = Ecommerce5_GPTS(B_cap, budgets, product_prices)
        ecomm5_ucb = Ecommerce5_UCB(B_cap, budgets, product_prices)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):


            arm, arm_idx = ecomm5_ucb.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_ucb.update(arm_idx, reward)

            arm, arm_idx = ecomm5_gpts.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_gpts.update(arm_idx, reward)

        gpucb_rewards_per_experiment.append(ecomm5_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm5_gpts.collected_rewards)

        opts.append(np.max(env.get_network().get_adjacency_matrix()))

    return gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts



def simulate_step6():
    
    swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts = [],[],[]

    tau = np.floor(np.sqrt(T)).astype(int)

    M = np.ceil(0.033 * T)
    eps = 0.1
    h = 2 * np.log(T)

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):

        env, _, _, product_prices, _, _ = generate_new_non_stationary_environment()
        
        env2 = env.copy()

        ecomm6_ucb = Ecommerce6_SWUCB(B_cap, budgets, product_prices, tau)
        ecomm6_cducb = Ecommerce6_CDUCB(B_cap, budgets, product_prices, M, eps, h)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):

            arm = ecomm6_ucb.pull_arm()
            reward, sold_items = env.round_step6(arm)
            ecomm6_ucb.update(arm, reward, sold_items)
        
            arm = ecomm6_cducb.pull_arm()
            reward, sold_items = env2.round_step6(arm)
            ecomm6_cducb.update(arm, reward, sold_items)


        swucb_rewards_per_experiment.append(ecomm6_ucb.collected_rewards)
        cducb_rewards_per_experiment.append(ecomm6_cducb.collected_rewards)

        opts.append(np.sum(env.get_users_alpha(), axis=0)[1:])

    return swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts


if __name__ == "__main__":

    # -----------SOCIAL INFLUENCE SIMULATION + STEP2 OPTIMIZATION PROBLEM --------------
    simulate_step2()

    # -----------STEP 3------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = simulate_step3()
    plot_regrets_step3(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts)

    # -----------STEP 4------------
    gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts = simulate_step4()
    plot_regrets_step4(gpts_rewards_per_experiment, gpucb_rewards_per_experiment, opts)

    # -----------STEP 5------------
    gpucb_rewards_per_experiment, gpts_rewards_per_experiment, opts = simulate_step5()
    plot_regrets_step5(gpts_rewards_per_experiment,gpucb_rewards_per_experiment,opts)

    # -----------STEP 6------------
    swucb_rewards_per_experiment, cducb_rewards_per_experiment, opts = simulate_step6()
    plot_regrets_step6(swucb_rewards_per_experiment,cducb_rewards_per_experiment, opts)
