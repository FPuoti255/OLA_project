import numpy as np
import random

from Utils import *
from constants import *
from Environment import Environment
from Network import Network

from Ecommerce import *
from Ecommerce3 import *
from Ecommerce5 import *


fully_connected_flag = True
B_cap = 200
budgets = np.arange(start=0, stop=B_cap + 1, step=B_cap / 10)

users_concentration_parameters = [82, 56, 80, 82, 42, 59]

users_price_range = 100
products_price_range = 60
product_prices = np.round(np.random.random(
    size=NUM_OF_PRODUCTS) * products_price_range, 2)

n_experiments = 10
T = 20


def generate_network_matrix(fully_connected: bool):
    '''
    :return: matrix representing the probability of going from a node to another
    '''

    adjacency_matrix = np.random.uniform(
        low=0.01, high=1.000001, size=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)
    )
    adjacency_matrix[np.diag_indices(n=NUM_OF_PRODUCTS, ndim=2)] = 0.0

    if not fully_connected:
        graph_mask = np.random.randint(
            low=0, high=2, size=adjacency_matrix.shape)
        adjacency_matrix = np.multiply(adjacency_matrix, graph_mask)

    # # maybe this normalization is not needed
    # for i in range(NUM_OF_PRODUCTS):
    #     col_sum = np.sum(adjacency_matrix[:, i])
    #     if col_sum != 0 :
    #         adjacency_matrix[:, i] = adjacency_matrix[:, i] / col_sum

    adjacency_matrix = np.round(adjacency_matrix, 2)
    return adjacency_matrix


def generate_observation_probabilities(click_probabilities):
    '''
    :return: probability 1 for observing the first slot of the secondary product
             and LAMBDA for the second slot
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


def generate_alphas():
    '''
    Alphas represents the percentage of users (for each class) landing on a specific product webpage including the competitor's
    '''
    # In order to generate a matrix of 𝛼_i with sum equal to 1, we used a multinomial distribution.
    # Notice that we needed to include also the 'competitors product', and we decided to give to all the products equal probability -> [1 / (NUM_OF_PRODUCTS+1)]
    # the 𝛼_0 is the one corresponding to the competitor(s) product

    alphas = np.random.dirichlet(alpha = users_concentration_parameters, size = NUM_OF_USERS_CLASSES) / NUM_OF_USERS_CLASSES

    return alphas


def generate_num_of_items_sold():
    return np.random.randint(low=1, high=10+1, dtype=np.int32, size=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))


def generate_new_environment():

    clk_prob = generate_network_matrix(fully_connected_flag)
    users_reservation_prices = np.round(np.random.random(
        size=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)) * users_price_range, 2)

    env = Environment(
        users_reservation_prices,
        click_probabilities=clk_prob,
        users_alpha=generate_alphas(),
        num_items_sold=generate_num_of_items_sold()
    )

    observations_probabilities = generate_observation_probabilities(clk_prob)

    # Network.print_graph(G=env.network.G)

    return env, observations_probabilities


def plot_regrets(gpts_regret, gpucb_regret):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(gpts_regret, "r")
    plt.plot(gpucb_regret, "g")

    plt.legend(["GPTS", "GPUCB"])
    plt.show()


if __name__ == "__main__":
    # -----------STEP 2------------
    ecommerce = Ecommerce(B_cap, budgets, product_prices, observations_probabilities)
    exp_clicks = env.estimate_expected_user_alpha(ecommerce.budgets / ecommerce.B_cap)
    optimal_allocation = ecommerce.solve_optimization_problem(
        env.get_network().get_adjacency_matrix(), env.get_num_items_sold(), env.get_users_reservation_prices(), exp_clicks)


    # -----------STEP 3------------
    gpucb_rewards_per_experiment = []
    gpts_rewards_per_experiment = []

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):
        env, observations_probabilities = generate_new_environment()

        ecomm3_gpts = Ecommerce3_TS(
            B_cap, budgets, product_prices, observations_probabilities)
        ecomm3_ucb = Ecommerce3_UCB(
            B_cap, budgets, product_prices, observations_probabilities)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):
            arm = ecomm3_ucb.pull_arm()
            reward = env.round_step3(arm)
            ecomm3_ucb.update(arm, reward)

            arm = ecomm3_gpts.pull_arm()
            reward = env.round_step3(arm)
            ecomm3_gpts.update(arm, reward)

        gpucb_rewards_per_experiment.append(ecomm3_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm3_gpts.collected_rewards)

    opt = opt = np.sum(env.get_users_alpha(), axis=0)[1:]
    # this np.mean is used to compute the average regret for each "product" -> output shape = (n_experiments x NUM_OF_PRODUCTS)

    gpts_regret_superarm = opt - np.mean(np.array(gpucb_rewards_per_experiment), axis=0).T
    gpucb_regret_superarm = opt - np.mean(np.array(gpts_rewards_per_experiment), axis=0).T 

    gpts_regret_per_round = np.sum(gpts_regret_superarm, axis = 1)
    gpucb_regret_per_round = np.sum(gpucb_regret_superarm, axis=1) 

    # # this np.mean before of the cumsum is to average over all the products
    gpts_regret = np.cumsum(gpts_regret_per_round)
    gpucb_regret = np.cumsum(gpucb_regret_per_round)

    plot_regrets(gpts_regret, gpucb_regret)

    # -----------STEP 5------------

    gpucb_rewards_per_experiment = []
    gpts_rewards_per_experiment = []

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):
        env, observations_probabilities = generate_new_environment()

        ecomm5_gpts = Ecommerce5_GPTS(
            B_cap, budgets, product_prices, observations_probabilities)
        ecomm5_ucb = Ecommerce5_UCB(
            B_cap, budgets, product_prices, observations_probabilities)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):
            arm, arm_idx = ecomm5_ucb.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_ucb.update(arm_idx, reward)

            arm, arm_idx = ecomm5_gpts.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_gpts.update(arm_idx, reward)

        gpucb_rewards_per_experiment.append(ecomm5_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm5_gpts.collected_rewards)

        opt = np.max(env.get_network().get_adjacency_matrix())

        gpts_regret = np.cumsum(
            np.mean(opt - gpucb_rewards_per_experiment, axis=0))
        gpucb_regret = np.cumsum(
            np.mean(opt - gpts_rewards_per_experiment, axis=0))

        plot_regrets(gpts_regret, gpucb_regret)
