import numpy as np
import random

from Utils import *
from constants import *
from Environment import Environment
from Network import Network

from Ecommerce2 import *
from Ecommerce3 import *
from Ecommerce5 import *


def generate_click_probabilities(fully_connected: bool):
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


def generate_prices(product_range: int, users_range: int):
    """
    :param users_range: is greater than product_range since we want to increase a little the probability
                        that a user will buy a given product
    """
    return np.round(
        np.random.random(size=NUM_OF_PRODUCTS) * product_range, 2
    ), np.round(np.random.random(size=NUM_OF_PRODUCTS) * users_range, 2)


def generate_new_environment(fully_connected_flag, B_cap, budgets):

    clk_prob = generate_click_probabilities(
        fully_connected=fully_connected_flag)
    product_prices, users_reservation_prices = generate_prices(
        product_range=60, users_range=100)
    tot_num_users = np.random.normal(loc=1000, scale=25)

    env = Environment(
        users_reservation_prices,
        click_probabilities=clk_prob,
        observations_probabilities=generate_observation_probabilities(
            clk_prob),
        tot_num_users=tot_num_users,
    )

    # Network.print_graph(G=env.network.G)

    # --- SOCIAL INFLUENCE--------
    nodes_activation_probabilities = env.get_nodes_activation_probabilities(
        product_prices)

    # ------STEP 2-------------
    ecomm2 = Ecommerce2(B_cap, budgets, product_prices, tot_num_users)
    optimal_allocation = ecomm2.solve_optimization_problem(
        env, nodes_activation_probabilities)

    return env, clk_prob, product_prices, users_reservation_prices, tot_num_users, nodes_activation_probabilities, optimal_allocation

def plot_regrets(gpts_regret, gpucb_regret):
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")

    plt.plot(gpts_regret, "r")
    plt.plot(gpucb_regret, "g")
    
    plt.legend(["GPTS","GPUCB"])
    plt.show()


if __name__ == "__main__":

    fully_connected_flag = True
    B_cap = 200
    budgets = np.arange(start=0, stop=B_cap + 1, step=B_cap / 10)

    # ------------------------------------
    # ---------BEGIN SIMULATIONS----------
    # ------------------------------------

    n_experiments = 10
    T = 20

    # -----------STEP 3------------
    gpucb_rewards_per_experiment = []
    gpts_rewards_per_experiment = []

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):
        env, clk_prob, product_prices, users_reservation_prices, tot_num_users, nodes_activation_probabilities, optimal_allocation = generate_new_environment(
            fully_connected_flag, B_cap, budgets)

        ecomm3_gpts = Ecommerce3_TS(B_cap, budgets, product_prices, tot_num_users)
        ecomm3_ucb = Ecommerce3_UCB(B_cap, budgets, product_prices, tot_num_users)

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):
            arm = ecomm3_ucb.pull_arm()
            reward = env.round_step3(arm)
            ecomm3_ucb.update(arm, reward)

            arm = ecomm3_gpts.pull_arm(nodes_activation_probabilities)
            reward = env.round_step3(arm)
            ecomm3_gpts.update(arm, reward)

        gpucb_rewards_per_experiment.append(ecomm3_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm3_gpts.collected_rewards)

    opt = env.round_step3(pulled_arm=optimal_allocation)
    # this np.mean is used to compute the average regret for each "product" -> output shape = (n_experiments x NUM_OF_PRODUCTS)
    gpts_regret_superarm = opt - np.mean(np.array(gpucb_rewards_per_experiment), axis=2)
    gpucb_regret_superarm = opt -np.mean(np.array(gpts_rewards_per_experiment), axis=2)

    # this np.mean before of the cumsum is to average over all the products
    gpts_regret = np.cumsum(np.mean(gpts_regret_superarm, axis=1))
    gpucb_regret = np.cumsum(np.mean(gpucb_regret_superarm, axis=1))

    plot_regrets(gpts_regret, gpucb_regret)


    # -----------STEP 5------------

    gpucb_rewards_per_experiment = []
    gpts_rewards_per_experiment = []

    for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):
        env, clk_prob, product_prices, users_reservation_prices, tot_num_users, nodes_activation_probabilities, optimal_allocation = generate_new_environment(
            fully_connected_flag, B_cap, budgets)

        ecomm5_gpts = Ecommerce5_GPTS(
            B_cap,
            budgets,
            product_prices,
            tot_num_users,
        )
        ecomm5_ucb = Ecommerce5_UCB(
            B_cap,
            budgets,
            product_prices,
            tot_num_users,
        )

        for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):
            arm, arm_idx = ecomm5_ucb.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_ucb.update(arm_idx, reward)

            arm, arm_idx = ecomm5_gpts.pull_arm()
            reward = env.round_step5(arm)
            ecomm5_gpts.update(arm_idx, reward)

        gpucb_rewards_per_experiment.append(ecomm5_ucb.collected_rewards)
        gpts_rewards_per_experiment.append(ecomm5_gpts.collected_rewards)

    opt = np.max(nodes_activation_probabilities)

    gpts_regret = np.cumsum(np.mean(opt - gpucb_rewards_per_experiment, axis=0))
    gpucb_regret = np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0))

    plot_regrets(gpts_regret, gpucb_regret)