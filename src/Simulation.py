import numpy as np
import random

from Utils import *
from constants import *
from Environment import Environment
from Network import Network

from Ecommerce2 import *
from Ecommerce3 import *


def generate_click_probabilities(fully_connected: bool):
    '''
    :return: matrix representing the probability of going from a node to another 
    '''

    adjacency_matrix = np.random.uniform(
        low=0.01, high=1.000001, size=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)
    )
    adjacency_matrix[np.diag_indices(n=NUM_OF_PRODUCTS, ndim=2)] = 0.0

    if not fully_connected:
        graph_mask = np.random.randint(low=0, high=2, size=adjacency_matrix.shape)
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


if __name__ == "__main__":

    # click_probabilities == edge weights in our case
    click_probabilities = generate_click_probabilities(fully_connected=False)
    observations_probabilities = generate_observation_probabilities(
        click_probabilities=click_probabilities
    )

    B_cap = 200
    budgets = np.arange(start=0, stop=B_cap + 1, step=B_cap / 10)

    tot_num_users = np.random.normal(loc=1000, scale=25)

    product_prices, users_reservation_prices = generate_prices(
        product_range=60, users_range=100
    )

    env = Environment(
        users_reservation_prices=users_reservation_prices,
        click_probabilities=click_probabilities,
        observations_probabilities=observations_probabilities,
        tot_num_users=tot_num_users,
    )

    Network.print_graph(G=env.network.G)

    # --- SOCIAL INFLUENCE--------
    nodes_activation_probabilities = env.get_nodes_activation_probabilities(
        product_prices=product_prices
    )

    # -----------STEP 2------------
    ecomm2 = Ecommerce2(
        B_cap=B_cap,
        budgets=budgets,
        product_prices=product_prices,
        tot_num_users=tot_num_users,
    )
    ecomm2.solve_optimization_problem(
        env=env, nodes_activation_probabilities=nodes_activation_probabilities
    )

    # -----------STEP 3------------
    ecomm3_ts = Ecommerce3_TS(B_cap = B_cap, budgets = budgets, product_prices = product_prices, tot_num_users = tot_num_users)
    ecomm3_ucb = Ecommerce3_UCB(B_cap = B_cap, budgets = budgets, product_prices = product_prices, tot_num_users = tot_num_users)
    
    for _ in range(100):
        print('------Thompson Sampling--------')
        arms_values = ecomm3_ts.pull_arm(nodes_activation_probabilities)
        print(arms_values)
        reward = env.round_step3(pulled_arm=arms_values)
        ecomm3_ts.update(pulled_arm = arms_values, reward = reward)

        print('------UCB--------')
        arms_values = ecomm3_ucb.pull_arm(nodes_activation_probabilities=nodes_activation_probabilities)
        print(arms_values)
        reward = env.round_step3(pulled_arm=arms_values)
        ecomm3_ucb.update(pulled_arm = arms_values, reward = reward)

    # -----------STEP 5------------