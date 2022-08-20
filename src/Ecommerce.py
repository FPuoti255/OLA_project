import numpy as np
from itertools import permutations, combinations_with_replacement

from Constants import *
from Environment import Environment

class Ecommerce(object):
    def __init__(self, B_cap : float):
        self.B_cap = B_cap
        self.budgets = np.arange(start = 0, stop = 71, step = 10)
        


    def solve_optimization_problem(self, product_prices = product_prices, nodes_activation_probabilities = nodes_activation_probabilities):
        expected_reward = compute_exp_rew(product_prices, nodes_activation_probabilities)
        pass


    # TODO this needs to be modified. Now I've added it just to implement the algorithm
    def compute_exp_rew(product_prices, nodes_activation_probabilities):

        products = np.arange(0, NUM_OF_PRODUCTS)
        matrix = np.zeros(shape=(NUM_OF_PRODUCTS, self.budgets.shape[0]))

        # generating all the possible combination with replacement of 5 (campaigns) 
        # over the 8 possible budgets
        combinations = np.array([comb for comb in combinations_with_replacement(budgets, 5) if np.sum(comb) <= self.B_cap])

        # the combinations do not have any order, thus using the permutation we consider
        # all the possible assignment of those budgets to a given campaign
        perms = []
        for comb in combinations:
            [perms.append(perm) for perm in permutations(comb)]
        perms = np.array(list(set(perms))) #set() to remove duplicates


        # TODO TO BE CONTINUED
        # The idea is that for each possible assignment of budget to the campaigns,
        # we compute the alpha as dirichlet(a=budgets), and then we compute the
        # expected reward using the probabilities estimated by the montecarlo sampling 
        # for each product

        pass

