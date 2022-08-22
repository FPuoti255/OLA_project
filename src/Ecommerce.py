import numpy as np
from itertools import permutations, combinations_with_replacement

from Constants import *
from Environment import Environment
import random

class Ecommerce(object):
    def __init__(self, B_cap : float):
        self.B_cap = B_cap
        self.budgets = np.array([random.uniform(i, i+10) for i in range(0, 101, 10)]) # np.arange(start = 0, stop = 101, step = 5)
        


    def solve_optimization_problem(self, env : Environment, nodes_activation_probabilities):

        value_per_click = np.dot(nodes_activation_probabilities, env.product_prices.T)

        # TODO implement this algorithm using the one on the slides 
        # For now I've used a dummy algorithm with very low performance
        # -------------------------------------------------------------
        # generating all the possible combination with replacement of 5 (campaigns) 
        # over the 8 possible budgets
        combinations = np.array([comb for comb in combinations_with_replacement(self.budgets, 5) if np.sum(comb) <= self.B_cap], dtype=float)

        # the combinations do not have any order, thus using the permutation we consider
        # all the possible assignment of those budgets to a given campaign
        perms = []
        for comb in combinations:
            [perms.append(perm) for perm in permutations(comb)]
        perms = np.array(list(set(perms))) #set() to remove duplicates

        best_allocation = []
        max_expected_reward = 0

        for allocation in perms:
            #the dirichlet does not accept values <= 0
            allocation[np.where(allocation == 0)] = 1.e-10
            
            # in order to get also the alpha_0 for the users landing on a webpage of a competitor,
            # we set the 'fictitious budget' of the competitor as the average of our allocations
            alphas = env.get_users_alphas(list(np.insert(allocation, obj=0, values = np.average(allocation))))
            
            # the notation inside alphas is to exclude the first column which represent alpha_0
            # but for alpha_0 our reward is 0 
            exp_rew = np.sum(np.dot(alphas[:, 1 :], value_per_click), axis=0)
            if exp_rew > max_expected_reward:
                max_expected_reward = exp_rew
                best_allocation = allocation

        return max_expected_reward, best_allocation

