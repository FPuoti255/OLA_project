import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Constants import *


class Ecommerce_step3(Ecommerce):
    def __init__(self, B_cap : float, budgets, product_prices, tot_num_users):
        
        super().__init__(B_cap=B_cap, budgets = budgets, product_prices=product_prices, tot_num_users = tot_num_users)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        self.rewards_per_arm = x =[[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

        self.means = np.ones(self.n_arms) * 0.5
        self.sigmas = np.ones(self.n_arms)* 0.4

        self.pulled_arms = []

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha, 
                                    normalize_y = True, n_restarts_optimizer = 9)


    def update_observations(self, pulled_arm, reward):
        for i in range(pulled_arm.shape[0]):
            self.rewards_per_arm[int(np.where(self.budgets == pulled_arm[i])[0])].append(reward[i])
            self.pulled_arms.append(pulled_arm[i])

        self.collected_rewards = np.append(self.collected_rewards, reward)
        

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.budgets).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, nodes_activation_probabilities):
        sampled_values = np.random.normal(loc=self.means, scale=self.sigmas)

        sampled_values[np.where(sampled_values < 0)] = 0
        sampled_values[np.where(sampled_values >= 1)] = 0.99

        value_per_click = np.dot(nodes_activation_probabilities, self.product_prices.T)
        
        exp_reward = np.zeros(shape = (NUM_OF_PRODUCTS, self.budgets.shape[0]))

        for prd in range(NUM_OF_PRODUCTS):
            for j in range(1, self.n_arms):
                exp_reward[prd][j] = value_per_click[prd] * sampled_values[j]
        
        superarm_idx, _ = self.dynamic_algorithm(table = exp_reward)

        return sampled_values[superarm_idx], superarm_idx
    
    def dynamic_algorithm(self, table):
        return super().dynamic_algorithm(table)


    def generate_superarms(self):
        # generating all the possible combination with replacement of 5 (campaigns) 
        # over the 8 possible budgets
        combinations = np.array([comb for comb in combinations_with_replacement(self.budgets, 5) if np.sum(comb) <= self.B_cap], dtype=float)

        # the combinations do not have any order, thus using the permutation we consider
        # all the possible assignment of those budgets to a given campaign
        perms = []
        for comb in combinations:
            [perms.append(perm) for perm in permutations(comb)]
        perms = np.array(list(set(perms))) #set() to remove duplicates

        return perms