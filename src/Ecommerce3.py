from itertools import combinations_with_replacement, permutations
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, alpha = None, kernel = None):

        super().__init__(B_cap, budgets, product_prices)

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        self.exploration_probability = 0.05
        self.perms = None

        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for _ in range(self.n_arms)] for _ in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        if kernel is None and alpha is None:    
            hyperparameters = json.load(open("hyperparameters.json"))
            alpha = hyperparameters["alpha"]
            kernel = C(
                constant_value=hyperparameters["constant_value"], 
                constant_value_bounds=(hyperparameters["constant_value_bounds1"],hyperparameters["constant_value_bounds2"])) * RBF(
                length_scale=hyperparameters["length_scale"], 
                length_scale_bounds=(hyperparameters["length_scale_bounds1"],hyperparameters["length_scale_bounds2"])
            )


        assert(alpha is not None and kernel is not None)

        # I'm generating a prior distribution over the budgets
        params = [[0.5,3.0], [0.5,5.0],[1.0,3.0],[1.0,5.0],[5.0,10.0], [5.0, 5.0], [10.0, 20.0]]
        now = 5 # [5.0, 10.0] best so far
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * params[now][0]
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * params[now][1]
        #print("means:", params[now][0], " and variance:", params[now][1])

        X = np.atleast_2d(self.budgets).T
        
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=9
            ).fit(X, np.random.normal(self.means[i], self.sigmas[i]))
            for i in range(NUM_OF_PRODUCTS)
        ]

    def update(self, pulled_arm_idxs, reward):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward)
        self.update_model()

    def pull_arm(self, num_sold_items):
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            value_per_click = self.compute_value_per_click(num_sold_items)
            estimated_reward = np.multiply(
                self.get_samples(),
                np.atleast_2d(value_per_click).T
            )
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()
        
    def update_observations(self, pulled_arm_idxs, reward):
        for prod_id in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[prod_id][pulled_arm_idxs[prod_id]].append(reward[prod_id])
            self.pulled_arms[prod_id].append(self.budgets[pulled_arm_idxs[prod_id]])
            self.collected_rewards[prod_id].append(reward[prod_id])

    def update_model(self):
        for prod_id in range(NUM_OF_PRODUCTS):
            X = np.atleast_2d(self.pulled_arms[prod_id]).T
            X_test = np.atleast_2d(self.budgets).T
            y = np.array(self.collected_rewards[prod_id])
            self.means[prod_id], self.sigmas[prod_id] = self.gaussian_regressors[prod_id].fit(X, y).predict(X=X_test, return_std=True)
            self.sigmas[prod_id] = np.maximum(self.sigmas[prod_id], 5e-2)

    def compute_value_per_click(self, num_sold_items):
        '''
        :returns: value per click for each product. Shape = (NUM_OF_PRODUCTS,)
        '''
        assert (num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis=1)

    def random_sampling(self):
        if self.perms is None:
            combinations = np.array([comb for comb in combinations_with_replacement(self.budgets, 5) if np.sum(comb) == self.B_cap], dtype=float)
            perms = []
            for comb in combinations:
                [perms.append(perm) for perm in permutations(comb)]
            self.perms = np.array(list(set(perms))) #set() to remove duplicates

        choice = self.perms[np.random.choice(self.perms.shape[0], size=1, replace=False), :].reshape((NUM_OF_PRODUCTS,))
        choice_idxs = np.zeros_like(choice)

        for prod in range(NUM_OF_PRODUCTS):
            choice_idxs[prod] = np.where(self.budgets==choice[prod])[0][0]
        
        return choice, choice_idxs.astype(int)


class Ecommerce3_GPTS(Ecommerce3):

    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)
        self.exploration_probability = 0     

    def get_samples(self):
        samples = np.empty(shape = (NUM_OF_PRODUCTS, self.n_arms))
        X = np.atleast_2d(self.budgets).T
        for prod in range(NUM_OF_PRODUCTS):
            samples[prod] = self.gaussian_regressors[prod].sample_y(X).T                
        return samples


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)

        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=1e400)
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm_idxs, reward):

        super().update_observations(pulled_arm_idxs, reward)

        for i in range(NUM_OF_PRODUCTS):
            self.N_a[i][pulled_arm_idxs[i]] += 1

        # bayesian UCB
        self.confidence_bounds = 0.2 * np.sqrt((2 * np.log(self.t) / self.N_a)) * self.sigmas #0.2 * self.sigmas #= np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = 1e400

    def get_samples(self):        
        return np.add(self.means, self.confidence_bounds)
