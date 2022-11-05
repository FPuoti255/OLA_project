import numpy as np
from Ecommerce3 import Ecommerce3_GPTS, Ecommerce3_GPUCB
from constants import *
from Utils import *


class Ecommerce5_GPTS(Ecommerce3_GPTS):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)
        self.exploration_probability = 0.3

    def get_samples(self):
        samples = np.empty(shape = (NUM_OF_PRODUCTS, self.n_arms))
        X = np.atleast_2d(self.budgets).T
        for prod in range(NUM_OF_PRODUCTS):
            samples[prod] = self.gaussian_regressors[prod].sample_y(X).T       

        # With respect to the Ecommerce3_GPTS I've removed the np.clip(0, 1)       
        return np.maximum(samples, 0.0)
    
    def update_model(self):
        super().update_model()

        if self.t < 50:
            self.exploration_probability = 1.0 / np.sqrt(self.t+10)
        else:
            self.exploration_probability = 0.01 

    def pull_arm(self):
        # Since the Ecommerce does not have the information about the value per click,
        # the estimated reward will directly be the samples drawn by the gaussian process
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            estimated_reward = self.get_samples()
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()


class Ecommerce5_GPUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)

    def pull_arm(self):
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            estimated_reward = self.get_samples()
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()


