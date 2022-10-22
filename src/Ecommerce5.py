import numpy as np

from Ecommerce3 import Ecommerce3_GPTS, Ecommerce3_GPUCB
from constants import *
from Utils import *


class Ecommerce5_GPTS(Ecommerce3_GPTS):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)

    def pull_arm(self):
        estimated_reward = self.get_samples()
        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)


class Ecommerce5_GPUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)

    def pull_arm(self):
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            estimated_reward = np.add(self.means, self.confidence_bounds)
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()


