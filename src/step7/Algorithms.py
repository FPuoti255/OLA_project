import numpy as np
from Ecommerce4 import *

confidence = 0.01

class TS(Ecommerce4_GPTS):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def get_new_instance(self):
        return TS(self.B_cap, self.budgets, self.product_prices)

    def train_offline(self, pulled_arms, rewards, sold_items):
        assert(self.t == 0)
        for i in range(len(pulled_arms)):
            self.update(pulled_arms[i], rewards[i], sold_items[i])
    
    def get_best_bound_arm(self):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)
        estimate_sold_product = np.random.normal(self.sold_items_means, self.sold_items_sigmas)
        exp_rew = np.multiply(samples.T, estimate_sold_product).T
        _, mu = self.revisited_knapsack_solver(table=exp_rew)        
        return min(0.01, mu - np.sqrt( - np.log(confidence) / (2 * self.t)))


class UCB(Ecommerce4_GPUCB):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def get_new_instance(self):
        return UCB(self.B_cap, self.budgets, self.product_prices)

    def train_offline(self, pulled_arms, rewards, sold_items):
        assert(self.t == 0)
        for i in len(pulled_arms):
            self.update(pulled_arms[i], rewards[i], sold_items[i])

    def get_best_bound_arm(self):
        upper_conf = self.means + self.confidence_bounds
        estimate_sold_product = np.random.normal(self.sold_items_means, self.sold_items_sigmas)
        exp_rew = np.multiply(upper_conf.T, estimate_sold_product).T
        _, mu = self.revisited_knapsack_solver(table=upper_conf)
        return min(0.01, mu - np.sqrt( - np.log(confidence) / (2 * self.t)))