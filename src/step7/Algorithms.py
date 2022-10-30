import numpy as np
from Ecommerce4 import *




class Algorithm(Ecommerce4):
    def __init__(self, algorithm: str, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(algorithm, B_cap, budgets, product_prices, gp_config)
        self.confidence = 0.01

    def get_new_instance(self):        
        return Algorithm(self.algorithm_name, 
                            self.algorithm.B_cap,
                            self.algorithm.budgets,
                            self.algorithm.product_prices,
                            self.algorithm.gp_config)

    def train_offline(self, pulled_arms, rewards, sold_items):
        assert(self.algorithm.t == 0)
        for i in range(len(pulled_arms)):
            self.update(pulled_arms[i], rewards[i], sold_items[i])

    def get_best_bound_arm(self):
        
        value_per_click = self.algorithm.compute_value_per_click(self.items_estimator.get_estimation())


        if self.algorithm_name == 'TS':
            estimated_reward = np.multiply(
                self.get_samples(),
                np.atleast_2d(value_per_click).T
            )
        else :
            estimated_reward = np.add(
                np.multiply(self.algorithm.means, np.atleast_2d(value_per_click).T),
                self.algorithm.confidence_bounds
            )

        _, mu = self.algorithm.dynamic_knapsack_solver(table = estimated_reward)
      
        return max(0.01, mu - np.sqrt( - np.log(self.confidence) / (2 * self.algorithm.t)))
    
