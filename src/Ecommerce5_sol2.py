import numpy as np

from Ecommerce import *
from constants import *
from Utils import *



class Ecommerce5(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)
        
        self.t = 0
        self.estimated_nodes_activations = np.ones(shape= (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) * 0.5
        self.collected_nodes_activations = [self.estimated_nodes_activations.copy()]

    def pull_arm(self, users_alpha, users_poisson_parameters):
        assert(users_poisson_parameters.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        
        estimated_sold_items = np.multiply(
            self.estimated_nodes_activations,
            np.random.poisson(users_poisson_parameters)
        )

        value_per_click = np.sum(np.multiply( estimated_sold_items, self.product_prices), axis = 1)

        estimated_reward = np.multiply(
            users_alpha,
            np.atleast_2d(value_per_click).T
        )

        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)

    def update(self, real_nodes_activations):
        self.t += 1
        self.collected_nodes_activations.append(real_nodes_activations)
        self.estimated_nodes_activations = np.mean(self.collected_nodes_activations, axis = 0)



