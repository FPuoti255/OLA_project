import numpy as np

from Ecommerce3 import *
from constants import *
from Utils import *


class Ecommerce4(Ecommerce3):
    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)
        self.sold_items_means = np.ones(shape = (NUM_OF_PRODUCTS,))

    def pull_arm(self):
        if np.random.binomial(n = 1, p = 1 - self.exploration_probability):
            value_per_click = self.compute_value_per_click()
            estimated_reward = np.multiply(
                self.get_samples(),
                np.atleast_2d(value_per_click).T
            )
            budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
            return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)
        else:
            return self.random_sampling()

    def update(self, pulled_arm_idxs, reward, num_sold_items):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward, num_sold_items)
        self.update_model()

    def update_observations(self, pulled_arm_idxs, reward, num_sold_items):
        super().update_observations(pulled_arm_idxs, reward)
        self.sold_items_means = np.divide(
                np.multiply(self.sold_items_means, self.t) + num_sold_items,
                self.t + 1
            )

    def compute_value_per_click(self):
        return np.multiply(self.sold_items_means, self.product_prices)


class Ecommerce4_GPTS(Ecommerce4):
    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)
        self.exploration_probability = 0     

    def get_samples(self):
        samples = np.empty(shape = (NUM_OF_PRODUCTS, self.n_arms))
        X = np.atleast_2d(self.budgets).T
        for prod in range(NUM_OF_PRODUCTS):
            samples[prod] = self.gaussian_regressors[prod].sample_y(X).T                
        return samples


class Ecommerce4_GPUCB(Ecommerce4):

    def __init__(self, B_cap: float, budgets, product_prices, alpha=None, kernel=None):
        super().__init__(B_cap, budgets, product_prices, alpha, kernel)
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=1e400)
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_bounds(self, pulled_arm_idxs):
        for i in range(NUM_OF_PRODUCTS):
            self.N_a[i][pulled_arm_idxs[i]] += 1

        # bayesian UCB
        self.confidence_bounds = 0.2 * np.sqrt((2 * np.log(self.t) / self.N_a)) * self.sigmas #0.2 * self.sigmas #= np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = 1e400

    def update_observations(self, pulled_arm_idxs, reward, num_sold_items):
        super().update_observations(pulled_arm_idxs, reward, num_sold_items)
        self.update_bounds(pulled_arm_idxs)

    def get_samples(self):        
        return np.add(self.means, self.confidence_bounds)




    

