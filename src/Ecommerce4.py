import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce3 import *

from constants import *
from Utils import *


class Ecommerce4(Ecommerce3):

    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.sold_items_means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 10
        self.sold_items_sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 5

        self.collected_sold_items = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        ]
        

    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        self.update_observations(pulled_arm, reward, sold_items)
        self.update_model()

    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(
                X=np.atleast_2d(self.budgets).T, return_std=True
            )
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)
        

    def solve_optimization_problem(self, nodes_activation_probabilities): #shape 5x5
        num_sold_items = np.random.normal(self.sold_items_means, self.sold_items_sigmas) # 5 x 21
        exp_num_clicks = np.random.normal(self.means, self.sigmas) # 5 x 21

        total_margin_for_each_product = np.multiply(num_sold_items.T, self.product_prices).T  # shape = 5x21

        value_per_click = np.ones_like(exp_num_clicks)
        for i in range(self.budgets.shape[0]):
            value_per_click[ : , i] = np.dot(nodes_activation_probabilities, total_margin_for_each_product[:, i])
        
        assert(value_per_click.shape == (NUM_OF_PRODUCTS, self.budgets.shape[0]))

        exp_reward = np.multiply(exp_num_clicks, value_per_click)

        budgets_indexes, reward = self.dynamic_knapsack_solver(
            table=exp_reward
        )
        optimal_allocation = self.budgets[budgets_indexes]

        return optimal_allocation, reward


class Ecommerce4_GPTS(Ecommerce4, Ecommerce3_GPTS):

    def update_observations(self, pulled_arm, reward, sold_items):
        for i in range(NUM_OF_PRODUCTS):
            budget_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.rewards_per_arm[i][budget_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

            self.collected_sold_items[i][budget_idx].append(sold_items[i])
            
            items  = np.array(self.collected_sold_items)
            self.sold_items_means[i][budget_idx] = np.mean(items[i][budget_idx])
            self.sold_items_sigmas[i][budget_idx] = np.std(items[i][budget_idx])



class Ecommerce4_GPUCB(Ecommerce4, Ecommerce3_GPUCB):

    def update_observations(self, pulled_arm, reward, sold_items):
        super().update_observations(pulled_arm, reward)
        for i in range(NUM_OF_PRODUCTS):
            budget_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.collected_sold_items[i][budget_idx].append(sold_items[i])

            items  = np.array(self.collected_sold_items)
            self.sold_items_means[i][budget_idx] = np.mean(items[i][budget_idx])
            self.sold_items_sigmas[i][budget_idx] = np.std(items[i][budget_idx])
    
        