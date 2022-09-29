import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)

        self.arms = [(prod, int(budget)) for prod in range(NUM_OF_PRODUCTS) for budget in budgets]
        self.n_arms = len(self.arms)
        assert(self.n_arms == self.budgets.shape[0] * NUM_OF_PRODUCTS)

        self.t = 0

        self.means = np.ones(self.n_arms) * 0.5
        self.sigmas = np.ones_like(self.n_arms) * 0.5

        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]

        self.collected_rewards = []

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))

        self.gaussian_process = GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
            )

        # # The budgets are our arms!
        # self.n_arms = self.budgets.shape[0]

        # self.t = 0
        # # I'm generating a distribution of the budgets for each product
        # self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.05
        # self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.01

        # self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        # self.rewards_per_arm = [
        #     [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        # ]
        # self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        # alpha = 10.0
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # # we need one gaussian regressor for each product
        # self.gaussian_regressors = [
        #     GaussianProcessRegressor(
        #         kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        #     )
        #     for _ in range(NUM_OF_PRODUCTS)
        # ]
    

    def update(self, pulled_budgets_idxs, reward):
        self.t += 1
        self.update_observations(pulled_budgets_idxs, reward)
        self.update_model()

    def update_observations(self, pulled_budgets_idxs, reward):
        pulled_arms = [(i, self.budgets[pulled_budgets_idxs[i]]) for i in range(len(pulled_budgets_idxs))]

        for i in range(len(pulled_arms)):
            arm_idx = self.arms.index(pulled_arms[i])
            self.rewards_per_arm[arm_idx] = reward[i]
            self.pulled_arms.append(pulled_arms[i])
            self.collected_rewards.append(reward[i])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms)
        y = np.array(self.collected_rewards)
        self.gaussian_process.fit(x, y)

        self.means, self.sigmas = self.gaussian_process.predict(
            X=np.atleast_2d(self.arms), return_std=True
        )
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def pull_arm(self, num_sold_items):
        estimated_reward = self.estimate_reward(num_sold_items)     
        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)

    def compute_value_per_click (self, num_sold_items):
        assert(num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis = 1) # shape = (NUM_OF_PRODUCTS,)

    def estimate_reward(self, num_sold_items):
        pass



class Ecommerce3_GPTS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def estimate_reward(self, num_sold_items):        
        value_per_click = self.compute_value_per_click(num_sold_items)
        samples = np.random.normal(loc = self.means, scale=self.sigmas).reshape((NUM_OF_PRODUCTS, self.budgets.shape[0]))        
        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        return estimated_reward


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=self.n_arms, fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros_like(self.confidence_bounds)

    def update_observations(self, pulled_budgets_idxs, reward):

        super().update_observations(pulled_budgets_idxs, reward)
        
        pulled_arms = [(i, self.budgets[pulled_budgets_idxs[i]]) for i in range(len(pulled_budgets_idxs))]
        for i in range(len(pulled_arms)):
            arm_idx = self.arms.index(pulled_arms[i])
            self.N_a[arm_idx] += 1

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf


    def estimate_reward(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)
        estimated_reward = np.multiply(
            np.add(self.means, self.confidence_bounds).reshape((NUM_OF_PRODUCTS, self.budgets.shape[0])),
            np.atleast_2d(value_per_click).T
        )
        return estimated_reward

