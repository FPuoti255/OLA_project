import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Environment import *
from Utils import *
from Social_influence import *


class Ecommerce5(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)
        
        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) * 0.5
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) * 0.2

        self.t = 0

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(NUM_OF_PRODUCTS)] for j in range(NUM_OF_PRODUCTS)
        ]

        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
            )
            for i in range(NUM_OF_PRODUCTS)
        ]

    def update(self, arm, reward):
        self.t += 1
        self.update_observations(arm,reward)
        self.update_model(arm)

    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm, reward):
        self.rewards_per_arm[arm[0]][arm[1]].append(reward)
        
        self.collected_rewards[arm[0]].append(reward)
        self.pulled_arms[arm[0]].append(arm[1])

    def update_model(self, arm):        
        x = np.atleast_2d(self.pulled_arms[arm[0]]).T
        y = np.array(self.collected_rewards[arm[0]])
        self.gaussian_regressors[arm[0]].fit(x, y)

        self.means[arm[0]], self.sigmas[arm[0]] = self.gaussian_regressors[arm[0]].predict(
            X=np.atleast_2d(np.arange(0, NUM_OF_PRODUCTS)).T, return_std=True
        )
        self.sigmas[arm[0]] = np.maximum(self.sigmas[arm[0]], 1e-2)

    def pull_arm(self):
        pass

    def get_estimated_nodes_activation_probabilities(self):
        samples = np.random.normal(self.means, self.sigmas)
        np.fill_diagonal(samples, 1)
        return samples


    def solve_optimization_problem(self, num_sold_items, exp_num_clicks):
        return super().solve_optimization_problem(num_sold_items,
                                                  exp_num_clicks,
                                                  self.get_estimated_nodes_activation_probabilities())


class Ecommerce5_GPTS(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def pull_arm(self):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)
        arm = np.unravel_index(np.argmax(samples), shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return arm


class Ecommerce5_GPUCB(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm = np.unravel_index(np.argmax(upper_conf), shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return arm

    def update_observations(self, arm, reward):
        super().update_observations(arm, reward)
        self.N_a[arm[0]][arm[1]] += 1
        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf
