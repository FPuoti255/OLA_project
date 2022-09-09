import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce3 import *

from constants import *
from Utils import *


class Ecommerce4(Ecommerce3):

    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.sold_items_means = np.ones(shape=NUM_OF_PRODUCTS)
        self.sold_items_sigmas = np.ones(shape=NUM_OF_PRODUCTS)

        self.collected_sold_items = [
            [] for i in range(NUM_OF_PRODUCTS)
        ]
        
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gaussian_regressors_sold_items = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        )

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
        
        x = np.atleast_2d(np.arange(NUM_OF_PRODUCTS)).T
        y = np.array(self.collected_sold_items)
        self.sold_items_means, self.sold_items_sigmas = self.gaussian_regressors_sold_items.predict(
            X = x, return_std=True
        )
        self.sold_items_sigmas = np.maximum(self.sold_items_sigmas, 1)


class Ecommerce4_GPTS(Ecommerce4):

    def update_observations(self, pulled_arm, reward, sold_items):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][
                int(np.where(self.budgets == pulled_arm[i])[0])
            ].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])
        
            self.collected_sold_items[i].append(sold_items[i])

    def pull_arm(self):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)
        num_items_sold = np.floor(np.random.normal(self.sold_items_means, self.sold_items_sigmas))

        samples = np.multiply(samples.copy().T , num_items_sold).T 

        arm_idxs, _ = self.dynamic_knapsack_solver(table=samples)
        return self.budgets[arm_idxs]


class Ecommerce4_GPUCB(Ecommerce4):

    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm, reward, sold_items):
        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.N_a[i][arm_idx] += 1
            self.rewards_per_arm[i][arm_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])
            
            self.collected_sold_items[i].append(sold_items[i])

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf
        

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds

        num_items_sold = np.floor(np.random.normal(self.sold_items_means, self.sold_items_sigmas))

        upper_conf = np.multiply(upper_conf.copy().T , num_items_sold).T 
        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
        return self.budgets[arm_idxs]
