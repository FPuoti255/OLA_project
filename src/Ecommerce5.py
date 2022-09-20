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

        self.arms = [
            [i, j]
            for i in range(NUM_OF_PRODUCTS)
            for j in range(NUM_OF_PRODUCTS)
            if i != j
        ]
        self.n_arms = len(self.arms)
        
        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.means = np.ones(shape=self.n_arms) * 0.5
        self.sigmas = np.ones(shape=self.n_arms) * 0.2

        self.t = 0

        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.collected_rewards = []

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        )

    def update(self, arm, arm_idx, reward):
        self.t += 1
        self.update_observations(arm, arm_idx, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards.append(reward)
        self.pulled_arms.append(arm)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms)
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(self.arms), return_std=True
        )
        self.means = np.maximum(self.means, 1e-3)
        self.sigmas = np.maximum(self.sigmas, 1e-3)

    def pull_arm(self):
        pass

    def get_estimated_nodes_activation_probabilities(self):
        #a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.clip(np.random.normal(self.means, self.sigmas), a_min = 0, a_max= 1)
        estimated_nap = np.identity(n=NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap

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
        arm_idx = np.argmax(samples)
        return self.arms[arm_idx], arm_idx


class Ecommerce5_GPUCB(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.N_a = np.zeros(shape=self.n_arms)
        self.confidence_bounds = np.full(shape=self.n_arms, fill_value=np.inf)

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm_idx = np.argmax(upper_conf)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm, arm_idx, reward):
        super().update_observations(arm, arm_idx, reward)
        self.N_a[arm_idx] += 1
        self.confidence_bounds[arm_idx] = np.sqrt(
            2 * np.log(self.t) / self.N_a[arm_idx]
        )
