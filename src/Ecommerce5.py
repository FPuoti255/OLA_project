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

        self.t = 0

        self.pulled_arms = np.array([])
        self.rewards_per_arm = x = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

    def update(self, arm_idx, reward):
        self.t += 1
        self.update_observations(arm_idx, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm_idx, reward):
        pass

    def update_model(self):
        pass

    def pull_arm(self):
        pass

    def get_estimated_nodes_activation_probabilities(self):
        pass


class Ecommerce5_GPTS(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.a = np.ones(shape=self.n_arms, dtype=np.int32)
        self.b = np.ones(shape=self.n_arms, dtype=np.int32)

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        )

    def pull_arm(self):
        samples = np.random.beta(a=self.a, b=self.b)
        arm_idx = np.argmax(samples)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms = np.append(self.pulled_arms, arm_idx)

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        means, sigmas = self.gp.predict(
            np.atleast_2d(np.arange(0, self.n_arms)).T, return_std=True
        )
        sigmas = np.maximum(sigmas, 1e-2)

        self.a, self.b = compute_beta_parameters(means, sigmas)

    def get_estimated_graph_weights(self):
        samples = np.random.beta(a=self.a, b=self.b)
        estimated_nap = np.identity(n=NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap


class Ecommerce5_GPUCB(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.means = np.ones(shape=self.n_arms) * 0.5
        self.sigmas = np.ones(shape=self.n_arms) * 2

        self.confidence_bounds = np.full(shape=self.n_arms, fill_value=np.inf)
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=self.n_arms)

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        )

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm_idx = np.argmax(upper_conf)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm_idx, reward):
        self.N_a[arm_idx] += 1
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms = np.append(self.pulled_arms, arm_idx)

        self.confidence_bounds[arm_idx] = np.sqrt(
            2 * np.log(self.t) / self.N_a[arm_idx]
        )

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(np.arange(0, self.n_arms)).T, return_std=True
        )

    def get_estimated_graph_weights(self):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)
        estimated_nap = np.identity(n=NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap
