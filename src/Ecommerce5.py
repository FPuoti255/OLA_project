import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Utils import *


class Ecommerce5(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, tot_num_users):

        super().__init__(
            B_cap=B_cap,
            budgets=budgets,
            product_prices=product_prices,
            tot_num_users=tot_num_users,
        )

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
        self.update_model(arm_idx, reward)

    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm_idx, reward):
        pass

    def update_model(self, arm_idx, reward):
        pass

    def pull_arm(self, nodes_activation_probabilities):
        pass


class Ecommerce5_TS(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)

        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.a = np.ones(shape=self.n_arms, dtype=np.int32)
        self.b = np.ones(shape=self.n_arms, dtype=np.int32)

    def pull_arm(self):
        samples = np.random.beta(a=self.a, b=self.b)
        arm_idx = np.argmax(samples)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms = np.append(self.pulled_arms, self.arms[arm_idx])

    def update_model(self, arm_idx, reward):
        self.a[arm_idx] += int(reward)
        self.b[arm_idx] += int(1 - reward)

    def get_estimated_nodes_activation_probabilities(self):
        samples = np.random.beta(a=self.a, b=self.b)
        estimated_nap = np.identity(n=NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap


class Ecommerce5_GPTS(Ecommerce5):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)

        # I'm generating a distribution for each possible PRODUCT-PRODUCT edge
        self.means = np.ones(shape=self.n_arms, dtype=np.int32) * 0.5
        self.sigmas = np.ones(shape=self.n_arms, dtype=np.int32) * 2

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
        )

    def pull_arm(self):
        a = np.multiply(self.means, self.sigmas)
        b = np.multiply(self.sigmas, 1 - self.means)
        a[np.where(a == 0)] = 0.00001
        b[np.where(b == 0)] = 0.00001

        samples = np.random.beta(a=a, b=b)
        arm_idx = np.argmax(samples)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms = np.append(self.pulled_arms, arm_idx)

    def update_model(self, arm_idx, reward):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(np.arange(0, self.n_arms)).T, return_std=True
        )
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def get_estimated_nodes_activation_probabilities(self):
        a = np.multiply(self.means, self.sigmas)
        b = np.multiply(self.sigmas, (1 - self.means))
        a[np.where(a == 0)] = 0.00001
        b[np.where(b == 0)] = 0.00001

        samples = np.random.beta(a=a, b=b)
        estimated_nap = np.identity(n=NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap
