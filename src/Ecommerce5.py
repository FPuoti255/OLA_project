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
        self.means = np.zeros(shape=self.n_arms)
        self.sigmas = np.ones(shape=self.n_arms)

        self.t = 0

        self.pulled_arms = []
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]

    def update(self, arm, arm_idx, reward):
        self.t += 1
        self.update_observations(arm, arm_idx, reward)
        self.update_model(arm_idx)

    # The methods below will be implemented in the sub-classes
    def update_observations(self, arm, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.pulled_arms.append(arm)


    def update_model(self, arm_idx):
        self.means[arm_idx] = np.mean(self.rewards_per_arm[arm_idx])
        self.sigmas[arm_idx] = np.std(self.rewards_per_arm[arm_idx])

    def pull_arm(self):
        pass

    def get_estimated_nodes_activation_probabilities(self):
        samples = np.random.normal(self.means, self.sigmas)
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
        self.N_a[arm_idx] += 1
        super().update_observations(arm, arm_idx, reward)
        self.confidence_bounds[arm_idx] = np.sqrt(
            2 * np.log(self.t) / self.N_a[arm_idx]
        )
