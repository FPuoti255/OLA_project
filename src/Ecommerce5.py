import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from Utils import *


class Ecommerce5(Ecommerce):
    def __init__(self, B_cap : float, budgets, product_prices, tot_num_users):
        
        super().__init__(B_cap=B_cap, budgets = budgets, product_prices=product_prices, tot_num_users = tot_num_users)

        
        self.arms = [[i, j] for i in range(NUM_OF_PRODUCTS) for j in range(NUM_OF_PRODUCTS) if i!=j]
        self.n_arms = len(self.arms)

        self.t = 0

        self.pulled_arms = []
        self.rewards_per_arm = x =[[] for i in range(self.n_arms)]
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
        self.a = np.ones(shape = self.n_arms, dtype=np.int32)
        self.b = np.ones(shape = self.n_arms, dtype=np.int32)

    def pull_arm(self):
        samples = np.random.beta(a=self.a, b=self.b)
        arm_idx = np.argmax(samples)
        return self.arms[arm_idx], arm_idx

    def update_observations(self, arm_idx, reward):
        self.rewards_per_arm[arm_idx].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self, arm_idx, reward):
        self.a[arm_idx] += int(reward)
        self.b[arm_idx] += int(1 - reward)

    def get_estimated_nodes_activation_probabilities(self):
        samples = np.random.beta(a=self.a, b=self.b)
        estimated_nap = np.identity(n = NUM_OF_PRODUCTS)
        for i in range(self.n_arms):
            row, col = self.arms[i][0], self.arms[i][1]
            estimated_nap[row][col] = samples[i]
        return estimated_nap




