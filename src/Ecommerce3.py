import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)


        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.5
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 2

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
            )
            for i in range(NUM_OF_PRODUCTS)
        ]
    

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward):
        pass

    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(
                X=np.atleast_2d(self.budgets).T, return_std=True
            )
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)

    def pull_arm(self, nodes_activation_probabilities):
        pass

    def dynamic_knapsack_solver(self, table):
        """
        In this phase we do not need to subtract the budgets to the final row,
        since we use the dynamic algorithm find the allocation that comply with the 
        TS/UCB pulling rules and the B_cap
        """
        table_opt, max_pointer = self.compute_table(table)
        return self.choose_best(table_opt, max_pointer)



class Ecommerce3_GPTS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def update_observations(self, pulled_arm, reward):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][
                int(np.where(self.budgets == pulled_arm[i])[0])
            ].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

    def pull_arm(self, num_items_sold):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)

        # Multiplied by the number of items sold 
        samples = np.multiply(samples.copy().T , num_items_sold).T 

        arm_idxs, _ = self.dynamic_knapsack_solver(table=samples)

        return self.budgets[arm_idxs]


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm, reward):

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.N_a[i][arm_idx] += 1
            self.rewards_per_arm[i][arm_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf

    def pull_arm(self, num_items_sold):
        upper_conf = self.means + self.confidence_bounds
        
        # Multiplied by the number of items sold 
        upper_conf = np.multiply(upper_conf.copy().T, num_items_sold).T 
        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
        return self.budgets[arm_idxs]

