import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from Ecommerce import *
from constants import *
from Utils import *
from genetic_algorithm import *

############### Ecommerce implementation with One GP for all the products ###############################
# class Ecommerce3(Ecommerce):
#     def __init__(self, B_cap: float, budgets, product_prices):

#         super().__init__(B_cap, budgets, product_prices)

#         self.arms = [(prod, int(budget)) for prod in range(NUM_OF_PRODUCTS) for budget in budgets]
#         self.n_arms = len(self.arms)
#         assert(self.n_arms == self.budgets.shape[0] * NUM_OF_PRODUCTS)

#         self.t = 0
#         self.exploration_probability = 0.03

#         self.means = np.ones(shape=self.n_arms) * 0.5
#         self.sigmas = np.ones(shape=self.n_arms) * 0.5

#         self.pulled_arms = []
#         self.rewards_per_arm = [[] for _ in range(self.n_arms)]

#         self.collected_rewards = []

#         alpha = 1.50984819e-02
#         kernel = C(constant_value = 2.3, constant_value_bounds=(1e-3, 1e3)) \
#                     * RBF(length_scale = 9.32, length_scale_bounds=(1e-3, 1e3))

#         self.gaussian_process = GaussianProcessRegressor(
#             kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
#         )

#     def update(self, pulled_budgets_idxs, reward):
#         self.t += 1
#         self.update_observations(pulled_budgets_idxs, reward)
#         self.update_model()

#     def update_observations(self, pulled_budgets_idxs, reward):
#         pulled_arms = [(i, self.budgets[pulled_budgets_idxs[i]]) for i in range(len(pulled_budgets_idxs))]

#         for i in range(len(pulled_arms)):
#             arm_idx = self.arms.index(pulled_arms[i])
#             self.rewards_per_arm[arm_idx] = reward[i]
#             self.pulled_arms.append(pulled_arms[i])
#             self.collected_rewards.append(reward[i])

#     def update_model(self):
#         x = np.atleast_2d(self.pulled_arms)
#         y = np.array(self.collected_rewards)
#         self.gaussian_process.fit(x, y)

#         self.means, self.sigmas = self.gaussian_process.predict(
#             X=np.atleast_2d(self.arms), return_std=True
#         )
#         self.sigmas = np.maximum(self.sigmas, 1e-2)

#     def pull_arm(self, num_sold_items):

#         if np.random.binomial(n=1, p= 1 - self.exploration_probability):
#             estimated_reward = self.estimate_reward(num_sold_items)    
#         else :
#             value_per_click = self.compute_value_per_click(num_sold_items)
#             estimated_reward = np.multiply(
#                 np.random.random(size=(NUM_OF_PRODUCTS, self.budgets.shape[0])),
#                 np.atleast_2d(value_per_click).T
#             ) 

#         budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
#         return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)

#     def compute_value_per_click (self, num_sold_items):
#         assert(num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
#         return np.sum(np.multiply(num_sold_items, self.product_prices), axis = 1) # shape = (NUM_OF_PRODUCTS,)

#     def estimate_reward(self, num_sold_items):
#         pass



# class Ecommerce3_GPTS(Ecommerce3):
#     def __init__(self, B_cap, budgets, product_prices):
#         super().__init__(B_cap, budgets, product_prices)

#     def estimate_reward(self, num_sold_items):        
#         value_per_click = self.compute_value_per_click(num_sold_items)
#         samples = np.random.normal(loc = self.means, scale=self.sigmas).reshape((NUM_OF_PRODUCTS, self.budgets.shape[0]))        
#         estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
#         return estimated_reward


# class Ecommerce3_GPUCB(Ecommerce3):
#     def __init__(self, B_cap, budgets, product_prices):
#         super().__init__(B_cap, budgets, product_prices)

#         self.confidence_bounds = np.full(
#             shape=self.n_arms, fill_value=np.inf
#         )
#         # Number of times the arm has been pulled
#         self.N_a = np.zeros(shape=self.n_arms)

#     def update_observations(self, pulled_budgets_idxs, reward):

#         super().update_observations(pulled_budgets_idxs, reward)
        
#         pulled_arms = [(i, self.budgets[pulled_budgets_idxs[i]]) for i in range(len(pulled_budgets_idxs))]
#         for i in range(len(pulled_arms)):
#             arm_idx = self.arms.index(pulled_arms[i])
#             self.N_a[arm_idx] += 1

#         self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
#         self.confidence_bounds[self.N_a == 0] = np.inf

#     def estimate_reward(self, num_sold_items):
#         value_per_click = self.compute_value_per_click(num_sold_items)
#         estimated_reward = np.multiply(
#             np.add(self.means, self.confidence_bounds).reshape((NUM_OF_PRODUCTS, self.budgets.shape[0])),
#             np.atleast_2d(value_per_click).T
#         )
#         estimated_reward[np.isinf(estimated_reward)] = 1e4
#         return estimated_reward



############### Ecommerce implementation with One GP per product ###############################
class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)


        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        self.exploration_probability = 0.03
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms))
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms))

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        
        alpha = 3.66730775e-04
        kernel = C(constant_value = 3.41159965e+01, constant_value_bounds=(1e-3, 1e3)) \
                      * RBF(length_scale = 4.21148459e+01, length_scale_bounds=(1e-2, 1e2))
        
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                n_restarts_optimizer=9
            )
            for i in range(NUM_OF_PRODUCTS)
        ]
    

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, num_sold_items):
        if np.random.binomial(n=1, p= 1 - self.exploration_probability):
            estimated_reward = self.estimate_reward(num_sold_items)    
        else :
            value_per_click = self.compute_value_per_click(num_sold_items)
            estimated_reward = np.multiply(
                np.random.random(size=(NUM_OF_PRODUCTS, self.budgets.shape[0])),
                np.atleast_2d(value_per_click).T
            ) 

        budget_idxs_for_each_product, _ = self.dynamic_knapsack_solver(table=estimated_reward)
        return self.budgets[budget_idxs_for_each_product], np.array(budget_idxs_for_each_product)


    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):

            X = np.atleast_2d(self.pulled_arms[i]).T
            X_test = np.atleast_2d(self.budgets).T
            y = np.array(self.collected_rewards[i])

            self.gaussian_regressors[i].fit(X, y)
            
            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(
                X=X_test,
                return_std=True
            )
            self.sigmas[i] = np.maximum(self.sigmas[i], 5e-2)


   
    def update_observations(self, pulled_arm_idxs, reward):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][pulled_arm_idxs[i]].append(reward[i])
            self.pulled_arms[i].append(self.budgets[pulled_arm_idxs[i]])
            self.collected_rewards[i].append(reward[i])


    def compute_value_per_click (self, num_sold_items):
        assert(num_sold_items.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        return np.sum(np.multiply(num_sold_items, self.product_prices), axis = 1) # shape = (NUM_OF_PRODUCTS,)



class Ecommerce3_GPTS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

    def estimate_reward(self, num_sold_items):        
        value_per_click = self.compute_value_per_click(num_sold_items)
        samples = np.random.normal(loc = self.means, scale=self.sigmas)      
        estimated_reward = np.multiply(samples, np.atleast_2d(value_per_click).T)
        return estimated_reward


class Ecommerce3_GPUCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices):
        super().__init__(B_cap, budgets, product_prices)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm_idxs, reward):

        super().update_observations(pulled_arm_idxs, reward)

        for i in range(NUM_OF_PRODUCTS):
            self.N_a[i][pulled_arm_idxs[i]] += 1
            
        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf


    def estimate_reward(self, num_sold_items):
        value_per_click = self.compute_value_per_click(num_sold_items)
        estimated_reward = np.multiply(
            np.add(self.means, self.confidence_bounds),
            np.atleast_2d(value_per_click).T
        )
        estimated_reward[np.isinf(estimated_reward)] = 1e4
        return estimated_reward
