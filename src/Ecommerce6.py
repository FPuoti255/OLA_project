import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *

class Ecommerce6(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices):

        super().__init__(B_cap, budgets, product_prices)


        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0

        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) 
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) 

        self.sold_items_means = np.ones(shape=NUM_OF_PRODUCTS)
        self.sold_items_sigmas = np.ones(shape=NUM_OF_PRODUCTS)


    
    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        self.update_observations(pulled_arm, reward, sold_items)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward, sold_items):
        pass

    def update_model(self):
        pass

    def pull_arm(self):
        pass

    def dynamic_knapsack_solver(self, table):
        """
        In this phase we do not need to subtract the budgets to the final row,
        since we use the dynamic algorithm find the allocation that comply with the 
        UCB pulling rules and the B_cap
        """
        table_opt, max_pointer = self.compute_table(table)
        return self.choose_best(table_opt, max_pointer)



class Ecommerce6_SWUCB(Ecommerce6):
    def __init__(self, B_cap, budgets, product_prices, tau:int ):
        super().__init__(B_cap, budgets, product_prices)
        
        self.tau = tau

        self.pulled_arms = np.full(shape = (NUM_OF_PRODUCTS, self.tau), fill_value=np.nan)
        self.rewards_per_arm = np.full(shape= (NUM_OF_PRODUCTS, self.n_arms, self.tau), fill_value= np.nan)        
        
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]


        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )

        # Number of times the arm has been pulled represented with a binary vector
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau))

        self.reward_sold_items = np.full(shape=(NUM_OF_PRODUCTS, self.tau), fill_value=np.nan)
        self.collected_sold_items = [
            [] for i in range(NUM_OF_PRODUCTS)
        ]



    def update_observations(self, pulled_arm, reward, sold_items):

        slot_idx = self.t % self.tau

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            non_pulled_arm_idxs = np.nonzero(np.in1d(self.budgets, np.setdiff1d(self.budgets, pulled_arm[i])))[0]
            
            self.N_a[i][arm_idx][slot_idx] = 1
            self.N_a[i][non_pulled_arm_idxs][slot_idx] = 0
            self.rewards_per_arm[i][arm_idx][slot_idx] = reward[i]
            self.rewards_per_arm[i][non_pulled_arm_idxs][slot_idx] = np.nan

            self.pulled_arms[i][slot_idx] = pulled_arm[i]

            self.reward_sold_items[i][slot_idx] = sold_items[i]
            
            self.collected_rewards[i].append(reward[i])
            self.collected_sold_items[i].append(sold_items[i])

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / np.sum(self.N_a, axis=2))
        # We do not set np.inf since the division by 0 yields np.inf by default in numpy
    
    def update_model(self):
        for i in range(0,NUM_OF_PRODUCTS):
            for j in range(0,self.n_arms):
                self.means[i][j] = np.nanmean(self.rewards_per_arm[i][j])
                self.sigmas[i][j] = np.nanstd(self.rewards_per_arm[i][j])
            self.sold_items_means[i] = np.nanmean(self.reward_sold_items[i])
            self.sold_items_sigmas[i] = np.nanstd(self.reward_sold_items[i])

        self.sigmas = np.maximum(self.sigmas, 1e-2)
        self.sold_items_sigmas = np.maximum(self.sold_items_sigmas, 1)
    
    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds

        num_items_sold = np.floor(np.random.normal(self.sold_items_means, self.sold_items_sigmas))

        upper_conf = np.multiply(upper_conf.copy().T , num_items_sold).T 
        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
        return self.budgets[arm_idxs]

