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
        
        self.alpha = 0.01

        self.t = 0

        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms))
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms))

        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )



        self.sold_items_means = np.ones(shape=NUM_OF_PRODUCTS)
        self.sold_items_sigmas = np.ones(shape=NUM_OF_PRODUCTS)

        self.collected_sold_items = [[] for i in range(NUM_OF_PRODUCTS)]



    def update(self, pulled_arm, reward, sold_items):
        pass

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward, sold_items):
        pass

    def update_model(self):
        pass

    def pull_arm(self):
        if np.random.binomial(1, 1-self.alpha):
            upper_conf = self.means + self.confidence_bounds

            num_items_sold = np.floor(np.random.normal(
                self.sold_items_means, self.sold_items_sigmas))

            upper_conf = np.multiply(upper_conf.copy().T, num_items_sold).T
            arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
            return self.budgets[arm_idxs]
        else:
            pulled_arm = np.zeros(shape=NUM_OF_PRODUCTS)
            current_budget_sum = 0
            for i in range(NUM_OF_PRODUCTS):
                found = False
                while not found:
                    arm = np.random.choice(self.budgets)
                    if current_budget_sum + arm <=self.B_cap:
                        pulled_arm[i] = arm
                        current_budget_sum +=arm
                        found = True
            return pulled_arm

            

    def dynamic_knapsack_solver(self, table):
        """
        In this phase we do not need to subtract the budgets to the final row,
        since we use the dynamic algorithm find the allocation that comply with the 
        UCB pulling rules and the B_cap
        """
        table_opt, max_pointer = self.compute_table(table)
        return self.choose_best(table_opt, max_pointer)


class Ecommerce6_SWUCB(Ecommerce6):
    def __init__(self, B_cap, budgets, product_prices, tau: int):
        super().__init__(B_cap, budgets, product_prices)

        self.tau = tau

        self.pulled_arms = np.full(
            shape=(NUM_OF_PRODUCTS, self.tau), fill_value=np.nan)
        self.rewards_per_arm = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau), fill_value=np.nan)
        # Number of times the arm has been pulled represented with a binary vector
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau))


        self.reward_sold_items = np.full(
            shape=(NUM_OF_PRODUCTS, self.tau), fill_value=np.nan)


    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        self.update_observations(pulled_arm, reward, sold_items)
        self.update_model()

    def update_observations(self, pulled_arm, reward, sold_items):

        slot_idx = self.t % self.tau

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            non_pulled_arm_idxs = np.nonzero(
                np.in1d(self.budgets, np.setdiff1d(self.budgets, pulled_arm[i])))[0]

            self.N_a[i][arm_idx][slot_idx] = 1
            self.N_a[i][non_pulled_arm_idxs][slot_idx] = 0
            self.rewards_per_arm[i][arm_idx][slot_idx] = reward[i]
            self.rewards_per_arm[i][non_pulled_arm_idxs][slot_idx] = np.nan

            self.pulled_arms[i][slot_idx] = pulled_arm[i]

            self.reward_sold_items[i][slot_idx] = sold_items[i]

            self.collected_rewards[i].append(reward[i])
            self.collected_sold_items[i].append(sold_items[i])

        
    def update_model(self):
        for i in range(0, NUM_OF_PRODUCTS):
            for j in range(0, self.n_arms):
                self.means[i][j] = np.nanmean(self.rewards_per_arm[i][j])
                self.sigmas[i][j] = np.nanstd(self.rewards_per_arm[i][j])
            self.sold_items_means[i] = np.nanmean(self.reward_sold_items[i])
            self.sold_items_sigmas[i] = np.nanstd(self.reward_sold_items[i])

        self.sigmas = np.maximum(self.sigmas, 1e-2)
        self.sold_items_sigmas = np.maximum(self.sold_items_sigmas, 1)

        self.confidence_bounds = np.sqrt(
            2 * np.log(self.t) / np.sum(self.N_a, axis=2))
        # We do not set np.inf since the division by 0 yields np.inf by default in numpy





class CUSUM:
    '''
    Liu, Fang & Lee, Joohyun & Shroff, Ness. (2017). 
    A Change-Detection Based Framework for Piecewise-Stationary Multi-Armed Bandit Problem. 
    Proceedings of the AAAI Conference on Artificial Intelligence. 32. 10.1609/aaai.v32i1.11746. 

    (https://www.researchgate.net/publication/321025435_A_Change-Detection_Based_Framework_for_Piecewise-Stationary_Multi-Armed_Bandit_Problem)
    '''

    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0


class Ecommerce6_CDUCB(Ecommerce6):
    def __init__(self, B_cap: float, budgets, product_prices, M, eps, h):
        super().__init__(B_cap, budgets, product_prices)

        self.change_detection = [[CUSUM(M, eps, h) for i in range(
            self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.detections = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.change_detection_sold_items = [
            CUSUM(M, eps, h) for i in range(NUM_OF_PRODUCTS)]

        self.detections_sold_items = [[] for i in range(NUM_OF_PRODUCTS)]


        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]


        self.rewards_sold_items = [[] for i in range(NUM_OF_PRODUCTS)]


    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])

            if self.change_detection[i][arm_idx].update(reward[i]):
                self.detections[i][arm_idx].append(self.t)
                self.rewards_per_arm[i][arm_idx] = []
                self.change_detection[i][arm_idx].reset()

            if self.change_detection_sold_items[i].update(sold_items[i]):
                self.detections_sold_items[i].append(self.t)
                self.rewards_sold_items[i] = []
                self.change_detection_sold_items[i].reset()

            self.update_observations(i, arm_idx, reward[i], sold_items[i])

        self.update_model()

    def update_observations(self, prod_id, arm_idx, arm_reward, prod_sold_items):
        self.rewards_per_arm[prod_id][arm_idx].append(arm_reward)
        self.pulled_arms[prod_id].append(self.budgets[arm_idx])
        self.collected_rewards[prod_id].append(arm_reward)

        self.rewards_sold_items[prod_id].append(prod_sold_items)
        self.collected_sold_items[prod_id].append(prod_sold_items)

    def update_model(self):
        # equivalent to n_t in the paper
        total_valid_rewards = np.zeros(shape=NUM_OF_PRODUCTS)
        for prod_id in range(0, NUM_OF_PRODUCTS):
            total_valid_rewards[prod_id] = np.sum(
                [len(x) for x in self.rewards_per_arm[prod_id]])

            for arm_idx in range(0, self.n_arms):
                self.means[prod_id][arm_idx] = np.mean(
                    self.rewards_per_arm[prod_id][arm_idx])
                self.sigmas[prod_id][arm_idx] = np.std(
                    self.rewards_per_arm[prod_id][arm_idx])

                n_reward = len(self.rewards_per_arm[prod_id][arm_idx])
                self.confidence_bounds[prod_id][arm_idx] = np.sqrt(
                    2 * np.log(total_valid_rewards[prod_id]) / n_reward) if n_reward > 0 else np.inf

            self.sold_items_means[prod_id] = np.mean(
                self.rewards_sold_items[prod_id])
            self.sold_items_sigmas[prod_id] = np.std(
                self.rewards_sold_items[prod_id])

        self.sigmas = np.maximum(self.sigmas, 1e-2)
        self.sold_items_sigmas = np.maximum(self.sold_items_sigmas, 1)