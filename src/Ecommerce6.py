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
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.5
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.01

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )

        self.sold_items_means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 10
        self.sold_items_sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) *5

    def update(self, pulled_arm, reward, sold_items):
        pass

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward, sold_items):
        pass

    def update_model(self):
        pass

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm_idxs, _ = self.revisited_knapsack_solver(table=upper_conf)
        return self.budgets[arm_idxs]


    def solve_optimization_problem(self, nodes_activation_probabilities): #shape 5x5
        num_sold_items = np.random.normal(self.sold_items_means, self.sold_items_sigmas) # 5 x 21
        exp_num_clicks = np.random.normal(self.means, self.sigmas) # 5 x 21

        total_margin_for_each_product = np.multiply(num_sold_items.T, self.product_prices).T  # shape = 5x21

        value_per_click = np.ones_like(exp_num_clicks)
        for i in range(self.budgets.shape[0]):
            value_per_click[ : , i] = np.dot(nodes_activation_probabilities, total_margin_for_each_product[:, i])
        
        assert(value_per_click.shape == (NUM_OF_PRODUCTS, self.budgets.shape[0]))

        exp_reward = np.multiply(exp_num_clicks, value_per_click)

        budgets_indexes, reward = self.dynamic_knapsack_solver(
            table=exp_reward
        )
        optimal_allocation = self.budgets[budgets_indexes]

        return optimal_allocation, reward


class Ecommerce6_SWUCB(Ecommerce6):
    def __init__(self, B_cap, budgets, product_prices, tau: int):
        super().__init__(B_cap, budgets, product_prices)

        self.tau=tau

        self.pulled_arms=np.full(
            shape=(NUM_OF_PRODUCTS, self.tau), fill_value=np.nan)
        self.rewards_per_arm=np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau), fill_value=np.nan)
        # Number of times the arm has been pulled represented with a binary vector
        self.N_a=np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau))


        self.reward_sold_items=np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms, self.tau), fill_value=np.nan)


    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        self.update_observations(pulled_arm, reward, sold_items)
        self.update_model()

    def update_observations(self, pulled_arm, reward, sold_items):

        slot_idx=self.t % self.tau

        for i in range(NUM_OF_PRODUCTS):
            arm_idx=int(np.where(self.budgets == pulled_arm[i])[0])
            non_pulled_arm_idxs=np.nonzero(
                np.in1d(self.budgets, np.setdiff1d(self.budgets, pulled_arm[i])))[0]
            
            self.N_a[i][arm_idx][slot_idx]=1
            self.N_a[i][non_pulled_arm_idxs][slot_idx]=0
            self.rewards_per_arm[i][arm_idx][slot_idx]=reward[i]
            self.rewards_per_arm[i][non_pulled_arm_idxs][slot_idx]=np.nan
            
            self.pulled_arms[i][slot_idx]=pulled_arm[i]
            
            self.reward_sold_items[i][arm_idx][slot_idx]=sold_items[i]
            self.reward_sold_items[i][non_pulled_arm_idxs][slot_idx]=np.nan




    def update_model(self):
        for i in range(0, NUM_OF_PRODUCTS):
            for j in range(0, self.n_arms):
                self.means[i][j]=np.nanmean(self.rewards_per_arm[i][j])
                self.sigmas[i][j]=np.nanstd(self.rewards_per_arm[i][j])
                self.sold_items_means[i][j]=np.nanmean(self.reward_sold_items[i][j])
                self.sold_items_sigmas[i][j]=np.nanstd(self.reward_sold_items[i][j])

        self.sigmas=np.maximum(self.sigmas, 1e-2)
        self.sold_items_sigmas=np.maximum(self.sold_items_sigmas, 1)

        self.confidence_bounds=np.sqrt(
            2 * np.log(self.t) / np.sum(self.N_a, axis=2))




class CUSUM:
    '''
    Liu, Fang & Lee, Joohyun & Shroff, Ness. (2017).
    A Change-Detection Based Framework for Piecewise-Stationary Multi-Armed Bandit Problem.
    Proceedings of the AAAI Conference on Artificial Intelligence. 32. 10.1609/aaai.v32i1.11746.

    (https://www.researchgate.net/publication/321025435_A_Change-Detection_Based_Framework_for_Piecewise-Stationary_Multi-Armed_Bandit_Problem)
    '''

    def __init__(self, M, eps, h):
        self.M=M
        self.eps=eps
        self.h=h
        self.t=0
        self.reference=0
        self.g_plus=0
        self.g_minus=0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0
        else:
            s_plus=(sample - self.reference) - self.eps
            s_minus=-(sample - self.reference) - self.eps
            self.g_plus=max(0, self.g_plus + s_plus)
            self.g_minus=max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t=0
        self.g_minus=0
        self.g_plus=0


class Ecommerce6_CDUCB(Ecommerce6):
    def __init__(self, B_cap: float, budgets, product_prices, M, eps, h):
        super().__init__(B_cap, budgets, product_prices)

        self.change_detection=[[CUSUM(M, eps, h) for i in range(
            self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.change_detection_sold_items=[[CUSUM(M, eps, h) for i in range(
            self.n_arms)] for j in range(NUM_OF_PRODUCTS)]


        self.pulled_arms=[[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm=[
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]

        self.rewards_sold_items=[
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)]


    def update(self, pulled_arm, reward, sold_items):
        self.t += 1
        for i in range(NUM_OF_PRODUCTS):
            arm_idx=int(np.where(self.budgets == pulled_arm[i])[0])

            if self.change_detection[i][arm_idx].update(reward[i]):
                print('CUSUM: change detected at time ', self.t)
                self.rewards_per_arm[i][arm_idx]=[]
                self.change_detection[i][arm_idx].reset()

            if self.change_detection_sold_items[i][arm_idx].update(sold_items[i]):
                self.rewards_sold_items[i][arm_idx]=[]
                self.change_detection_sold_items[i][arm_idx].reset()

            self.update_observations(i, arm_idx, reward[i], sold_items[i])

        self.update_model()

    def update_observations(self, prod_id, arm_idx, arm_reward, prod_sold_items):
        self.rewards_per_arm[prod_id][arm_idx].append(arm_reward)
        self.pulled_arms[prod_id].append(self.budgets[arm_idx])

        self.rewards_sold_items[prod_id][arm_idx].append(prod_sold_items)


    def update_model(self):
        # equivalent to n_t in the paper
        total_valid_rewards=np.zeros(shape=NUM_OF_PRODUCTS)
        for prod_id in range(0, NUM_OF_PRODUCTS):
            total_valid_rewards[prod_id]=np.sum(
                [len(x) for x in self.rewards_per_arm[prod_id]])

            for arm_idx in range(0, self.n_arms):
                self.means[prod_id][arm_idx]=np.mean(
                    self.rewards_per_arm[prod_id][arm_idx])
                self.sigmas[prod_id][arm_idx]=np.std(
                    self.rewards_per_arm[prod_id][arm_idx])

                n_reward=len(self.rewards_per_arm[prod_id][arm_idx])
                self.confidence_bounds[prod_id][arm_idx]=np.sqrt(
                    2 * np.log(total_valid_rewards[prod_id]) / n_reward) if n_reward > 0 else np.inf

                self.sold_items_means[prod_id][arm_idx]=np.mean(
                    self.rewards_sold_items[prod_id][arm_idx])
                self.sold_items_sigmas[prod_id][arm_idx]=np.std(
                    self.rewards_sold_items[prod_id][arm_idx])

        self.sigmas=np.maximum(self.sigmas, 1e-2)
        self.sold_items_sigmas=np.maximum(self.sold_items_sigmas, 1)
