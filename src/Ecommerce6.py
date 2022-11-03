import numpy as np

from Ecommerce import *
from Ecommerce3 import Ecommerce3_GPUCB
from constants import *
from Utils import *
from SoldItemsEstimator import *



class Ecommerce6_SWUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict, tau : int):
        super().__init__(B_cap, budgets, product_prices, gp_config)

        # New attributes to be added
        self.tau=tau
        self.sold_items_estimator = SW_SoldItemsEstimator(self.tau)


    def update(self, pulled_arm_idxs, reward, sold_items):
        '''
        :pulled_arm_idxs: it is a vector of shape (NUM_OF_PRODUCTS,) containing
                          for each product the index of the budget selected in the allocation
        '''
        self.t += 1
        self.update_observations(pulled_arm_idxs, reward, sold_items)
        self.update_model()


    def update_observations(self, pulled_arm_idxs, reward, sold_items):

        for prod_id in range(NUM_OF_PRODUCTS):
            arm_idx = pulled_arm_idxs[prod_id] 

            self.pulled_arms[prod_id].append(self.budgets[arm_idx])
            self.collected_rewards[prod_id].append(reward[prod_id])

            self.N_a[prod_id][arm_idx] += 1

            assert(len(self.pulled_arms[prod_id]) == len(self.collected_rewards[prod_id]))

            if self.t >= self.tau:
                to_be_forgotten = self.pulled_arms[prod_id].pop(0)
                to_be_forgotten_idx = np.where(self.budgets == to_be_forgotten)
                self.N_a[prod_id][to_be_forgotten_idx] = max(0, self.N_a[prod_id][to_be_forgotten_idx] - 1)

                self.collected_rewards[prod_id].pop(0)
        
        self.sold_items_estimator.update(sold_items)
       

    def pull_arm(self):
        num_sold_items = self.sold_items_estimator.get_estimation()
        return super().pull_arm(num_sold_items)




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


class Ecommerce6_CDUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config : dict, M, eps, h):
        super().__init__(B_cap, budgets, product_prices, gp_config)

        self.change_detection_algorithms = [[CUSUM(M, eps, h) for _ in range(self.n_arms)] for _ in range(NUM_OF_PRODUCTS)]        
        self.time_of_detections = [0]

        self.sold_items_estimator = SoldItemsEstimator()


    def reset(self):
        self.pulled_arms = [[] for _ in range(NUM_OF_PRODUCTS)]
        self.collected_rewards = [[] for _ in range(NUM_OF_PRODUCTS)]

        self.t = 0
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))
        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf)

        self.sold_items_estimator = SoldItemsEstimator()

        for prod in range(NUM_OF_PRODUCTS):
            for arm in range(self.n_arms):
                self.change_detection_algorithms[prod][arm].reset()


    def change_detected(self, pulled_arm_idxs, reward):
        for prod_id in range(NUM_OF_PRODUCTS):
            if self.change_detection_algorithms[prod_id][pulled_arm_idxs[prod_id]].update(reward[prod_id]):
                return True

        return False

    
    def update(self, pulled_arm_idxs, reward, sold_items):

        if self.change_detected(pulled_arm_idxs, reward):
            print(f'Change detected at time t = {self.time_of_detections[-1] + self.t}')
            self.time_of_detections.append(self.t)
            self.reset()

        super().update(pulled_arm_idxs, reward)
        self.sold_items_estimator.update(sold_items)
        

    def pull_arm(self):
        num_sold_items = self.sold_items_estimator.get_estimation()
        return super().pull_arm(num_sold_items)

    def get_detections(self):
        return self.time_of_detections