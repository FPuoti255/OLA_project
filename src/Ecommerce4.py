import numpy as np

from Ecommerce3 import *
from constants import *
from Utils import *
from SoldItemsEstimator import SoldItemsEstimator



class Ecommerce4:
    def __init__(self, algorithm : str, B_cap: float, budgets, product_prices, gp_config : dict):
        
        if algorithm == 'TS':
            self.algorithm = Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)
        elif algorithm == 'UCB':
            self.algorithm = Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_config)
        else:
            raise ValueError('Please choose one between TS or UCB')
        self.algorithm_name = algorithm
        self.items_estimator = SoldItemsEstimator()
        

    def pull_arm(self):   
        return self.algorithm.pull_arm(self.items_estimator.get_estimation())
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        self.algorithm.update(pulled_arm_idxs, reward)
        self.items_estimator.update(num_items_sold)






    

