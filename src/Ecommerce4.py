from Ecommerce3 import *
from SoldItemsEstimator import SoldItemsEstimator


class Ecommerce4_GPTS(Ecommerce3_GPTS):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)
        self.items_estimator = SoldItemsEstimator()    

    def pull_arm(self):   
        return super().pull_arm(self.items_estimator.get_estimation())
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        super().update(pulled_arm_idxs, reward)
        self.items_estimator.update(num_items_sold)


class Ecommerce4_GPUCB(Ecommerce3_GPUCB):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)
        self.items_estimator = SoldItemsEstimator()    

    def pull_arm(self):   
        return super().pull_arm(self.items_estimator.get_estimation())
    
    def update(self, pulled_arm_idxs, reward, num_items_sold):
        super().update(pulled_arm_idxs, reward)
        self.items_estimator.update(num_items_sold)





    

