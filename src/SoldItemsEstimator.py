import numpy as np
from constants import *

class SoldItemsEstimator:
    def __init__(self) -> None:
        self.sold_items_means = np.ones(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)) * 20.0
        self.collected_sold_items = [self.sold_items_means.copy()]

    def get_estimation(self):
        return self.sold_items_means
    
    def update(self, num_items_sold):
        self.collected_sold_items.append(num_items_sold)
        self.sold_items_means = np.mean(self.collected_sold_items, axis = 0)

class SW_SoldItemsEstimator(SoldItemsEstimator):
    def __init__(self, tau : int) -> None:
        super().__init__()
        self.tau = tau
    
    def update(self, num_items_sold):
        if (len(self.collected_sold_items) + 1 >= self.tau):
            # we remove the oldest element
            self.collected_sold_items.pop(0)
        return super().update(num_items_sold)