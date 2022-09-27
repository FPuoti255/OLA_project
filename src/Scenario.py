import numpy as np

from Utils import *
from constants import *


'''

PRODUCTS:
    1: basketball
    2: t-shirt
    3: gloves
    4: book
    5: phone

USERS CLASSES:
    1: teenagers
    2: adults
    3: elderlies

'''

class Scenario:

    def __init__(self):
        self.product_prices = self.get_product_prices()
        self.users_reservation_prices = self.get_users_reservation_prices()
    
    def get_product_prices(self):       # 1 x 5
        return np.array([20,15,5,8,50])*2

    def get_users_reservation_prices(self):        # 3 x 5

        users_reservation_prices = np.zeros(shape=(NUM_OF_USERS_CLASSES,NUM_OF_PRODUCTS))
        product_prices = self.get_product_prices()
        appreciation = np.array([[5,4,-1,-1,5],[-1,-2,3,4,3],[-1,-1,5,-2,-1]])

        #for i in range(NUM_OF_USERS_CLASSES):
            #for j in range(NUM_OF_PRODUCTS):
                #users_reservation_prices[i][j] = np.random.normal(loc=product_prices[j], scale=appreciation[i][j])
        users_reservation_prices = product_prices + appreciation 
        return users_reservation_prices


    
