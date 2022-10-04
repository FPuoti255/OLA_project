import numpy as np

from Utils import *
from constants import *


'''

PRODUCTS:
    1: basketball
    2: t-shirt
    3: gloves
    4: encyclopedia
    5: phone

USERS CLASSES:
    1: teenagers
    2: adults
    3: elders

'''

class Scenario:

    def __init__(self):
        self.product_prices = self.get_product_prices()
        self.users_reservation_prices = self.get_users_reservation_prices()
    
    def get_product_prices(self):       # 1 x 5
        return np.array([150,70,15,250,750])

    def get_users_reservation_prices(self):        # 3 x 5

        users_reservation_prices = np.zeros(shape=(NUM_OF_USERS_CLASSES,NUM_OF_PRODUCTS))
        product_prices = self.get_product_prices()
        appreciation = np.array([
                                    [50,30,-10,-250,250],
                                    [-15,-20,5,300,-250],
                                    [-150,-50,35,400,-720]
                                ])

        # TODO : implements the users_reservation_prices sampling from a normal distribution
        # with loc = product_prices and scale = sqrt(appreciation)

        users_reservation_prices = product_prices + appreciation 
        return users_reservation_prices


    
