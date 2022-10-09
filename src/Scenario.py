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
        return np.array([25,14,30,50,40]) * 2

    def get_users_reservation_prices(self):        # 3 x 5
        users_reservation_prices = self.get_product_prices() + np.array([[20,13,13,11,14],[6,10,12,15,10],[-13,16,-21,-24,-15]])
        return users_reservation_prices
     


    
