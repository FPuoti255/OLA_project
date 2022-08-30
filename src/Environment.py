import numpy as np
from Constants import *
from Network import Network
import random
from tqdm import tqdm

debug = False

def log(msg):
    if(debug):
        print(msg)

class Environment():
    def __init__(self, users_reservation_prices, product_prices, click_probabilities, observations_probabilities):
        self.users_reservation_prices = users_reservation_prices
        self.product_prices = product_prices
        self.network = Network(adjacency_matrix= click_probabilities) 
        self.observations_probabilities = observations_probabilities
        

    def get_users_reservation_prices(self):
        return self.users_reservation_prices
    
    def get_product_prices(self):
        return self.product_prices
    
    def get_network(self):
        return self.network

    def get_observations_probabilities(self):
        return self.observations_probabilities

    def get_users_alphas(self, a):
        return np.random.dirichlet(alpha = a, size = NUM_OF_USERS_CLASSES)