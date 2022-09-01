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
    def __init__(self, users_reservation_prices, 
                        product_prices, 
                        click_probabilities, 
                        observations_probabilities,
                        tot_num_users):

        self.users_reservation_prices = users_reservation_prices
        self.product_prices = product_prices
        self.network = Network(adjacency_matrix= click_probabilities) 
        self.observations_probabilities = observations_probabilities
        self.tot_num_users = tot_num_users

        # ---- STEP 2 VARIABLES--------

        # For very campaign, you can imagine a maximum expected value of 𝛼_i (say 𝛼_i_bar)
        # In order to generate an array of 𝛼_i_bar with sum equal to 1, we used a multinomial distribution.
        # Notice that we needed to include also the 'competitors product', and we decided to give to all the products equal probability -> [1 / (NUM_OF_PRODUCTS+1)]
        # the 𝛼_0 is the one corresponding to the competitor(s) product
        self.alpha_bars = np.random.multinomial(self.tot_num_users, [1/(NUM_OF_PRODUCTS+1)] * (NUM_OF_PRODUCTS+1))/self.tot_num_users

        # For each campaign (product) this functions map the budget allocated into the expected number of clicks(in percentage)
        self.functions_dict = [
            lambda x : x /np.sqrt( 1 + x**2) ,
            lambda x : np.tanh(x) ,
            lambda x : x / (1 + x) ,
            lambda x : np.arctan(x) ,
            lambda x : 1 / ( 1 + np.exp(-x))  
        ]

        # -------------------------
        

    def get_users_reservation_prices(self):
        return self.users_reservation_prices
    
    def get_product_prices(self):
        return self.product_prices
    
    def get_network(self):
        return self.network

    def get_observations_probabilities(self):
        return self.observations_probabilities

    # We used this function only in the step 2
    def get_users_alphas(self, prod_id, concentration_params):
        # I expect the concentration parameter to be of the form:
        # [beta_prod, 1 - beta_prod]

        # we multiplied by 1000 to reduce the variance in the estimation
        samples = np.random.dirichlet(alpha = np.multiply(concentration_params , 1000), size = NUM_OF_USERS_CLASSES)

        # min because for each campaign we expect a maximum alpha, which is alpha_bar
        return min(np.sum(samples[:, 0]) / NUM_OF_USERS_CLASSES , self.alpha_bars[prod_id])


    def mapping_function (self, budget : float, prod_id : int):
        '''
        this function maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
        '''
        return self.functions_dict[prod_id](budget)
    
    # utility functions for us
    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.functions_dict[i](bu) for bu in budgets])