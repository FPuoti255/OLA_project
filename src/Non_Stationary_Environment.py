import numpy as np

from constants import *
from Utils import *
from Network import Network
from Environment import *


class Non_Stationary_Environment(Environment):

    def __init__(self, users_reservation_prices, 
                        graph_weights, 
                        alpha_bars, 
                        users_poisson_parameters,
                        n_phases, 
                        phase_len):
        
        self.rng = np.random.default_rng(12345)

        self.t = 0

        self.current_phase = 0
        self.n_phases = n_phases
        self.phase_len = phase_len

        assert(users_reservation_prices.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(graph_weights.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert(alpha_bars.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS + 1) )
        assert(users_poisson_parameters.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        self.environments = [ 
            Environment(users_reservation_prices[i], graph_weights.copy(), alpha_bars[i], users_poisson_parameters[i])    
            for i in range(self.n_phases)]
        

    def get_current_phase(self):
        return self.current_phase

    def get_alpha_bars(self):
        return self.environments[self.current_phase].get_alpha_bars()
    
    def get_users_reservation_prices(self):
        return self.environments[self.current_phase].get_users_reservation_prices()
    
    def get_users_poisson_parameters(self):
        return self.environments[self.current_phase].get_users_poisson_parameters()

    def get_network(self):
        return self.environments[self.current_phase].get_network()

    def compute_clairvoyant_reward(self, num_sold_items, product_prices, budgets):
        return self.environments[self.current_phase].compute_clairvoyant_reward(num_sold_items, product_prices, budgets)
    

    def round_step6(self, pulled_arm, pulled_arm_idxs, num_sold_items, optimal_arm, end_phase = False):

        alpha, reward, real_sold_items = self.environments[self.current_phase].round_step4(pulled_arm, pulled_arm_idxs, num_sold_items, optimal_arm)
        
        if end_phase : #in this way also the second learner to pull will have the same phase parameters
            self.t += 1
            self.current_phase = int(np.floor(self.t / self.phase_len))

        return alpha, reward, real_sold_items


