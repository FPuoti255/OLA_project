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
        self.n_phases = n_phases
        self.phase_len = phase_len

        self.current_phase = 0

        assert(users_reservation_prices.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(graph_weights.shape == (self.n_phases, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert(alpha_bars.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS + 1) )
        assert(users_poisson_parameters.shape == (self.n_phases, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        self.users_reservation_prices = users_reservation_prices
        self.networks = [Network(adjacency_matrix=graph_weights[i]) for i in range(graph_weights.shape[0])]
        self.alpha_bars = alpha_bars
        self.users_poisson_parameters = users_poisson_parameters

        self.expected_users_alpha = None
        self.expected_reward = None


    def get_current_phase(self):
        return self.current_phase

    def get_alpha_bars(self):
        return self.alpha_bars[self.current_phase]
    
    def get_users_reservation_prices(self):
        return self.users_reservation_prices[self.current_phase]
    
    def get_users_poisson_parameters(self):
        return self.users_poisson_parameters[self.current_phase]

    def get_network(self):
        return self.networks[self.current_phase]

    def mapping_function(self, prod_id, budget):
        '''
        @returns a map for each user class. shape = (NUM_OF_USER_CLASSES, 1)
        '''
        alpha_bars = self.alpha_bars[self.current_phase]
        return np.clip(a = 2 * alpha_bars[:, prod_id + 1] / (1 + 1/budget), a_min=0.001, a_max=0.999)


    def compute_users_alpha(self, budgets: np.ndarray):
        '''
        :budgets: must be passed normalized ( between 0 and 1), thus budgets / B_cap
        :return: the expected alpha for each couple (prod_id, budget_allocated)
        '''
        bdgts = budgets.copy() / budgets[-1]
        exp_user_alpha = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, budgets.shape[0]))
        alpha_bars = self.alpha_bars[self.current_phase]

        for user_class in range(NUM_OF_USERS_CLASSES):
            for prod_id in range(NUM_OF_PRODUCTS):
                for j in range(1, bdgts.shape[0]):

                    conc_params = self.mapping_function(prod_id, bdgts[j])

                    exp_user_alpha[user_class, prod_id, j] = min(
                        self.rng.dirichlet(
                            np.multiply([conc_params[user_class], 1 - conc_params[user_class]], 100)
                        )[0],
                        alpha_bars[user_class, prod_id + 1]
                    )

                exp_user_alpha[user_class, prod_id] = np.sort(exp_user_alpha[user_class, prod_id])

        self.expected_users_alpha = exp_user_alpha


    def round_step6(self, pulled_arm, pulled_arm_idxs, num_sold_items, end_phase = False):

        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))        

        alpha, reward = self.round_step3(pulled_arm, pulled_arm_idxs)

        aggregated_num_sold_items = np.sum(num_sold_items, axis = (0,1))
        assert(aggregated_num_sold_items.shape == (NUM_OF_PRODUCTS,))

        alpha_bars = self.alpha_bars[self.current_phase]
        real_sold_items = aggregated_num_sold_items * alpha / np.sum(alpha_bars, axis = 0)[1:]

        if end_phase : #in this way also the second learner to pull will have the same phase parameters
            self.t += 1
            self.current_phase = int(np.floor(self.t / self.phase_len))

        return alpha, reward, real_sold_items


