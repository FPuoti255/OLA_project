import numpy as np
import random
from tqdm import tqdm

from constants import *
from Utils import *
from Network import Network


class Environment:
    def __init__(
        self,
        users_reservation_prices,
        click_probabilities, # == network weights
        users_alpha
    ):
        self.rng = np.random.default_rng(12345)

        self.users_reservation_prices = users_reservation_prices
        self.users_alpha = users_alpha

        self.network = Network(adjacency_matrix=click_probabilities)

        self.nodes_activation_probabilities = None
        self.dirichlet_variance_keeper = 100


        # For each campaign (product) this functions map the budget allocated into the expected number of clicks(in percentage)
        self.functions_dict = [
            lambda x: 0.5 if x > 0.5 else x+0.001,
            lambda x: 0.001 if x<0.2 else (np.exp(x**2)-1 if x>= 0.2 and x<=0.7 else 0.64) ,
            lambda x: min(x + 0.001, 0.99),
            lambda x: np.log(x+1) + 0.001,
            lambda x: 1 / (1 + np.exp(- (x ** 4))) - 0.499,
        ]

    def get_users_reservation_prices(self):
        return self.users_reservation_prices

    def get_users_alpha(self):
        return self.users_alpha

    def get_network(self):
        return self.network

    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.functions_dict[i](bu) for bu in budgets])

    # -----------------------------------------------
    # --------STEP 2 ENVIRONMENT FUNCTIONS-----------
    def estimate_expected_user_alpha(self, budgets: np.ndarray):
        '''
        budgets must be passed normalized ( between 0 and 1), thus budgets / B_cap

        returns the expected alpha for each couple (prod_id, budget_allocated)
        '''
        exp_user_alpha = np.zeros(shape=(NUM_OF_PRODUCTS, budgets.shape[0]))

        for prod_id in range(NUM_OF_PRODUCTS):
            for j in range(budgets.shape[0]):
                
                # maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
                conc_param =  self.functions_dict[prod_id](budgets[j])

                # we multiplied by 1000 to reduce the variance in the estimation
                samples = self.rng.dirichlet(
                    alpha=np.multiply([conc_param, 1 - conc_param], self.dirichlet_variance_keeper), size=NUM_OF_USERS_CLASSES
                )
                
                # min because for each campaign we expect a maximum alpha, which is alpha_bar
                exp_user_alpha[prod_id][j] = min(
                    np.sum(samples[:, 0]) / NUM_OF_USERS_CLASSES, np.sum(self.users_alpha[ : ,prod_id])
                )
        
        return np.array(exp_user_alpha)

    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------
    def round_step3(self, pulled_arm):
        # # We supposed that the competitors invest the maximum of the e-commerce
        # if np.all(pulled_arm == 0):
        #     return np.zeros_like(pulled_arm)


        concentration_parameters = np.array([ self.functions_dict[i](pulled_arm[i]) for i in range(len(pulled_arm))])

        concentration_parameters = np.insert(
            arr=concentration_parameters, obj=0, values=np.max(concentration_parameters)
        )

        # Multiply the concentration parameters by 100 to give more stability
        concentration_parameters = np.multiply(concentration_parameters, self.dirichlet_variance_keeper)
        concentration_parameters[np.where(concentration_parameters == 0)] = 0.001

        samples = self.rng.dirichlet(
            alpha=concentration_parameters, size=NUM_OF_USERS_CLASSES
        ) 

        samples = (
            np.sum(a=samples, axis=0) 
        ) / NUM_OF_USERS_CLASSES  # sum over the columns + renormalization

        return np.minimum(samples[1:], np.sum(self.users_alpha, axis=0)[1:])

    # -----------------------------------------------
    # --------STEP 4 ENVIRONMENT FUNCTIONS-----------

    
    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm):
        row = pulled_arm[0]
        col = pulled_arm[1]
        return np.random.binomial(n=1, p=self.network.get_adjacency_matrix()[row][col])
