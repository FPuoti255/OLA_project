import numpy as np

from constants import *
from Utils import *
from Network import Network
from Environment import *


class Non_Stationary_Environment(Environment):
    def __init__(self, users_reservation_prices, product_functions_idxs, click_probabilities, users_alpha, num_sold_items, nodes_activation_probabilities, users_poisson_parameters, horizon):

        super().__init__(users_reservation_prices, click_probabilities, users_alpha)
        self.user_poisson_parameters = users_poisson_parameters
        
        self.t = 0
        n_phases = len(users_alpha)
        self.phase_size = horizon / n_phases

        # Instead of defining a lot of different functions_dicts, we use always the same
        # but at each phase the products will exchange among themselves the functions
        self.product_functions_idxs = product_functions_idxs
        self.num_sold_items = num_sold_items
        self.nodes_activation_probabilities = nodes_activation_probabilities

        self.current_phase = 0


    def get_users_alpha(self):
        return self.users_alpha[self.current_phase]

    def get_num_sold_items(self):
        return self.num_sold_items[self.current_phase]

    def get_nodes_activation_probabilities(self):
        return self.nodes_activation_probabilities[self.current_phase]

    def get_users_poisson_parameters(self):
        return self.user_poisson_parameters[self.current_phase]

    def estimate_num_of_clicks(self, budgets: np.ndarray):
        '''
        :budgets: must be passed normalized ( between 0 and 1), thus budgets / B_cap is expected
        :return: the expected alpha for each couple (prod_id, budget_allocated)
        '''
        exp_user_alpha = np.zeros(shape=(NUM_OF_PRODUCTS, budgets.shape[0]))

        prd_function_idx = self.product_functions_idxs[self.current_phase]

        for prod_id in range(NUM_OF_PRODUCTS):
            for j in range(budgets.shape[0]):
                
                conc_param = self.functions_dict[prd_function_idx[prod_id]](budgets[j])

                samples = self.rng.dirichlet(
                    alpha=np.multiply([conc_param, 1 - conc_param], self.dirichlet_variance_keeper), size=NUM_OF_USERS_CLASSES
                ) / NUM_OF_USERS_CLASSES

                prod_samples = np.minimum(samples[:, 0], self.users_alpha[self.current_phase][:, (prod_id +1)])

                assert(prod_samples.shape == (NUM_OF_USERS_CLASSES,))
                exp_user_alpha[prod_id][j] = np.sum(prod_samples)
        
        exp_user_alpha[:, 0] = 0 # set to zero the expected alpha when the budget allocated is zero

        return exp_user_alpha
    

    def compute_alpha(self, allocation):
        '''
        This function is the same as the estimate_num_of_clicks but instead of computing the alpha for each couple (product, budget)
        it computes tha alpha just for the budgets of the allocation

        @returns:
            - exp_user_alpha -> shape= (NUM_OF_USERS_CLASSES, allocation.shape[0]) 3x5
        '''

        # if the allocation is composed all of zero, return zero !
        if not np.any(allocation):
            return np.zeros(shape=(NUM_OF_USERS_CLASSES, allocation.shape[0]))

        prd_function_idx = self.product_functions_idxs[self.current_phase]


        exp_user_alpha = np.zeros(shape= (NUM_OF_USERS_CLASSES, allocation.shape[0]))

        for prod_id in range(NUM_OF_PRODUCTS):
            # maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
            conc_param = self.functions_dict[prd_function_idx[prod_id]](allocation[prod_id])
            
            # we multiplied by dirichlet_variance_keeper to reduce the variance in the estimation
            samples = self.rng.dirichlet(
                alpha=np.multiply([conc_param, 1 - conc_param], self.dirichlet_variance_keeper), size=NUM_OF_USERS_CLASSES
            ) / NUM_OF_USERS_CLASSES

            prod_samples = np.minimum(samples[:, 0], self.users_alpha[self.current_phase][:, (prod_id +1)])

            # min because for each campaign we expect a maximum alpha, which is alpha_bar
            exp_user_alpha[:, prod_id] = prod_samples

        return exp_user_alpha


    def round_step6(self, pulled_arm, B_cap, end_phase = False):

        self.current_phase = int(np.floor(self.t / self.phase_size))

        assert (pulled_arm.shape[0] == NUM_OF_PRODUCTS)

        alpha = self.compute_alpha(pulled_arm / B_cap)
        assert (alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(np.greater_equal(self.users_alpha[self.current_phase][:, 1:], alpha).all())

        tot_alpha_per_product = np.sum(alpha, axis=0)         

        tot_sold_per_product = np.sum(self.num_sold_items[self.current_phase], axis=0)

        estimated_sold_items = np.sum(
            np.multiply(self.nodes_activation_probabilities[self.current_phase].T, tot_alpha_per_product).T,
            axis = 0) * tot_sold_per_product
            
        if end_phase : #in this way also the second learner to pull will have the same phase parameters
            self.t += 1

        return tot_alpha_per_product, estimated_sold_items
