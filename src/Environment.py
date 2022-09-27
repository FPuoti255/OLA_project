import numpy as np
from itertools import combinations_with_replacement, permutations

from constants import *
from Utils import *
from Network import Network


class Environment:
    def __init__(
        self,
        users_reservation_prices,
        graph_weights,
        alpha_bars
    ):

        self.rng = np.random.default_rng(12345)

        self.users_reservation_prices = users_reservation_prices
        self.alpha_bars = alpha_bars

        self.expected_users_alpha = None
        self.expected_reward = None

        self.network = Network(adjacency_matrix=graph_weights)


    def get_users_reservation_prices(self):
        return self.users_reservation_prices

    def get_alpha_bars(self):
        return self.alpha_bars

    def get_network(self):
        return self.network

    def mapping_function(self, prod_id, budget):
        '''
        @returns a map for each user class. shape = (NUM_OF_USER_CLASSES, 1)
        '''
        return 2 * self.alpha_bars[:, prod_id + 1] / (1 + 1/budget)

    def plot_mapping_functions(self, budgets):
        for i in range(NUM_OF_PRODUCTS):
            plt.plot(budgets, [self.mapping_function(i, bu) for bu in budgets])



    def compute_users_alpha(self, budgets: np.ndarray):
        '''
        :budgets: must be passed normalized ( between 0 and 1), thus budgets / B_cap
        :return: the expected alpha for each couple (prod_id, budget_allocated)
        '''
        budgets = budgets / budgets[-1]
        exp_user_alpha = np.zeros(shape=(NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, budgets.shape[0]))

        for prod_id in range(NUM_OF_PRODUCTS):
            for j in range(1, budgets.shape[0]):
                # maps (budget, prod_id) -> concentration_parameters to give to the dirichlet
                conc_params = self.mapping_function(prod_id, budgets[j])

                for user_class in range(NUM_OF_USERS_CLASSES):
                    exp_user_alpha[user_class, prod_id, j] = min(
                        self.rng.dirichlet(
                            np.multiply([conc_params[user_class], 1 - conc_params[user_class]], 1000)
                            )[0],
                        self.alpha_bars[user_class, prod_id + 1]
                    )

        self.expected_users_alpha = exp_user_alpha   


    def compute_clairvoyant_reward(self, num_sold_items, product_prices, budgets):
        '''
        This function computes the expected reward = expected_users_alpha x value_per_click
        for each couple (product, budget_allocated)
        '''
        
        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        assert(product_prices.shape == (NUM_OF_PRODUCTS,))

        value_per_click = np.sum(np.multiply(num_sold_items, product_prices), axis = 2) # shape = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)
        
        self.compute_users_alpha(budgets) # (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_BUDGETS)

        # expected_reward = np.zeros_like(self.expected_users_alpha)

        # for user_class in range(NUM_OF_USERS_CLASSES):
        #     for prod in range(NUM_OF_PRODUCTS):
        #         expected_reward[user_class][prod] = \
        #             self.expected_users_alpha[user_class][prod] * value_per_click[user_class][prod]

        # self.expected_reward = expected_reward 

        # if aggregated:
        #     return np.sum(expected_reward, axis = 0)
        # else:
        #     return expected_reward

        aggregated_value_per_click = np.sum(value_per_click, axis=0)
        aggregated_users_alpha = np.sum(self.expected_users_alpha, axis = 0)

        exp_reward = np.multiply(aggregated_users_alpha, np.atleast_2d(aggregated_value_per_click).T)

        self.expected_reward = exp_reward
        return exp_reward


    # -----------------------------------------------
    # --------STEP 3 ENVIRONMENT FUNCTIONS-----------
    def round_step3(self, pulled_arm, pulled_arm_idxs):

        assert (pulled_arm_idxs.shape == (NUM_OF_PRODUCTS,))
        assert(self.expected_reward is not None)
        assert(self.expected_users_alpha is not None)

        # pulled_arm = ecommerce.budgets[pulled_arm_idxs]

        aggregated_exp_users_alpha = np.sum(self.expected_users_alpha, axis = 0)

        alpha = np.zeros_like(pulled_arm_idxs)
        reward = 0

        for i in range(NUM_OF_PRODUCTS):
            alpha[i] = aggregated_exp_users_alpha[i][pulled_arm_idxs[i]]
            reward += self.expected_reward[i][pulled_arm_idxs[i]] 

        return alpha, reward - np.sum(pulled_arm)


    # -----------------------------------------------
    # --------STEP 4 ENVIRONMENT FUNCTIONS-----------

    def round_step4(self, pulled_arm, B_cap, num_sold_items):

        assert (num_sold_items.shape == (
            NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        
        tot_alpha_per_product = self.round_step3(pulled_arm, B_cap)

        tot_sold_per_product = np.sum(num_sold_items, axis=0)

        estimated_sold_items = np.divide(tot_alpha_per_product, np.sum(self.alpha_bars, axis = 0)[1:])* tot_sold_per_product
            
        return tot_alpha_per_product, estimated_sold_items


    # -----------------------------------------------
    # --------STEP 5 ENVIRONMENT FUNCTIONS-----------
    def round_step5(self, pulled_arm, nodes_activation_probabilities):
        row = pulled_arm[0]
        col = pulled_arm[1]
        return np.random.binomial(n = 1, p = nodes_activation_probabilities[row][col])

    # -----------------------------------------------
    # --------STEP 7 ENVIRONMENT FUNCTIONS----------- 
    def estimate_disaggregated_num_clicks(self, budgets):
        return self.estimate_num_of_clicks(budgets, aggregated=False)
    
    def round_step7(self, pulled_arm, B_cap, nodes_activation_probabilities, num_sold_items):
        assert (pulled_arm.shape == (NUM_OF_USERS_CLASSES,NUM_OF_PRODUCTS))

        alpha = self.compute_alpha(pulled_arm / B_cap)
        assert (alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))

        estimated_sold_items = np.multiply(
            np.dot(num_sold_items, nodes_activation_probabilities.T),
            alpha
        )

        return alpha, estimated_sold_items



