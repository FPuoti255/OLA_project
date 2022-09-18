from step7 import ContextNode
from step7 import Algorithms
import numpy as np
from Ecommerce import *

class Ecommerce7(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, features:dict, algorithm_type = 'TS'):
        super().__init__(B_cap, budgets, product_prices)
        self.features = features


        if algorithm_type == 'TS' :
            algorithm = Algorithms.TS(B_cap, budgets, product_prices)
        elif algorithm_type == 'UCB' :
            algorithm = Algorithms.UCB(B_cap, budgets, product_prices)
        else :
            raise ValueError ('Algorithm type can be TS or UCB')

        self.contexts_tree = ContextNode.ContextNode(context_features=features.copy(), algorithm = algorithm)
        
        self.pulled_arms = []
        self.collected_rewards = []
        self.collected_sold_items = []
    
    def update_history(self, pulled_arms, collected_rewards, collected_sold_items):
        self.pulled_arms.append(pulled_arms)
        self.collected_rewards.append(collected_rewards)
        self.collected_sold_items.append(collected_sold_items)


    def get_pulled_arms(self):
        return np.array(self.pulled_arms)
    
    def get_collected_rewards(self):
        return np.array(self.collected_rewards)
    
    def get_collected_sold_items(self):
        return np.array(self.collected_sold_items)
    
    
    def get_context_tree(self):
        return self.contexts_tree
    
    def clairvoyant_solve_optimization_problem(self, num_sold_items, exp_num_clicks, nodes_activation_probabilities):
        '''
        Disaggregated version of the optimization problem
        '''

        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(exp_num_clicks.shape == (NUM_OF_USERS_CLASSES,NUM_OF_PRODUCTS, self.budgets.shape[0]))
        assert(nodes_activation_probabilities.shape == (NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        tot_margin_per_product = np.multiply(num_sold_items, self.product_prices) # NUM_OF_USER_CLASSES x NUM_OF_PRODUCTS

        value_per_click = np.dot(tot_margin_per_product, nodes_activation_probabilities.T) # NUM_OF_USER_CLASSES x NUM_OF_PRODUCT


        value_per_click = np.repeat(value_per_click, exp_num_clicks.shape[-1], axis = -1).reshape(exp_num_clicks.shape)

        exp_reward = np.multiply(exp_num_clicks, value_per_click).reshape((NUM_OF_USERS_CLASSES*NUM_OF_PRODUCTS, self.budgets.shape[0]))

        budgets_indexes, reward = self.dynamic_knapsack_solver(
            table=exp_reward
        )
        
        optimal_allocation = self.budgets[budgets_indexes]

        return optimal_allocation.reshape((NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)), reward