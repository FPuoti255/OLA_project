import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "step7"))

import numpy as np


from Ecommerce3 import *
from SoldItemsEstimator import *
from features_utility import *


class ContextNode(object):
    def __init__(self, context_features : dict, algorithm : Ecommerce3) -> None:
        # feature are all the possible configurations, 
        # wheres context_features represent the configuration belonging to the context of the current node
        self.context_features = context_features
        self.context_features_idxs =  get_feature_idxs(self.context_features)       
        
        self.algorithm = algorithm
        self.sold_items_estimator = SoldItemsEstimator()

        self.left_child : ContextNode = None
        self.right_child : ContextNode = None


    def is_leaf(self):
        return self.left_child == None and self.right_child == None

    def get_leaves(self):
        """ Recursive method that returns the leaves of the tree. """
        # base case of the recursion
        if self.is_leaf():
            return [self]
        # otherwise this is not a leaf node, so check the child
        left_leaves = self.left_child.get_leaves()
        right_leaves = self.right_child.get_leaves()
        # concatenation of the children' leaves
        return left_leaves + right_leaves


    def train_offline(self, pulled_arms, collected_rewards, collected_sold_items):
        for t in range(pulled_arms.shape[0]):
            self.update(pulled_arms[t], collected_rewards[t], collected_sold_items[t])


    def get_alpha_estimation(self):
        return self.algorithm.get_samples()

    def get_sold_items_estimation(self):
        return self.sold_items_estimator.get_estimation()

    def get_best_bound_arm(self):
        return self.algorithm.get_best_bound_arm(self.sold_items_estimator.get_estimation())

    def update(self, pulled_arm_idxs, context_reward, context_sold_items):
        assert(pulled_arm_idxs.shape == (NUM_OF_PRODUCTS,))
        assert(context_reward.shape == (NUM_OF_PRODUCTS,))
        assert(context_sold_items.shape == (NUM_OF_PRODUCTS,NUM_OF_PRODUCTS))
        self.algorithm.update(pulled_arm_idxs, context_reward)
        self.sold_items_estimator.update(context_sold_items)    


