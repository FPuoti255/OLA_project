import numpy as np
from itertools import combinations, chain

from Ecommerce4 import *
from step7 import Algorithms as alg
from constants import *



class ContextNode(object):
    def __init__(self, context_features : dict, algorithm) -> None:
        # feature are all the possible configurations, 
        # wheres context_features represent the configuration belonging to the context of the current node
        self.context_features = context_features        
        self.algorithm = algorithm # algorithm can be TS or UCB

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

    def generate_splits(self):
        subsets = [v for a in range(len(self.context_features)) for v in combinations(self.context_features, a)]
        permutations = []
        for i in range(1, int(len(subsets)/2 + 1)):
            permutations.append( (list(chain(subsets[i])), [e for e in self.context_features if e not in subsets[i]]) )
        return permutations

    def generate_feature_idxs(self, features : dict):
        idxs = []
        for _, idx in features.items():
            idxs.append(idx)
        return idxs

    def get_context_data(self, features : dict, pulled_arms, collected_rewards, collected_sold_items):

        idxs = self.generate_feature_idxs(features)
        return pulled_arms[:, idxs[0], :], np.sum(collected_rewards[: , idxs, :], axis=1), \
            np.sum(collected_sold_items[: , idxs, :], axis=1)



    def evaluate_splitting_condition(self, features : list,
                                        pulled_arms, collected_rewards, collected_sold_items):

        assert(collected_rewards.shape[1 : ] == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(pulled_arms.shape[0] == collected_rewards.shape[0] == collected_sold_items.shape[0])

        if not self.is_leaf():
            raise ValueError('the node has already been splitted')

        if len(self.context_features) == 1:
            print('its is not possible to split further. The number of feature for this context is 1')
            return
        
        splits = self.generate_splits()
        mu_0 = self.algorithm.get_best_bound_arm()        

        for left, right in splits:
            p_left = len(left) / len(features)
            p_right = len(right) / len(features)

            alg1 = self.algorithm.get_new_instance()
            alg2 = self.algorithm.get_new_instance()

            c1 = dict.fromkeys(left)
            c2 = dict.fromkeys(right)

            for k, _ in c1.items():
                c1[k] = self.context_features[k]
            
            for k, _ in c2.items():
                c2[k] = self.context_features[k]

            arms, rew, sold_it  = self.get_context_data(c1, pulled_arms, collected_rewards, collected_sold_items)
            alg1.train_offline(arms, rew, sold_it)

            arms, rew, sold_it = self.get_context_data(c2, pulled_arms, collected_rewards, collected_sold_items)
            alg2.train_offline(arms, rew, sold_it)
  
            mu_1 = alg1.get_best_bound_arm()
            mu_2 = alg2.get_best_bound_arm()

            if mu_1 * p_left + mu_2 * p_right >= mu_0:

                self.left_child = ContextNode(c1, alg1)
                self.right_child = ContextNode(c2, alg2)
                print('Better split_found', c1, c2)
                return

        print('No better split found!')

    def pull_arm(self):
        idxs = self.generate_feature_idxs(self.context_features)        
        return self.algorithm.pull_arm(), list(set(idxs)) #we return also the index of the user classes of this context

    def update(self, pulled_arm, context_reward, context_sold_items):
        assert(pulled_arm.shape == (NUM_OF_PRODUCTS,))
        assert(context_reward.shape == (NUM_OF_PRODUCTS,))
        assert(context_sold_items.shape == (NUM_OF_PRODUCTS,))
        self.algorithm.update(pulled_arm, context_reward, context_sold_items)
        
            