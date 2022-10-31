import numpy as np
from itertools import combinations, chain

from Ecommerce3 import *
from SoldItemsEstimator import *
from constants import *
from Utils import *






class ContextNode(object):
    def __init__(self, context_features : dict, algorithm : Ecommerce3) -> None:
        # feature are all the possible configurations, 
        # wheres context_features represent the configuration belonging to the context of the current node
        self.context_features = context_features        
        
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

    def get_feature_idxs(self, features_dict):
        idxs = []
        for _, idx in features_dict.items():
            idxs.append(idx)
        return list(set(idxs))

    def generate_splits(self):
        '''
        returns all the possible splits of the context_features dictionary
        '''
        feature_idxs = self.get_feature_idxs(self.context_features)
        possible_splits = []

        for size in range(1, len(feature_idxs)):
            combs = combinations(feature_idxs, size)
            for comb in combs:
                possible_splits.append([list(comb), list(set(feature_idxs) -  set(comb))])
        
        splits_dictionaries = []
        for split in possible_splits:
            left = {}
            right = {}

            for key, value in self.context_features.items():
                if value in split[0]:
                    left.update({key : value})
                if value in split[1]:
                    right.update({key : value})
            splits_dictionaries.append((left, right))
        
        
        return splits_dictionaries

    def get_context_data(self, context_dict, pulled_arms, collected_rewards, collected_sold_items):
        context_dict_idxs = self.get_feature_idxs(context_dict)
        num_of_rounds = pulled_arms.shape[0]

        context_dict_pulled_arms = pulled_arms[:, context_dict_idxs, : ]
        context_dict_collected_rewards = collected_rewards[:, context_dict_idxs, :]
        context_dict_collected_sold_items = collected_sold_items[:, context_dict_idxs, :]

        if(len(context_dict_idxs)>1):
            return np.minimum(
                            np.sum(context_dict_pulled_arms, axis = 1),
                            budgets.shape[0]
                    ),\
                    np.sum(context_dict_collected_rewards, axis = 1),\
                    np.sum(context_dict_collected_sold_items, axis = 1)
        else:
            return context_dict_pulled_arms.reshape(num_of_rounds, NUM_OF_PRODUCTS), \
                        context_dict_collected_rewards.reshape(num_of_rounds, NUM_OF_PRODUCTS),\
                        context_dict_collected_sold_items.reshape(num_of_rounds, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS)

    def train_offline(self, pulled_arms, collected_rewards, collected_sold_items):
        for t in range(pulled_arms.shape[0]):
            self.update(pulled_arms[t], collected_rewards[t], collected_sold_items[t])

    def evaluate_splitting_condition(self, features, pulled_arms, collected_rewards, collected_sold_items):

        if not self.is_leaf():
            raise ValueError('the node has already been splitted')

        if len(self.get_feature_idxs(self.context_features)) == 1:
            print('its is not possible to split further. The number of feature for this context is 1')
            return

        # The first value of the shape will be the number of rounds
        # after which the split is evaluated
        assert(pulled_arms.shape[1:] == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(collected_rewards.shape[1:] == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(collected_sold_items.shape[1:] == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        
        mu_0 = self.algorithm.get_best_bound_arm(self.sold_items_estimator.get_estimation())

        splits = self.generate_splits()   

        for c1, c2 in splits:
            p_left = len(c1) / len(features)
            p_right = len(c2) / len(features)

            arms, rewards, sold_items = self.get_context_data(c1, pulled_arms, collected_rewards, collected_sold_items)
            alg1_node = ContextNode(c1, self.algorithm.get_new_instance())
            alg1_node.train_offline(arms, rewards, sold_items)

            arms, rewards, sold_items = self.get_context_data(c2, pulled_arms, collected_rewards, collected_sold_items)
            alg2_node = ContextNode(c2, self.algorithm.get_new_instance())
            alg2_node.train_offline(arms, rewards, sold_items)

  
            mu_1 = alg1_node.algorithm.get_best_bound_arm(alg1_node.sold_items_estimator.get_estimation())
            mu_2 = alg2_node.algorithm.get_best_bound_arm(alg2_node.sold_items_estimator.get_estimation())

            if mu_1 * p_left + mu_2 * p_right >= mu_0:
                self.left_child = alg1_node
                self.right_child = alg2_node
                print('Better split_found', c1, c2)
                return

        print('No better split found!')


    def get_alpha_estimation(self):
        return self.algorithm.get_samples()

    def get_sold_items_estimation(self):
        return self.sold_items_estimator.get_estimation()

    def pull_arm(self):
        idxs = self.get_feature_idxs(self.context_features)
        arm, arm_idxs = self.algorithm.pull_arm(self.sold_items_estimator.get_estimation())        
        return arm, arm_idxs, list(set(idxs)) #we return also the index of the user classes of this context

    def update(self, pulled_arm_idxs, context_reward, context_sold_items):
        assert(pulled_arm_idxs.shape == (NUM_OF_PRODUCTS,))
        assert(context_reward.shape == (NUM_OF_PRODUCTS,))
        assert(context_sold_items.shape == (NUM_OF_PRODUCTS,NUM_OF_PRODUCTS))
        self.algorithm.update(pulled_arm_idxs, context_reward)
        self.sold_items_estimator.update(context_sold_items)    




class Ecommerce7(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, algorithm_type :str,
                    gp_config : dict, features : dict, split_time : int):
        
        super().__init__(B_cap, budgets, product_prices)

        self.t = 0
        self.split_time = split_time

        self.features = features

        if algorithm_type == 'TS':
            self.context_tree = ContextNode(
                self.features,
                Ecommerce3_GPTS(B_cap, budgets, product_prices, gp_config)
            )
        elif algorithm_type == 'UCB':
            self.context_tree = ContextNode(
                self.features,
                Ecommerce3_GPUCB(B_cap, budgets, product_prices, gp_config)
            )
        else:
            raise ValueError('Please choose one between TS or UCB')

        # This array will be of shape (self.t, NUM_OF_PRODUCTS)
        self.pulled_arms = []

        # These arrays will be of shape (self.t, NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS) --> disaggregated_rewards
        self.collected_rewards = []
        self.collected_sold_items = []


   
    def pull_arm(self):
        context_learners = self.context_tree.get_leaves()

        if self.t % self.split_time == 0 and self.t != 0:
            for learner in context_learners:
                learner.evaluate_splitting_condition(
                    self.features, 
                    np.array(self.pulled_arms), 
                    np.array(self.collected_rewards),
                    np.array(self.collected_sold_items)
                )
            context_learners = self.context_tree.get_leaves()

        estimated_alpha = [ [] for _ in range(NUM_OF_USERS_CLASSES)]
        estimated_sold_items = [[] for _ in range(NUM_OF_USERS_CLASSES)]

        for learner in context_learners:
            context_idxs = learner.get_feature_idxs(learner.context_features)
            alpha = learner.get_alpha_estimation()
            num_sold_items = learner.get_sold_items_estimation()

            for idx in context_idxs:
                estimated_alpha[idx].append(alpha)
                estimated_sold_items[idx].append(num_sold_items)


        table = np.zeros(shape=(NUM_OF_USERS_CLASSES ,NUM_OF_PRODUCTS, budgets.shape[0]))

        for user_class in range(NUM_OF_USERS_CLASSES):

            # we compute the mean because it may happen that for a particular
            # split of the feature, more then one algorithm will analyze the
            # same feature set
            if len(estimated_alpha[user_class]) > 1:
                estimated_alpha[user_class] = np.mean(estimated_alpha[user_class], axis = 0)
                estimated_sold_items[user_class] = np.mean(estimated_sold_items[user_class], axis = 0)

        
            value_per_click = np.sum(np.multiply(estimated_sold_items, self.product_prices),
                                    axis=2)  # shape = (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS)


            table[user_class] = np.multiply(
                estimated_alpha[user_class],
                np.atleast_2d(value_per_click[user_class]).T
            )
            
        arm, arm_idxs, _ = self.clairvoyant_disaggregated_optimization_problem(table)
        return arm, arm_idxs


    def update(self, pulled_arm_idxs, reward, num_sold_items):
        assert(reward.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        
        self.t += 1
        
        context_learners = self.context_tree.get_leaves()

        for learner in context_learners:
            context_idxs = learner.get_feature_idxs(learner.context_features)

            # we need to divide the case in which a single learner is using
            if(len(context_idxs) > 1):
                learner.update(
                        np.minimum(
                            np.sum(pulled_arm_idxs[context_idxs, :], axis = 0),
                            self.budgets.shape[0]
                        ),
                        np.sum(reward[context_idxs, :], axis = 0),
                        np.sum(num_sold_items[context_idxs, :], axis = 0)
                        )
            else:
                learner.update(pulled_arm_idxs,
                            reward[context_idxs[0], :],
                            num_sold_items[context_idxs[0], :]
                        )
        
        self.pulled_arms.append(pulled_arm_idxs)
        self.collected_rewards.append(reward)
        self.collected_sold_items.append(num_sold_items)




