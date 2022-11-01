import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "step7"))

import numpy as np

from Ecommerce3 import *
from SoldItemsEstimator import *
from ContextNode import *
from features_utility import *

from constants import *
from Utils import *



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

        if self.t % self.split_time == 0 and self.t != 0:
            self.evaluate_splitting_condition()
        
        context_learners = self.context_tree.get_leaves()
            

        estimated_alpha = [ [] for _ in range(NUM_OF_USERS_CLASSES)]
        estimated_sold_items = [[] for _ in range(NUM_OF_USERS_CLASSES)]

        for learner in context_learners:
            context_idxs = get_feature_idxs(learner.context_features)
            num_of_features = len(context_idxs)

            alpha = learner.get_alpha_estimation() / num_of_features
            num_sold_items = learner.get_sold_items_estimation() / num_of_features

            for idx in context_idxs:
                estimated_alpha[idx].append(alpha)
                estimated_sold_items[idx].append(num_sold_items)


        table = np.zeros(shape=(NUM_OF_USERS_CLASSES ,NUM_OF_PRODUCTS, budgets.shape[0]))

        # we compute the mean because it may happen that for a particular
        # split of the feature, more then one algorithm will analyze the
        # same feature set
        estimated_alpha = np.mean(estimated_alpha, axis = 1)
        estimated_sold_items = np.mean(estimated_sold_items, axis = 1)

        estimated_sold_items = np.array(estimated_sold_items)
        estimated_alpha = np.array(estimated_alpha)

        assert( estimated_alpha.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, self.budgets.shape[0]) )
        assert(estimated_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

        value_per_click = np.sum(np.multiply(estimated_sold_items, self.product_prices),
                                 axis=2)

        for user_class in range(NUM_OF_USERS_CLASSES):
            table[user_class] = np.multiply(
                estimated_alpha[user_class],
                np.atleast_2d(value_per_click[user_class]).T
            )
            
        arm, arm_idxs, _ = self.clairvoyant_disaggregated_optimization_problem(table)
        return arm, arm_idxs


    def evaluate_splitting_condition(self):

        context_learners = self.context_tree.get_leaves()

        pulled_arms_array = np.array(self.pulled_arms)
        collected_rewards_array = np.array(self.collected_rewards)
        collected_sold_items_array = np.array(self.collected_sold_items)
        
        for learner in context_learners:
            if len(learner.context_features) == 1:
                print('its is not possible to split further. The number of feature for this context is 1')
                continue

            mu_0 = learner.get_best_bound_arm()

            splits = generate_splits(learner.context_features)


            for left, right in splits:
                p_left = len(left) / len(learner.context_features)
                p_right = len(right) / len(learner.context_features)

                c1 = dict.fromkeys(left)
                c2 = dict.fromkeys(right)

                for k, _ in c1.items():
                    c1[k] = self.features[k]                
                for k, _ in c2.items():
                    c2[k] = self.features[k]
            

                c1_arms, c1_rewards, c1_sold_items = get_context_data(c1, pulled_arms_array, collected_rewards_array, collected_sold_items_array)
                c2_arms, c2_rewards, c2_sold_items = get_context_data(c2, pulled_arms_array, collected_rewards_array, collected_sold_items_array)

                alg1_node = ContextNode(c1, learner.algorithm.get_new_instance())
                alg2_node = ContextNode(c2, learner.algorithm.get_new_instance())


                alg1_node.train_offline(c1_arms, c1_rewards, c1_sold_items)
                alg2_node.train_offline(c2_arms, c2_rewards, c2_sold_items)

                mu_1 = alg1_node.get_best_bound_arm()
                mu_2 = alg2_node.get_best_bound_arm()

                if mu_1 * p_left + mu_2 * p_right >= mu_0 :
                    learner.left_child = alg1_node
                    learner.right_child = alg2_node
                    print('Better split_found', c1, c2)
                    break

        

    def update(self, pulled_arm_idxs, reward, num_sold_items):
        assert(reward.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        
        self.t += 1
        
        context_learners = self.context_tree.get_leaves()

        for learner in context_learners:
            context_idxs = get_feature_idxs(learner.context_features)

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
                learner.update(
                            pulled_arm_idxs[context_idxs[0], :],
                            reward[context_idxs[0], :],
                            num_sold_items[context_idxs[0], :]
                        )
        
        self.pulled_arms.append(pulled_arm_idxs)
        self.collected_rewards.append(reward)
        self.collected_sold_items.append(num_sold_items)




