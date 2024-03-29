from re import U
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "step7"))

import numpy as np

from Algorithm import *
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
                GPTS(B_cap, budgets, product_prices, gp_config)
            )
        elif algorithm_type == 'UCB':
            self.context_tree = ContextNode(
                self.features,
                GPUCB(B_cap, budgets, product_prices, gp_config)
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
            

        alpha_samples = [ [] for _ in range(NUM_OF_USERS_CLASSES)]
        sold_items_samples = [[] for _ in range(NUM_OF_USERS_CLASSES)]

        for learner in context_learners:
            context_idxs = get_feature_idxs(learner.context_features)
            num_of_features = len(context_idxs)

            # I've divided by the num of features of a given leaner because
            # in the case in which a single learner has all the features together,
            # the alpha and the sold items can be considered as equally divided
            # among all the concerned user classes
            alpha = learner.get_alpha_estimation() / num_of_features
            sold_items = learner.get_sold_items_estimation() / num_of_features

            for idx in context_idxs:
                alpha_samples[idx].append(alpha)
                sold_items_samples[idx].append(sold_items)


        table = np.zeros(shape=(NUM_OF_USERS_CLASSES ,NUM_OF_PRODUCTS, budgets.shape[0]))

        for user_class in range(NUM_OF_USERS_CLASSES):
            user_class_alpha = np.zeros(shape=(NUM_OF_PRODUCTS, budgets.shape[0]))
            user_class_sold_items = np.zeros(shape=(NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))

            if len(alpha_samples[user_class]) > 1:
                user_class_alpha = np.mean(alpha_samples[user_class], axis = 0)
                user_class_sold_items = np.mean(sold_items_samples[user_class], axis = 0)
            else:
                user_class_alpha = alpha_samples[user_class]
                user_class_sold_items = sold_items_samples[user_class]

            user_class_value_per_click = np.sum(
                np.multiply(user_class_sold_items, self.product_prices),
                axis = 1
            )

            table[user_class] = np.multiply(
                user_class_alpha,
                np.atleast_2d(user_class_value_per_click).T
            )
            
        arm, arm_idxs, _ = self.clairvoyant_disaggregated_optimization_problem(table)
        return arm, arm_idxs


    def evaluate_splitting_condition(self):

        context_learners = self.context_tree.get_leaves()

        pulled_arms_array = np.array(self.pulled_arms)
        collected_rewards_array = np.array(self.collected_rewards)
        collected_sold_items_array = np.array(self.collected_sold_items)

        for learner in context_learners:
            print(f'\nEvaluating for learner {learner}')
            if len(learner.context_features) == 1:
                print(f'its is not possible to split further -> just one feature: {learner.context_features}')
                continue

            mu_0 = learner.get_best_bound_arm()
            print(f'm_0 = {mu_0}')

            splits = generate_splits(learner.context_features)


            for left, right in splits:
                print('trying: ', left, right)
                p_left = np.round(len(left) / len(learner.context_features), 2)
                p_right = np.round(len(right) / len(learner.context_features), 2)

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

                split_exp_reward = mu_1 * p_left + mu_2 * p_right
                print(f'{mu_1} * {p_left} + {mu_2} * {p_right} = {split_exp_reward}')
                if split_exp_reward >= mu_0 :
                    learner.left_child = alg1_node
                    learner.right_child = alg2_node
                    print('Better split_found', left, right)
                    break

        

    def update(self, pulled_arm_idxs, reward, num_sold_items):
        assert(reward.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS))
        assert(num_sold_items.shape == (NUM_OF_USERS_CLASSES, NUM_OF_PRODUCTS, NUM_OF_PRODUCTS))
        
        self.t += 1
        
        context_learners = self.context_tree.get_leaves()

        for learner in context_learners:
            
            learner_arm, learner_reward, learner_sold_items = get_context_data(
                learner.context_features,
                pulled_arm_idxs, 
                reward,
                num_sold_items,
                single_sample= True
            )
            learner.update(learner_arm, learner_reward, learner_sold_items)
        
        self.pulled_arms.append(pulled_arm_idxs)
        self.collected_rewards.append(reward)
        self.collected_sold_items.append(num_sold_items)




