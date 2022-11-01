import sys, os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "step7"))


import numpy as np
from itertools import combinations, chain


from constants import *


def get_feature_idxs(features_dict):
    idxs = []
    for _, idx in features_dict.items():
        idxs.append(idx)
    return list(set(idxs))


def get_context_data(context_dict, pulled_arms, collected_rewards, collected_sold_items):
    context_dict_idxs = get_feature_idxs(context_dict)
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


def generate_splits(context_features : dict):
    subsets = [v for a in range(len(context_features)) for v in combinations(context_features, a)]
    permutations = []
    for i in range(1, int(len(subsets)/2 + 1)):
        permutations.append( (list(chain(subsets[i])), [e for e in context_features if e not in subsets[i]]) )

    # sorting putting in the first place the most balanced splits (where the len of the splits is equal)
    permutations = sorted(permutations, 
                            key = lambda el :  abs(len(el[0]) - len(el[1])))
    return permutations