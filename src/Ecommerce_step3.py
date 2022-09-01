import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import Ecommerce


value_per_click = np.dot(nodes_activation_probabilities, env.product_prices.T)

# TODO implement this algorithm using the one on the slides 
# For now I've used a dummy algorithm with very low performance
# -------------------------------------------------------------
# generating all the possible combination with replacement of 5 (campaigns) 
# over the 8 possible budgets
combinations = np.array([comb for comb in combinations_with_replacement(self.budgets, 5) if np.sum(comb) <= self.B_cap], dtype=float)

# the combinations do not have any order, thus using the permutation we consider
# all the possible assignment of those budgets to a given campaign
perms = []
for comb in combinations:
    [perms.append(perm) for perm in permutations(comb)]
perms = np.array(list(set(perms))) #set() to remove duplicates

best_allocation = []
max_expected_reward = 0

for allocation in perms:
    #the dirichlet does not accept values <= 0
    allocation[np.where(allocation == 0)] = 1.e-10
    
    # in order to get also the alpha_0 for the users landing on a webpage of a competitor,
    # we set the 'fictitious budget' of the competitor as the average of our allocations
    alphas = env.get_users_alphas(list(np.insert(allocation, obj=0, values = np.average(allocation))))
    
    # the notation inside alphas is to exclude the first column which represent alpha_0
    # but for alpha_0 our reward is 0 
    exp_rew = np.sum(np.dot(alphas[:, 1 :], value_per_click), axis=0)
    if exp_rew > max_expected_reward:
        max_expected_reward = exp_rew
        best_allocation = allocation


class Ecommerce_step3(Ecommerce):
    def __init__(self, B_cap : float, budgets, tot_num_users):
        
        super().__init__(B_cap=B_cap, budgets = budgets, tot_num_users = tot_num_users)
        
        # The budgets are our arms!
        self.n_arms = budgets.shape[0]

        self.t = 0
        self.rewards_per_arm = x =[[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])

        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms)*10
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha, 
                                    normalize_y = True, n_restarts_optimizer = 9)


    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(self.budgets[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)