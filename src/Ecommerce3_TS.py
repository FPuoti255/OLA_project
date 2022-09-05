from Ecommerce3 import *

class Ecommerce3_TS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)
        
        self.sigmas = np.ones(shape = (NUM_OF_PRODUCTS, self.n_arms)) * 2


    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(X = np.atleast_2d(self.budgets).T, return_std=True)
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)


    def update_observations(self, pulled_arm, reward):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][int(np.where(self.budgets == pulled_arm[i])[0])].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])


    def pull_arm(self, nodes_activation_probabilities):
        # https://stats.stackexchange.com/a/316088
        a = np.multiply(self.means, self.sigmas)
        b = np.multiply(self.sigmas, (1 - self.means))
        samples = np.random.beta(a = a, b = b)

        value_per_click = np.dot(nodes_activation_probabilities, self.product_prices)
        reshaped_value_per_click = np.tile(A = np.atleast_2d(value_per_click).T, reps = self.n_arms)
        exp_reward = np.multiply(samples, reshaped_value_per_click)

        arm_idxs, _ = self.dynamic_knapsack_solver(table = exp_reward)

        return self.budgets[arm_idxs] 