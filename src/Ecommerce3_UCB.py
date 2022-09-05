from Ecommerce3 import *

class Ecommerce3_UCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)

        self.confidence_bounds = np.full(shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value = np.inf)
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))


    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i] = self.gaussian_regressors[i].predict(X = np.atleast_2d(self.budgets).T)
        
            
    def update_observations(self, pulled_arm, reward):
        
        self.confidence_bounds = np.sqrt( 2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a==0] = np.inf

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.N_a[i][arm_idx] += 1
            self.rewards_per_arm[i][arm_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])
        

    def pull_arm(self, nodes_activation_probabilities):
        upper_conf = self.means + self.confidence_bounds        
        arm_idxs, _ = self.dynamic_knapsack_solver(table = upper_conf)
        return self.budgets[arm_idxs]


    def dynamic_knapsack_solver(self, table):
        '''
        In the UCB we do not need to subtract the budgets 
        since we use the dynamic algorithm for another purpose
        '''
        table_opt, max_pointer = self.compute_table(table)
        return self.choose_best(table_opt, max_pointer)