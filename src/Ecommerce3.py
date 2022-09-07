import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce import *
from constants import *
from Utils import *


class Ecommerce3(Ecommerce):
    def __init__(self, B_cap: float, budgets, product_prices, tot_num_users):

        super().__init__(
            B_cap=B_cap,
            budgets=budgets,
            product_prices=product_prices,
            tot_num_users=tot_num_users,
        )

        # The budgets are our arms!
        self.n_arms = self.budgets.shape[0]

        self.t = 0
        # I'm generating a distribution of the budgets for each product
        self.means = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 0.5
        self.sigmas = np.ones(shape=(NUM_OF_PRODUCTS, self.n_arms)) * 2

        self.pulled_arms = [[] for i in range(NUM_OF_PRODUCTS)]
        self.rewards_per_arm = [
            [[] for i in range(self.n_arms)] for j in range(NUM_OF_PRODUCTS)
        ]
        self.collected_rewards = [[] for i in range(NUM_OF_PRODUCTS)]

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # we need one gaussian regressor for each product
        self.gaussian_regressors = [
            GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=9
            )
            for i in range(NUM_OF_PRODUCTS)
        ]
    
    @staticmethod
    def run_experiments():
        gpucb_rewards_per_experiment = []
        gpts_rewards_per_experiment = []
        n_experiments = 10
        T = 20

        env = Environment(
            users_reservation_prices,
            click_probabilities,
            observations_probabilities,
            tot_num_users,
        )
        ecomm2 = Ecommerce2(B_cap, budgets, product_prices, tot_num_users)
        optimal_allocation = ecomm2.solve_optimization_problem(env,nodes_activation_probabilities)

        for e in tqdm(range(0, n_experiments), position=0, desc="n_experiment", leave=False):
            ecomm3_gpts = Ecommerce3_TS(B_cap = B_cap, budgets = budgets, product_prices = product_prices, tot_num_users = tot_num_users)
            ecomm3_ucb = Ecommerce3_UCB(B_cap = B_cap, budgets = budgets, product_prices = product_prices, tot_num_users = tot_num_users)
            nodes_activation_probabilities = env.get_nodes_activation_probabilities(product_prices)
            for t in tqdm(range(0, T), position=1, desc="n_iteration", leave=False):
                arm = ecomm3_ucb.pull_arm()
                reward = env.round_step3(arm)
                ecomm3_ucb.update(arm, reward)

                arm = ecomm3_gpts.pull_arm(nodes_activation_probabilities)
                reward = env.round_step3(arm)
                ecomm3_gpts.update(arm, reward)

            gpucb_rewards_per_experiment.append(ecomm3_ucb.collected_rewards)
            gpts_rewards_per_experiment.append(ecomm3_gpts.collected_rewards)


        opt = env.round_step3(pulled_arm=optimal_allocation)
        plt.figure(0)
        plt.ylabel("Regret")
        plt.xlabel("t")

        # this np.mean is used to compute the average regret for each "product" -> output shape = (n_experiments x NUM_OF_PRODUCTS)
        gpts_regret_superarm = opt - np.mean(np.array(gpucb_rewards_per_experiment), axis=2)
        gpucb_regret_superarm = opt - np.mean(np.array(gpts_rewards_per_experiment), axis=2)

        # this np.mean before of the cumsum is to average over all the products
        plt.plot(np.cumsum( np.mean(gpts_regret_superarm, axis=1)), "r")
        plt.plot(np.cumsum(np.mean(gpucb_regret_superarm, axis=1)), "g")
        plt.legend(["GPUCB", "GPTS"])
        plt.show()

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # The methods below will be implemented in the sub-classes
    def update_observations(self, pulled_arm, reward):
        pass

    def update_model(self):
        for i in range(NUM_OF_PRODUCTS):
            x = np.atleast_2d(self.pulled_arms[i]).T
            y = np.array(self.collected_rewards[i])
            self.gaussian_regressors[i].fit(x, y)

            self.means[i], self.sigmas[i] = self.gaussian_regressors[i].predict(
                X=np.atleast_2d(self.budgets).T, return_std=True
            )
            self.sigmas[i] = np.maximum(self.sigmas[i], 1e-2)

    def pull_arm(self, nodes_activation_probabilities):
        pass


class Ecommerce3_TS(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)

    def update_observations(self, pulled_arm, reward):
        for i in range(NUM_OF_PRODUCTS):
            self.rewards_per_arm[i][
                int(np.where(self.budgets == pulled_arm[i])[0])
            ].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

    def pull_arm(self, nodes_activation_probabilities):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)

        value_per_click = (
            np.dot(nodes_activation_probabilities, self.product_prices)
            * self.tot_num_users
        )
        reshaped_value_per_click = np.tile(
            A=np.atleast_2d(value_per_click).T, reps=self.n_arms
        )
        exp_reward = np.multiply(samples, reshaped_value_per_click)

        arm_idxs, _ = self.dynamic_knapsack_solver(table=exp_reward)

        return self.budgets[arm_idxs]


class Ecommerce3_UCB(Ecommerce3):
    def __init__(self, B_cap, budgets, product_prices, tot_num_users):
        super().__init__(B_cap, budgets, product_prices, tot_num_users)

        self.confidence_bounds = np.full(
            shape=(NUM_OF_PRODUCTS, self.n_arms), fill_value=np.inf
        )
        # Number of times the arm has been pulled
        self.N_a = np.zeros(shape=(NUM_OF_PRODUCTS, self.n_arms))

    def update_observations(self, pulled_arm, reward):

        for i in range(NUM_OF_PRODUCTS):
            arm_idx = int(np.where(self.budgets == pulled_arm[i])[0])
            self.N_a[i][arm_idx] += 1
            self.rewards_per_arm[i][arm_idx].append(reward[i])
            self.pulled_arms[i].append(pulled_arm[i])
            self.collected_rewards[i].append(reward[i])

        self.confidence_bounds = np.sqrt(2 * np.log(self.t) / self.N_a)
        self.confidence_bounds[self.N_a == 0] = np.inf

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
        # arm_idxs = np.argmax(upper_conf, axis=1)
        return self.budgets[arm_idxs]

    def dynamic_knapsack_solver(self, table):
        """
        In the UCB we do not need to subtract the budgets to the final row,
        since we use the dynamic algorithm for another purpose
        """
        table_opt, max_pointer = self.compute_table(table)
        return self.choose_best(table_opt, max_pointer)
