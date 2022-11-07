from Ecommerce4 import *

confidence=0.1

class Algorithm:
    def get_best_bound_arm(self):
        '''
        This function will be used in the step7 during estimation splitting condition
        '''
        value_per_click = self.compute_value_per_click(self.items_estimator.get_estimation())
        estimated_reward = np.multiply(
            self.get_samples(),
            np.atleast_2d(value_per_click).T
        )
        _, mu = self.dynamic_knapsack_solver(table=estimated_reward)
        return mu - np.sqrt( - np.log(confidence) / (2 * self.t))



class GPTS(Ecommerce4_GPTS, Algorithm):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)
    
    def gp_init(self):
        kernel = Matern(
            length_scale=self.gp_config['length_scale'],
            length_scale_bounds=(self.gp_config['length_scale_lb'],self.gp_config['length_scale_ub']),
            nu = 1.5
        )
     
        gaussian_regressors = [
            GaussianProcessRegressor(
                alpha=self.gp_config['gp_alpha'],
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=9
            )
            for _ in range(NUM_OF_PRODUCTS)
        ]

        return gaussian_regressors
    
    def update_model(self):
        super().update_model()
        if self.t < 40:
            self.exploration_probability = 0.1
        else:
            self.exploration_probability = 0.01

    def get_new_instance(self):
        return GPTS(self.B_cap, self.budgets, self.product_prices, self.gp_config)


class GPUCB(Ecommerce4_GPUCB, Algorithm):
    def __init__(self, B_cap: float, budgets, product_prices, gp_config: dict):
        super().__init__(B_cap, budgets, product_prices, gp_config)    

    def get_new_instance(self):
        return GPUCB(self.B_cap, self.budgets, self.product_prices, self.gp_config)
