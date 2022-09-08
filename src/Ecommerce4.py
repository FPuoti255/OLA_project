import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Ecommerce3 import *

from constants import *
from Utils import *


class Ecommerce4_GPTS(Ecommerce3_GPTS):

    def pull_arm(self):
        a, b = compute_beta_parameters(self.means, self.sigmas)
        samples = np.random.beta(a=a, b=b)
        arm_idxs, _ = self.dynamic_knapsack_solver(table=samples)
        return self.budgets[arm_idxs]


class Ecommerce4_GPUCB(Ecommerce3_GPUCB):

    def pull_arm(self):
        upper_conf = self.means + self.confidence_bounds
        arm_idxs, _ = self.dynamic_knapsack_solver(table=upper_conf)
        return self.budgets[arm_idxs]

