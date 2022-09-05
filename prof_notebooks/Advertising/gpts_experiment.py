import numpy as np
import matplotlib.pyplot as plt
from GP_in_MABs.BiddingEnvironment import *
from GP_in_MABs.GTS_Learner import *
from GP_in_MABs.GPTS_Learner import *



n_arms = 20
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 60
n_experiments = 100
gts_rewards_per_experiment = []
gpts_rewards_per_experiment = []


for e in range(0, n_experiments):
    env = BiddingEnvironment(bids=bids, sigma = sigma)
    gts_learner = GTS_Learner(n_arms=n_arms)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)
    for t in range(0, T):
        #Gaussian Thompson Sampling
        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gts_learner.update(pulled_arm, reward)

        #GP Thompson sampling
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)

    gts_rewards_per_experiment.append(gts_learner.collected_rewards)
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt-gts_rewards_per_experiment, axis = 0)), 'r')
plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis = 0)), 'g')
plt.legend(["GTS", "GPTS"])
plt.show()
