import numpy as np
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete
from irlc import train
from tqdm import tqdm
import sys
from irlc import cache_read, cache_write, cache_exists

class BanditEnvironment(Env): #!s=a
    def __init__(self, k):
        super().__init__()
        self.observation_space = Discrete(0)  # Empty observation space (no observations)
        self.action_space = Discrete(k)       # The arms labelled 0,1,...,k-1.
        self.k = k # Number of arms

    def reset(self):
        """ You should extend this method. Use it to reset all internal parameters of the environment; i.e. shuffle best arm etc. """
        pass

    def bandit_step(self, a):
        """
        You should extend this method. It should return the reward and average_regret.
        The 'reward' is the reward from pulling arm a, and 'average_regret' is the regret. Once done, the
        'step'-function will be automatically defined (see below) """
        reward = 0 # Compute the reward associated with arm a
        regret = 0 # Compute the regret, by comparing to the optimal arms reward.
        return reward, regret

    def step(self, action):
        """ We also return the average regret. Average regret = 0 means the optimal arm was chosen.
        We return it as a dict because this is the recommended way to pass extra information from the
        environment in openai gym. The train(env,agent,...) method allows us to gather/use the information again. """
        reward, average_regret = self.bandit_step(action)
        info = {'average_regret': average_regret}
        return None, reward, False, info #!s

class StationaryBandit(BanditEnvironment):
    """
    Implement the 'stationary bandit environment' which is described in \cite[Section 2.3]{sutton}
    and used as a running example throughout the chapter.

    We will implement a version with a constant mean offset (q_star_mean), so that

     q* = x + q_star_mean,   x ~ Normal(0,1)

    q_star_mean can just be considered to be zero at first.
    """
    def __init__(self, k, q_star_mean=0):
        super().__init__(k)
        self.q_star_mean = q_star_mean
        self.reset()

    def reset(self):
        """ Set q^*_k = N(0,1) + mean_value. The mean_value is 0 in most examples. I.e., implement the 10-armed testbed environment. """
        self.q_star = np.random.randn(self.k) + self.q_star_mean
        self.optimal_action = np.argmax(self.q_star) # Optimal action is the one with the largest q^*-value.

    def bandit_step(self, a):
        """ Return the reward/regret for action a for the simple bandit. Use self.q_star (see reset-function above).
         To implement it, implement the reward (see the description of the 10-armed testbed for more information.
         How is it computed from from q^*_k?) and also compute the regret.

         As a small hint, since we are computing the average regret, it will in fact be the difference between the
         value of q^* corresponding to the current arm, and the q^* value for the optimal arm.
         Remember it is 0 if the optimal action is selected.
         """
        reward = self.q_star[a] + np.random.randn() #!b
        regret = self.q_star[self.optimal_action] - self.q_star[a] #!b
        return reward, regret

    def __str__(self):
        return f"{type(self).__name__}_{self.q_star_mean}"

"""
Helper function for running a bunch of bandit experiments and plotting the results.

The function will run the agents in 'agents' (a list of bandit agents) 
on the bandit environment 'bandit' and plot the result.

Each agent will be evaluated for num_episodes episodes, and one episode consist of 'steps' steps.
However, to speed things up you can use cache, and the bandit will not be evaluated for more than 
'max_episodes' over all cache runs. 

"""
def eval_and_plot(bandit, agents, num_episodes=2000, max_episodes=2000, steps=1000, labels=None, use_cache=True):
    if labels is None:
        labels = [str(agent) for agent in agents]

    f, axs = plt.subplots(nrows=3, ncols=1)
    f.set_size_inches(10,7)
    (ax1, ax2, ax3) = axs
    for i,agent in enumerate(agents):
        rw, oa, regret, num_episodes = run_agent(bandit, agent, episodes=num_episodes, max_episodes=max_episodes, steps=steps, use_cache=use_cache)
        ax1.plot(rw, label=labels[i])
        ax2.plot(oa, label=labels[i])
        ax3.plot(regret, label=labels[i])

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Steps")

    ax1.set_ylabel("Average Reward")
    ax2.set_ylabel("% optimal action")
    ax3.set_ylabel("Regret $L_t$")
    ax3.legend()
    f.suptitle(f"Evaluated on {str(bandit)} for {num_episodes} episodes")

def run_agent(env, agent, episodes=2000, max_episodes=2000, steps=1000, use_cache=False, verbose=True):
    """
    Helper function. most of the work involves the cache; the actual training is done by 'train'.
    """
    C_regrets_cum_sum, C_oas_sum, C_rewards_sum, C_n_episodes = 0, 0, 0, 0
    if use_cache:
        cache = f"cache/{str(env)}_{str(agent)}_{steps}.pkl"
        if cache_exists(cache):
            print("> Reading from cache", cache)
            C_regrets_cum_sum, C_oas_sum, C_rewards_sum, C_n_episodes = cache_read(cache)

    regrets = []
    rewards = []
    cruns = max(0, min(episodes, max_episodes - C_n_episodes)) # Missing runs.
    for _ in tqdm(range(cruns), file=sys.stdout, desc=str(agent),disable=not verbose):
        stats, traj = train(env, agent, max_steps=steps, verbose=False, return_trajectory=True)
        regret = np.asarray([r['average_regret'] for r in traj[0].env_info])
        regrets.append(regret)
        rewards.append(traj[0].reward)

    regrets_cum_sum = C_regrets_cum_sum
    oas_sum = C_oas_sum
    rewards_sum = C_rewards_sum
    episodes = C_n_episodes
    if len(regrets) > 0:
        regrets_cum_sum += np.cumsum(np.sum(np.stack(regrets), axis=0))
        oas_sum += np.sum(np.stack(regrets) == 0, axis=0)
        rewards_sum += np.sum(np.stack(rewards), axis=0)
        episodes += cruns
    if use_cache and cruns > 0:
        cache_write((regrets_cum_sum, oas_sum, rewards_sum, episodes), cache, protocol=4)
    return rewards_sum/episodes, oas_sum/episodes, regrets_cum_sum/episodes, episodes
