import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.simple_agents import BasicAgent
from irlc.ex08.bandits import StationaryBandit, eval_and_plot
from irlc import savepdf

class NonstationaryBandit(StationaryBandit):
    def __init__(self, k, q_star_mean=0, reward_change_std=0.01):
        self.reward_change_std = reward_change_std
        super().__init__(k, q_star_mean)

    def bandit_step(self, a): #!f
        """ Implement the non-stationary bandit environment (as described in \cite{sutton}).
        Hint: use reward_change_std * np.random.randn() to generate a single random number with the given std.
         then add one to each coordinate. Remember you have to compute the regret as well, see StationaryBandit for ideas.
         (remember the optimal arm will change when you add noise to q_star) """
        self.q_star += self.reward_change_std * np.random.randn(self.k)
        self.optimal_action = np.argmax(self.q_star)
        return super().bandit_step(a)

    def __str__(self):
        return f"{type(self).__name__}_{self.q_star_mean}_{self.reward_change_std}"


class MovingAverageAgent(BasicAgent):
    """
    The simple bandit from \cite[Section 2.4]{sutton}, but with moving average alpha
    as described in \cite[Eqn. (2.3)]{sutton}
    """
    def __init__(self, env, epsilon, alpha): #!f
        self.alpha=alpha
        super().__init__(env, epsilon=epsilon)

    def train(self, s, a, r, sp, done=False): #!f
        self.Q[a] = self.Q[a] + self.alpha * (r-self.Q[a])

    def __str__(self):
        return f"{type(self).__name__}_{self.epsilon}_{self.alpha}"


if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    epsilon = 0.1
    alphas = [0.15, 0.1, 0.05]

    bandit = NonstationaryBandit(k=10) #!b

    agents = [BasicAgent(bandit, epsilon=epsilon)]
    agents += [MovingAverageAgent(bandit, epsilon=epsilon, alpha=alpha) for alpha in alphas] #!b

    labels = [f"Basic agent, epsilon={epsilon}"]
    labels += [f"Mov.avg. agent, epsilon={epsilon}, alpha={alpha}" for alpha in alphas] #!b #!b
    use_cache = True #!r use_cache = False # Set this to True to use cache (after code works!)
    eval_and_plot(bandit, agents, steps=10000, num_episodes=200, labels=labels, use_cache=use_cache)
    savepdf("nonstationary_bandits")
    plt.show()