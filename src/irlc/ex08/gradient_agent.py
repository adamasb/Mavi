from irlc import savepdf
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.bandits import eval_and_plot, StationaryBandit
from irlc import Agent

class GradientAgent(Agent):
    def __init__(self, env, alpha=None, use_baseline=True):
        self.k = env.action_space.n
        self.alpha = alpha
        self.baseline=use_baseline
        self.H = np.zeros((self.k,))
        super().__init__(env)

    def Pa(self):
        """ This helper method returns the probability distribution P(A=a) of chosing the
        arm a as a vector
        """
        pi_a = np.exp(self.H)
        return pi_a / np.sum(pi_a)

    def pi(self, s, n=None):
        if n == 0:
            self.R_bar = 0  # average reward baseline
            self.H *= 0 # Reset H to all-zeros.
        self.n = n
        return np.random.choice( self.k, p=self.Pa() )

    def train(self, s, a, r, sp, done=False): #!f
        pi_a = self.Pa()
        for b in range(self.k):
            if b == a:
                self.H[b] += self.alpha * (r - self.R_bar) * (1 - pi_a[b])
            else:
                self.H[b] -= self.alpha * (r - self.R_bar) * pi_a[b]

        # pi_a, e = self.Pa(), np.zeros( (self.k,) )
        # e[a] = 1.0
        # self.H = self.H + self.alpha * (r - self.R_bar) * ( e * (1-pi_a) - (1-e) * pi_a)
        if self.baseline:
            self.R_bar = self.R_bar + (self.alpha if self.alpha is not None else 1/(self.n+1)) * (r - self.R_bar)

    def __str__(self):
        return f"{type(self).__name__}_{self.alpha}_{'baseline' if self.baseline else 'no_baseline'}"

if __name__ == "__main__":
    baseline_bandit = StationaryBandit(k=10, q_star_mean=4)
    alphas = [0.1, 0.4]
    agents = [GradientAgent(baseline_bandit, alpha=alpha, use_baseline=False) for alpha in alphas]
    agents += [GradientAgent(baseline_bandit, alpha=alpha, use_baseline=True) for alpha in alphas]

    labels = [f'Gradient Bandit alpha={alpha}' for alpha in alphas ]
    labels += [f'With baseline: Gradient Bandit alpha={alpha}' for alpha in alphas ]
    use_cache = False #!r use_cache = False
    eval_and_plot(baseline_bandit, agents, max_episodes=2000, num_episodes=100, labels=labels, use_cache=use_cache)
    savepdf("gradient_baseline")
    plt.show()



