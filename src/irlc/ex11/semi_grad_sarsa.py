import matplotlib.pyplot as plt
from irlc import main_plot, savepdf
from irlc.ex01.agent import train
import numpy as np
import gym
from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
np.seterr(all='raise')

class LinearSemiGradSarsa(LinearSemiGradQAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, q_encoder=None):
        """ Implement the Linear semi-gradient Sarsa method from \cite[Section 10.1]{sutton}"""
        super().__init__(env, gamma, epsilon=epsilon, alpha=alpha, q_encoder=q_encoder)
        self.t = 0

    def pi(self, s, k=None): #!f
        return self.a if self.t > 0 else super().pi(s)

    def train(self, s, a, r, sp, done=False):
        a_prime = super().pi(sp) #!b
        delta = r + (0 if done else self.gamma * self.Q(sp, a_prime)) - self.Q(s, a)
        self.Q.w += self.alpha * delta * self.Q.x(s,a)
        self.a = a_prime #!b

        if sum(np.abs(self.Q.w)) > 1e5: raise Exception("Weights diverged. Decrease alpha")
        self.t += 1
        if done:
            self.t = 0

    def __str__(self):
        return f"LinSemiGradSarsa{self.gamma}_{self.epsilon}_{self.alpha}"

experiment_sarsa = "experiments/mountaincar_Sarsa"

if __name__ == "__main__":
    from irlc.ex11.semi_grad_q import experiment_q, alpha, x
    from irlc.ex09 import envs

    env = gym.make("MountainCar500-v0")
    for _ in range(10):
        agent = LinearSemiGradSarsa(env, gamma=1, alpha=alpha, epsilon=0)
        train(env, agent, experiment_sarsa, num_episodes=300, max_runs=10)

    main_plot(experiments=[experiment_q, experiment_sarsa], x_key=x, y_key='Length', smoothing_window=30)
    savepdf("semigrad_q_sarsa")
    plt.show()

    # Turn off averaging
    main_plot(experiments=[experiment_q, experiment_sarsa], x_key=x, y_key='Length', smoothing_window=30, units="Unit", estimator=None)
    savepdf("semigrad_q_sarsa_individual")
    plt.show()