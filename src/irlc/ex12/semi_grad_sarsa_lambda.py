import gym
import numpy as np
from irlc.ex01.agent import train
from irlc import main_plot, savepdf
import matplotlib.pyplot as plt
from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa

class LinearSemiGradSarsaLambda(LinearSemiGradSarsa):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9, q_encoder=None):
        """
        Sarsa(Lambda) with linear feature approximators (see \cite[Section 12.7]{sutton}).
        """
        super().__init__(env, gamma, alpha=alpha, epsilon=epsilon, q_encoder=q_encoder)
        self.z = np.zeros(self.Q.d) # Vector to store eligibility trace (same dimension as self.w)
        self.lamb = lamb # lambda in Sarsa(lambda). We cannot use the reserved keyword 'lambda'.

    def pi(self, s, k=None):
        if self.t == 0:
            self.a = self.pi_eps(s)
            self.x = self.Q.x(s,self.a)
            self.Q_old = 0
        return self.a

    def train(self, s, a, r, sp, done=False):
        a_prime = self.pi_eps(sp) if not done else -1
        x_prime = self.Q.x(sp, a_prime) if not done else None
        """
        Update the eligibility trace self.z and the weights self.w here. 
        Note Q-values are approximated as Q = w @ x.
        We use Q_prime = w * x(s', a') to denote the new q-values for (stored for next iteration as in the pseudo code)
        """
        Q = self.Q.w @ self.x #!b
        Q_prime = self.Q.w @ x_prime if not done else None
        delta = r + (self.gamma * Q_prime if not done else 0) - Q
        self.z = self.gamma * self.lamb * self.z + (1-self.alpha * self.gamma * self.lamb *self.z @ self.x) * self.x
        self.Q.w += self.alpha * (delta + Q - self.Q_old) * self.z - self.alpha * (Q-self.Q_old) * self.x #!b Update z, w
        if done: # Reset eligibility trace and time step t as in Sarsa.
            self.t = 0
            self.z = self.z * 0
        else:
            self.Q_old, self.x, self.a = Q_prime, x_prime, a_prime
            self.t += 1

    def __str__(self):
        return f"LinearSarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"


from irlc.ex11.semi_grad_q import experiment_q, x, episodes
from irlc.ex11.semi_grad_sarsa import experiment_sarsa
from irlc.ex09 import envs
experiment_sarsaL = "experiments/mountaincar_sarsaL"
num_of_tilings = 8
alpha = 1 / num_of_tilings / 2 # learning rate

def plot_including_week10(experiments, output):
    exps = ["../ex11/" + e for e in [experiment_q, experiment_sarsa]] + experiments

    main_plot(exps, x_key=x, y_key='Length', smoothing_window=30, resample_ticks=100)
    savepdf(output)
    plt.show()

    # Turn off averaging
    main_plot(exps, x_key=x, y_key='Length', smoothing_window=30, units="Unit", estimator=None, resample_ticks=100)
    savepdf(output+"_individual")
    plt.show()

if __name__ == "__main__":
    env = gym.make("MountainCar500-v0")
    for _ in range(5): # run experiment 10 times
        agent = LinearSemiGradSarsaLambda(env, gamma=1, alpha=alpha, epsilon=0)
        train(env, agent, experiment_sarsaL, num_episodes=episodes, max_runs=10)
    # Make plots (we use an external function so we can re-use it for the semi-gradient n-step controller)
    plot_including_week10([experiment_sarsaL], output="semigrad_sarsaL")