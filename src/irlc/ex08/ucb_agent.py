import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.simple_agents import BasicAgent
from irlc import savepdf
from irlc import Agent

class UCBAgent(Agent):
    def __init__(self, env, c=2):
        self.c = c
        super().__init__(env)

    def train(self, s, a, r, sp, done=False): #!f Train agent here
        self.N[a] += 1
        self.Q[a] += 1/self.N[a] * (r - self.Q[a])

    def pi(self, s, t=None):
        if t == 0: #!f Reset agent (i.e., make it ready to learn in a new episode with a new optimal action)
            """ Initialize the agent"""
            k = self.env.action_space.n
            self.Q = np.zeros((k,))
            self.N = np.zeros((k,))
        return np.argmax( self.Q + self.c * np.sqrt( np.log(t+1)/(self.N+1e-8)  )  ) #!b #!b Compute (and return) optimal action

    def __str__(self):
        return f"{type(self).__name__}_{self.c}"

from irlc.ex08.bandits import StationaryBandit, eval_and_plot
if __name__ == "__main__":
    """ Reproduce \cite[Fig. 2.4]{sutton} comparing UCB agent to epsilon greedy """
    runs, use_cache = 2000, True #!r runs, use_cache = 100, False
    c = 2
    eps = 0.1

    steps = 1000
    env = StationaryBandit(k=10)
    agents = [UCBAgent(env,c=c), BasicAgent(env, epsilon=eps)]
    eval_and_plot(bandit=env, agents=agents, num_episodes=runs, steps=steps, max_episodes=2000, use_cache=use_cache)
    savepdf("UCB_agent")
    plt.show()
