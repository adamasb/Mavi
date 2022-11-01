import numpy as np
import matplotlib.pyplot as plt
from irlc.ex08.bandits import StationaryBandit, eval_and_plot
from irlc import Agent
from irlc import savepdf

class BasicAgent(Agent):
    """
    Simple bandit as described on \cite[Section 2.4]{sutton}.
    """
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.k = env.action_space.n
        self.epsilon = epsilon

    def pi(self, s, t=None): #!s=a
        """ Since this is a bandit, s=None and can be ignored, while k refers to the time step in the current episode """
        if t == 0:
            # At step 0 of episode. Re-initialize data structure. #!s
            self.Q = np.zeros((self.k,)) #!b
            self.N = np.zeros((self.k,)) #!b
        # compute action here #!s=a #!s
        return np.random.randint(self.k) if np.random.rand() < self.epsilon else np.argmax(self.Q) #!b #!b

    def train(self, s, a, r, sp, done=False): #!f
        """ Since this is a bandit, s=sp=None and can be ignored, and done=False and can also be ignored. """
        self.N[a] = self.N[a] + 1
        self.Q[a] = self.Q[a] + 1/self.N[a] * (r-self.Q[a])

    def __str__(self):
        return f"BasicAgent_{self.epsilon}"

if __name__ == "__main__":
    N = 100000
    S = [np.max( np.random.randn(10) ) for _ in range(100000) ]
    print( np.mean(S), np.std(S)/np.sqrt(N) )

    use_cache = True #!r use_cache = False # Set this to True to use cache (after code works!)
    from irlc.utils.timer import Timer
    timer = Timer(start=True)
    R = 100
    steps = 1000
    env = StationaryBandit(k=10) #!s=ex
    agents = [BasicAgent(env, epsilon=.1), BasicAgent(env, epsilon=.01), BasicAgent(env, epsilon=0) ]
    eval_and_plot(env, agents, num_episodes=100, steps=1000, max_episodes=150, use_cache=use_cache)
    savepdf("bandit_epsilon")
    plt.show() #!s
    print(timer.display())
