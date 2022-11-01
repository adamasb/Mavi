import numpy as np
import matplotlib.pyplot as plt
from irlc.ex09.rl_agent import ValueAgent
from irlc import savepdf
from irlc.ex01.agent import train
import gym_minigrid # For the small gridwordl.

class TD0ValueAgent(ValueAgent):
    def __init__(self, env, policy=None, gamma=0.99, alpha=0.05, v_init_fun=None):
        self.alpha = alpha
        super().__init__(env, gamma=gamma, policy=policy, v_init_fun=v_init_fun)

    def train(self, s, a, r, sp, done=False): #!f
        self.v[s] += self.alpha * (r + self.gamma * (self.v[sp] if not done else 0) - self.v[s])

    def __str__(self):
        return f"TD0Value_{self.gamma}_{self.alpha}"

def value_function_test(env, agent, v_true, episodes=200):
    err = []
    for t in range(episodes):
        train(env, agent, num_episodes=1, verbose=False)
        err.append( np.mean( [(v_true - v0) ** 2 for k, v0 in agent.v.items()] ) )
    return np.asarray(err)

if __name__ == "__main__":
    envn = "SmallGridworld-v0"

    from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment #!s=a
    from irlc import VideoMonitor
    env = SuttonCornerGridEnvironment() # Make the gridworld environment itself #!s

    gamma = 1   #!s=b
    agent = TD0ValueAgent(env, gamma=gamma, alpha=0.05) # Make a TD(0) agent
    train(env, agent, num_episodes=2000) # Train for 2000 episodes #!s
    env = VideoMonitor(env, agent=agent) # Add a video monitor, the environment will now show an animation #!s=c
    train(env,agent,num_episodes=1) # Train for a (single) new episode
    env.plot() # Plot the current state of the environment/agent
    plt.title(f"TD0 evaluation of {envn}")
    savepdf("TD_value_random_smallgrid")
    plt.show() #!s
