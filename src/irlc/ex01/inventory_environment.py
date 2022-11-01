import gym
import numpy as np
from gym.spaces.discrete import Discrete
from irlc.ex01.agent import Agent, train

class InventoryEnvironment(gym.Env): #!s=a
    def __init__(self, N=2):
        self.N = N                               # planning horizon
        self.action_space      = Discrete(3)     # Possible actions {0, 1, 2}
        self.observation_space = Discrete(3)     # Possible observations {0, 1, 2}

    def reset(self):
        self.s = 0                               # reset initial state x0=0
        self.k = 0                               # reset time step k=0
        return self.s, {}                        # Return the state we reset to (and an empty dict)

    def step(self, a): #!f
        w = np.random.choice(3, p=(.1, .7, .2))       # Generate random disturbance
        s_next = max(0, min(2, self.s-w+a))           # next state; x_{k+1} =  f_k(x_k, u_k, w_k) #!b
        reward = -(a + (self.s + a - w)**2)           # reward = -cost      = -g_k(x_k, u_k, w_k)
        terminated = self.k == self.N-1               # Have we terminated? (i.e. is k==N-1)
        self.s = s_next                               # update environment state
        self.k += 1                                   # update current time step #!b
        return s_next, reward, terminated, False, {}  # return transition information  #!s=a

class RandomAgent(Agent): #!s=b
    def pi(self, s, k=None): #!f
        """ Return action to take in state s at time step k """
        return np.random.choice(3) # Return a random action

    def train(self, s, a, r, sp, done=False):
        """ Called at each step of the simulation to allow the agent to train.
        The agent was in state s, took action a, ended up in state sp (with reward r).
        'done' is a bool which indicates if the environment terminated when transitioning to sp. """
        pass #!s

def simplified_train(env, agent): #!s=d
    s, _ = env.reset()
    J = 0  # Accumulated reward for this rollout
    for k in range(1000): #!f
        a = agent.pi(s, k)
        sp, r, terminated, truncated, metadata = env.step(a)
        agent.train(s, a, sp, r, terminated)
        s = sp
        J += r
        if terminated or truncated:
            break
    return J #!s

def run_inventory():
    env = InventoryEnvironment() #!o=a #!s=train1
    agent = RandomAgent(env)
    stats, _ = train(env,agent,num_episodes=1,verbose=False)  # Perform one rollout.
    print("Accumulated reward of first episode", stats[0]['Accumulated Reward']) #!s
    # I recommend inspecting 'stats' in a debugger; why do you think it is a list of length 1?

    stats, _ = train(env, agent, num_episodes=1000,verbose=False)  # do 1000 rollouts #!s=train2
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[RandomAgent class] Average cost of random policy J_pi_random(0)=", -avg_reward) #!s
    # Try to inspect stats again in a debugger here. How long is the list now?

    stats, _ = train(env, Agent(env), num_episodes=1000,verbose=False)  # Perform 1000 rollouts using Agent class #!s=train3
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[Agent class] Average cost of random policy J_pi_random(0)=", -avg_reward)  #!s
    #!o

    """ Second part: Using the simplified training method. I.e. do not use train() below.
     You can find some pretty strong hints about what goes on in simplified_train in the lecture slides for today. """
    avg_reward_simplified_train = np.mean( [simplified_train(env, agent) for i in range(1000)]) #!o=c
    print("[simplified train] Average cost of random policy J_pi_random(0) =", -avg_reward_simplified_train)  #!o



if __name__ == "__main__":
    run_inventory()