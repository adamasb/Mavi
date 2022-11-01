import matplotlib.pyplot as plt
from irlc.ex11.q_agent import QAgent
from irlc import main_plot, savepdf
from irlc.ex01.agent import train
from irlc.ex11.q_agent import cliffwalk, alpha, epsilon

class SarsaAgent(QAgent):
    """ Implement the Sarsa control method from \cite[Section 6.4]{sutton}. It is recommended you complete
    the Q-agent first because the two methods are very similar and the Q-agent is easier to implement. """
    def __init__(self, env, gamma=1, alpha=0.5, epsilon=0.1):
        self.t = 0 # indicate we are at the beginning of the episode
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s,k=None):
        if self.t == 0: #!f
            """ we are at the beginning of the episode. Generate a by being epsilon-greedy"""
            return self.pi_eps(s)
        else: #!f
            """ Return the action self.a you generated during the train where you know s_{t+1} """
            return self.a

    def train(self, s, a, r, sp,done=False):
        """
        generate A' as self.a by being epsilon-greedy. Re-use code from the Agent class.
        """
        self.a = self.pi_eps(sp) if not done else -1 #!b #!b self.a = ....
        """ now that you know A' = self.a, perform the update to self.Q[s,a] here """
        delta = r + (self.gamma * self.Q[sp,self.a] if not done else 0) - self.Q[s,a] #!b
        self.Q[s,a] += self.alpha * delta #!b
        self.t = 0 if done else self.t + 1 # update current iteration number

    def __str__(self):
        return f"Sarsa{self.gamma}_{self.epsilon}_{self.alpha}"

sarsa_exp = f"experiments/cliffwalk_Sarsa"
if __name__ == "__main__":
    env, q_experiments = cliffwalk()  # get results from Q-learning
    agent = SarsaAgent(env, epsilon=epsilon, alpha=alpha)
    for _ in range(10):
        train(env, agent, sarsa_exp, num_episodes=200, max_runs=10)
    main_plot(q_experiments + [sarsa_exp], smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q and Sarsa learning on " + env.spec._env_name)
    savepdf("QSarsa_learning_cliff")
    plt.show()
