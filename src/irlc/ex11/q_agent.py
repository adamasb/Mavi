from irlc.ex09.mdp import GymEnv2MDP
from irlc.ex09.rl_agent import TabularAgent
from irlc import train
import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex09.value_iteration_agent import ValueIterationAgent

class QAgent(TabularAgent):
    """
    Implement the Q-learning agent \cite[Section 6.5]{sutton}
    Note that the Q-datastructure already exist, as do helper functions useful to compute an epsilon-greedy policy.
    You can access these as

    > self.Q[s,a] = 31 # Set a Q-value.

    See the TabularAgent class for more information.
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s, k=None): #!f Implement the epsilon-greedy policy here.
        """
        Return current action using epsilon-greedy exploration. Look at the TabularAgent class
        for ideas.
        """
        return self.pi_eps(s)

    def train(self, s, a, r, sp, done=False): #!f Update the Q[s,a]-values here.
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp,a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s, a] = ...

        You may want to look at self.Q.get_optimal_action(state) to compute a = argmax_a Q[s,a].
        """
        if not done:
            a_star = self.Q.get_optimal_action(sp)
        self.Q[s,a] += self.alpha * (r + self.gamma * (0 if done else self.Q[sp,a_star]) - self.Q[s,a])

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

q_exp = f"experiments/cliffwalk_Q"
epsilon = 0.1
max_runs = 10
alpha = 0.5
def cliffwalk():
    env = gym.make('CliffWalking-v0')
    agent = QAgent(env, epsilon=epsilon, alpha=alpha)

    train(env, agent, q_exp, num_episodes=200, max_runs=max_runs)


    # As a baseline, we set up/evaluate a value-iteration agent to get an idea about the optimal performance.
    # To do so, we need an MDP object. We create an MDP object out of the gym environment below.
    # You can look at the code if you like, but it is simply a helper function to convert from one datastructure to another,
    # and all it does is to give a MDP object which is needed for our value-iteration implementation from the previous
    # week.
    mdp = GymEnv2MDP(env)
    vi_exp = "experiments/cliffwalk_VI"
    Vagent = ValueIterationAgent(env, mdp=mdp, epsilon=epsilon)
    train(env, Vagent, vi_exp, num_episodes=200, max_runs=max_runs)

    vi_exp_opt = "experiments/cliffwalk_VI_optimal"
    Vagent_opt = ValueIterationAgent(env, mdp=mdp, epsilon=0) # Same, but with epsilon=0
    train(env, Vagent_opt, vi_exp_opt, num_episodes=200, max_runs=max_runs)

    exp_names = [q_exp, vi_exp, vi_exp_opt]
    return env, exp_names

if __name__ == "__main__":
    for _ in range(10):
        env, exp_names = cliffwalk()
    main_plot(exp_names, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    savepdf("Q_learning_cliff")
    plt.show()
