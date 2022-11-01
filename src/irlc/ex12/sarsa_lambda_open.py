from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc import PlayWrapper, train, VideoMonitor

def keyboard_play(Agent, method_label='MC', num_episodes=1000, alpha=0.5, autoplay=False, **args):
    print("Evaluating", Agent, "on the open gridworld environment.")
    print("Press p to follow the agents policy or use the keyboard to input actions")
    print("(Please be aware that Sarsa, N-step Sarsa, and Sarsa(Lambda) do not always make the right updates when you input actions with the keyboard)")

    env = OpenGridEnvironment()
    try:
        agent = Agent(env, gamma=0.99, epsilon=0.1, alpha=alpha, **args)
    except Exception as e: # If it is a value agent without the epsilon.
        agent = Agent(env, gamma=0.99, alpha=alpha, **args)
    agent = PlayWrapper(agent, env,autoplay=autoplay)
    env = VideoMonitor(env, agent=agent, fps=100, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': method_label})
    train(env, agent, num_episodes=num_episodes)
    env.close()

if __name__ == "__main__":
    """ 
    Example: Play a three episodes and save a snapshot of the Q-values as a .pdf
    """
    env = OpenGridEnvironment()
    agent = SarsaLambdaAgent(env, gamma=0.99, epsilon=0.1, alpha=.5)
    env = VideoMonitor(env, agent=agent, agent_monitor_keys=('pi', 'Q'), render_kwargs={'method_label': "Sarsa(Lambda)"})
    train(env, agent, num_episodes=3)
    env.savepdf("sarsa_lambda_opengrid")
    env.close()

    """ Example: Keyboard play 
    You can input actions manually with the keyboard, but the Q-values are not necessarily updates correctly in this mode. Can you tell why? 
    You can let the agent play by pressing `p`, in which case the Q-values will be updated correctly. """

    keyboard_play(SarsaLambdaAgent, method_label="Sarsa(Lambda)", lamb=0.8)

