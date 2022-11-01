import gym
from ex11.sarsa_lambda_agent import SarsaLambdaAgent
from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
from irlc.ex11.q_agent import QAgent
from irlc.ex11.sarsa_agent import SarsaAgent
from scratch.old import value_iteration
import matplotlib.pyplot as plt
from irlc.ex01.agent import train
from irlc import savepdf
from gym.wrappers import TimeLimit
from irlc.ex02.frozen_lake_dp import plot_value_function
import numpy as np
from irlc import main_plot
from irlc import log_time_series
from irlc.ex01.agent import ValueAgent
from irlc.utils.irlc_plot import existing_runs

def base_env(env):
    return env if hasattr(env, "s") else env.model


def compare_by_return(env, agents=None, num_episodes=500, max_episode_steps=None, max_runs=5):
    env,name = prepare(env, max_episode_steps)

    exps = []
    for agc in agents:
        agent = agc(env)
        expn = f"experiments/{envn}_{agent}_{num_episodes}"
        if existing_runs(expn) <= max_runs:
            train(env, agent, expn, num_episodes=num_episodes)
        exps.append(expn)

    main_plot(exps, smoothing_window=20)
    return exps

def prepare(env, max_episode_steps=None):
    if max_episode_steps:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    name = env.spec._env_name
    return env, name

def stepwise_eval_episode(env, agent):
    Vs = []
    env.reset()
    while True:
        _, _, done = train(env, agent, num_episodes=1, verbose=False, reset=False, max_steps=1)
        Vs.append(agent.get_v())
        if done:
            break
    return Vs

def stepwise_eval(env, agent, num_episodes=200):
    Vs = []
    for i in range(num_episodes):
        Vs += stepwise_eval_episode(env, agent)
    return Vs

def value_function_after_episodes(env, agent, episodes=(1,5,50), max_episode_steps=100):
    # get the value function after the given number of episodes and plot it as a grid.
    H,W = 1,len(episodes)
    env, name = prepare(env, max_episode_steps=max_episode_steps)
    V = []
    plt.figure(figsize=(6 * len(episodes), 6))
    for i in range(max(episodes)+1):
        stepwise_eval_episode(env, agent)
        if i in episodes:
            plt.subplot(H, W, episodes.index(i)+1)
            plot_value_function(base_env(env), agent.get_v(), figsize=None)
            plt.title(f"Episodes={i}")
            V.append(agent.get_v())
    return V

def agent_value_function_mse(env, agents, gamma=1, repeats=5, num_episodes=200, max_episode_steps=100, max_runs=5,
                             max_xticks_to_log=300):
    """ If 'Agents' are ordinary control agents compute optimal value function using policy iteration
    otherwise obtain optimal policy using value iteration and test using optimal policy.
    """
    env, name = prepare(env, max_episode_steps=max_episode_steps)
    pi_opt, v_opt = value_iteration(env,gamma=gamma)
    lds = []
    for agent in agents:
        for r in range(repeats):
            if isinstance(agent, ValueAgent):
                raise Exception()
                # Nothing wrong with this, but seems not that important.
                ag = agent(env, policy=pi_opt)
            else:
                ag = agent(env,gamma=gamma)
            ld = f"experiments/vMSE_{name}_{ag}"
            if ld not in lds:
                lds.append(ld)
            if existing_runs(ld) >= max_runs:
                print(ld +"> existed, continuing..")
                continue
            Vs = stepwise_eval(env,ag,num_episodes=num_episodes)
            ts = []
            for i,va in enumerate(Vs):
                mse = np.sqrt( np.sum( [ (v_opt[s] - va[s])**2 for s in range(len(v_opt)) ] ) ) / len(v_opt)
                ts.append({'mse': mse})
            log_time_series(ld, ts, max_xticks_to_log)

    print("plotting... ", ", ".join(lds))
    main_plot(experiments=lds, y_key="mse", x_key="Steps")
    # plt.show()
    print("Done!")


"""
logdir = f"{experiment}/{date}"
    with LazyLog(logdir, data) as logz:
        for l in list_obs:
            for k,v in l.items():
                logz.log_tabular(k,v)
            logz.dump_tabular(verbose=False)
"""

if __name__ == "__main__":
    envn = "Gambler-v0"  # does not work bc stupid   action spaces (fix later)
    envn = "SmallGridworld-v0"

    env = gym.make(envn)
    agents = [QAgent, SarsaAgent, SarsaLambdaAgent, SarsaNAgent]

    print("Comparing agents by value function MSE...")
    agent_value_function_mse(env, agents)
    plt.title("Agents compared by MSE to true value function: " + envn)
    # plt.ylim([-10,0])
    savepdf("agent_mse_" + envn)
    plt.show()


    """
    Test agents based on how well they can estimate the optimal value function.
    """
    print("Plotting value function after certain number of episodes...")
    value_function_after_episodes(env, QAgent(env))
    plt.show()

    """
    Compare agents by how much return they obtain after a given number of episodes
    """
    compare_by_return(env, agents, max_episode_steps=100)  # otherwise they get stuck
    ename = env.spec._env_name
    plt.title("Agents compared by return: " + envn)
    plt.ylim([-200, 0])
    savepdf("cmp_by_return_"+ename)
    plt.show()