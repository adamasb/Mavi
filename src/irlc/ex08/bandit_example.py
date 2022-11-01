import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from irlc import Agent, train, savepdf #!s=a
    from irlc.ex08.bandits import StationaryBandit
    bandit = StationaryBandit(k=10) # A 10-armed bandit
    agent = Agent(bandit)  # Recall the agent takes random actions
    _, trajectories = train(bandit, agent, return_trajectory=True, num_episodes=1, max_steps=500)
    plt.plot(trajectories[0].reward)
    plt.xlabel("Time step")
    plt.ylabel("Reward per time step") #!s
    savepdf("dumbitA")
    plt.show()

    agent = Agent(bandit)  # Recall the agent takes random actions  #!s=b
    for i in range(10):
        _, trajectories = train(bandit, agent, return_trajectory=True, num_episodes=1, max_steps=500)
        regret = np.asarray([r['average_regret'] for r in trajectories[0].env_info])
        cum_regret = np.cumsum(regret)
        plt.plot(cum_regret, label=f"Episode {i}")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Accumulated Regret") #!s
    savepdf("dumbitB")
    plt.show()

