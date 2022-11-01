import matplotlib.pyplot as plt
from irlc.ex08.simple_agents import BasicAgent
from irlc.ex08.bandits import StationaryBandit, eval_and_plot
from irlc.ex08.nonstationary import MovingAverageAgent, NonstationaryBandit
from irlc.ex08.gradient_agent import GradientAgent
from irlc.ex08.ucb_agent import UCBAgent
from irlc import savepdf
import time

if __name__ == "__main__":
    print("Ladies and gentlemen. It is time for the graaand bandit race")
    def intro(bandit, agents):
        print("We are live from the beautiful surroundings where they will compete in:")
        print(bandit)
        print("Who will win? who will have the most regret? we are about to find out")
        print("in a minute after a brief word from our sponsors")
        time.sleep(1)
        print("And we are back. Let us introduce todays contestants:")
        for a in agents:
            print(a)
        print("And they are off!")
    epsilon = 0.1
    alpha = 0.1
    c = 2
    bandit1 = StationaryBandit(k=10) #!b #!b Define the bandit here: bandit1 = ...
    agents = [BasicAgent(bandit1, epsilon=epsilon)] #!b
    agents += [MovingAverageAgent(bandit1, epsilon=epsilon, alpha=alpha)]
    agents += [GradientAgent(bandit1, alpha=alpha,use_baseline=False) ]
    agents += [GradientAgent(bandit1, alpha=alpha,use_baseline=True) ]
    agents += [UCBAgent(bandit1, c=2)] #!b define agents list here
    labels = ["Basic", "Moving avg.", "gradient", "Gradient+baseline", "UCB"]
    '''
    Stationary, no offset. Vanilla setting.
    '''
    intro(bandit1, agents)
    eval_and_plot(bandit1, agents, max_episodes=2000, labels=labels) #!b #!b Call eval_and_plot here
    plt.suptitle("Stationary bandit (no offset)")
    savepdf("grand_race_1")
    plt.show()
    '''
    Stationary, but with offset
    '''
    print("Whew what a race. Let's get ready to next round:")
    bandit2 = StationaryBandit(k=10, q_star_mean=4) #!b #!b Define bandit2 = ... here
    intro(bandit2, agents)
    eval_and_plot(bandit2, agents, max_episodes=2000, labels=labels) #!b #!b Call eval_and_plot here
    plt.suptitle("Stationary bandit (with offset)")
    savepdf("grand_race_2")
    plt.show()
    '''
    Long (nonstationary) simulations
    '''
    print("Whew what a race. Let's get ready to next round which will be a long one.")
    bandit3 = NonstationaryBandit(k=10)  #!b #!b define bandit3 here
    intro(bandit3, agents)
    eval_and_plot(bandit3, agents, max_episodes=2000, steps=10000, labels=labels)  #!b #!b call eval_and_plot here
    plt.suptitle("Non-stationary bandit (no offset)")
    savepdf("grand_race_3")
    plt.show()

    '''
    Stationary, no offset, long run. Exclude stupid bandits.
    '''
    agents2 = []
    agents2 += [GradientAgent(bandit1, alpha=alpha, use_baseline=False)]
    agents2 += [GradientAgent(bandit1, alpha=alpha, use_baseline=True)]
    agents2 += [UCBAgent(bandit1, c=2)]
    labels = ["Gradient", "Gradient+baseline", "UCB"]
    intro(bandit1, agents2)
    eval_and_plot(bandit1, agents2, steps=10000, labels=labels)   #!b #!b Call eval_and_plot here
    plt.suptitle("Stationary bandit (no offset)")
    savepdf("grand_race_4")
    plt.show()
