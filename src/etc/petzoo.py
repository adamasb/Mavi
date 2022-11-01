import numpy as np

# from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_v2
env = simple_v2.env()
# Lets implement the cooperative listener thingy.

# env = knights_archers_zombies_v10.env()
def pi1(s):
    return 4
    pass

def pi2(s):
    return 0

policies = [pi1, pi2]
s = env.reset()
env.env.env.world
import time
while True:
    X = []
    oo = 0
    for na, agent in enumerate(env.agent_iter()):
        observation, reward, done, info = env.last()
        if done:
            break
        # action = policy(observation, agent)
        a = na % len(env.agents)
        print(a)
        env.step(policies[a](observation))
        env.render('human')
        time.sleep(0.1)
        print("pos", observation, "diff", observation - oo)
        X.append(observation)
        # print("position", env.agents[0].state.p_pos)
        oo = observation
        if done:
            break
    X = np.stack(X)
    import matplotlib.pyplot as plt
    plt.plot(X)
    plt.show()

    b = 234




    a = 234
