from irlc.ex02.old.dp_pacman import SS1tiny, SS2tiny, DPPacmanModel, TERM
from irlc.ex09.mdp import MDP
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc.ex09.value_iteration_agent import ValueIterationAgent
from irlc.ex09.mdp import MDP2GymEnv
from irlc.ex09.value_iteration import value_iteration
import numpy as np
from irlc import train
from irlc import Timer

class PacmanInfiniteMDP(MDP):
    def __init__(self, layout_str, **kwargs):
        self.env = GymPacmanEnvironment(layout_str=layout_str, animate_movement=False)
        self.dp = DPPacmanModel(self.env, N=1)
        super().__init__(initial_state=self.env.reset().copy(), **kwargs)

    def is_terminal(self, state):
        return state in TERM

    def A(self, state):
        return self.dp.A(state, 0)

    def Psr(self, x, u):
        return { (sp, -self.dp.g(x, u, sp, 0) - self.dp.gN(sp) if self.is_terminal(sp) else 0): p for sp, p in self.dp.Pw(x, u, 0).items()}


class PacmanTiny2Env(MDP2GymEnv):
    def __init__(self, **kwargs):
        mdp = PacmanInfiniteMDP(SS2tiny, **kwargs)
        super().__init__(mdp)

class PacmanTiny1Env(MDP2GymEnv):
    def __init__(self, **kwargs):
        mdp = PacmanInfiniteMDP(SS1tiny, **kwargs)
        super().__init__(mdp)

def pacman_dp_J_sample(env, N=8, T=1000):
    # layout = "smallGrid"
    # env = GymPacmanEnvironment(layout=None, layout_str=layout_str, animate_movement=False)
    # model = DPPacmanModel(env, N=N)
    agent = ValueIterationAgent(env)
    mdp = env.mdp
    print("Simulating")
    timer = Timer()
    timer.tic("Training..")
    # t = time.time()
    stats, _ = train(env, agent, num_episodes=T,verbose=False)
    timer.toc()

    # J,pi = DP_stochastic(model)
    pi, v = value_iteration(env.mdp, gamma=1., theta=0.01, verbose=True)
    print()

    avg = np.mean( [s['Accumulated Reward']  for s in stats] )
    ll = np.mean( [s['Length'] for s in stats] )
    # print(f"Pacman environment {layout_str}: ")
    # print(f"> |S_1|={len(model.S(1))}, |S_N|={len(model.S(N))}")
    print(f"> VI computed cost: ", v[mdp.initial_state], "reward obtained by sampling", avg, "avg length", ll)
    env.close()
    return env.mdp, v, avg, ll


def simulate_1_game(env):
    N = 8
    # env = GymPacmanEnvironment(layout=None, layout_str=layout_str, animate_movement=False)
    # model = DPPacmanModel(env, N=N)
    agent = ValueIterationAgent(env)
    # env = TimeLimit(env, max_episode_steps=N)
    # env = PacmanWinWrapper(env)
    print("Simulating")
    timer = Timer()
    timer.tic('simulate 1 game')
    T = 1000
    stats, trajectories = train(env, agent, num_episodes=T, verbose=True, return_trajectory=True)
    s = env.reset()
    print(str(s))
    sp, r, done, ex = env.step("Stop")
    print(str(sp))
    max( [len(t.action) for t in trajectories] )
    timer.toc()
    print(timer.display())
    env.close()


def value_iter():
    print("Now for simulation")
    mdp = PacmanInfiniteMDP(SS2tiny, verbose=True)
    print(len(mdp.states))
    pi, v = value_iteration(mdp, gamma=1., theta=0.01, verbose=True)
    print(v[mdp.initial_state])

if __name__ == '__main__':
    env1 = PacmanTiny1Env()
    pacman_dp_J_sample(env1)

    env2 = PacmanTiny2Env()

    # simulate_1_game(env1)
    pacman_dp_J_sample(env2)

    # value_iter()