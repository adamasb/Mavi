import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from irlc import Agent
from mazeenv.maze_environment import MazeEnvironment
from irlc.gridworld.gridworld_environments import CliffGridEnvironment, BookGridEnvironment
from irlc import PlayWrapper, VideoMonitor
from irlc import Agent, train
import ray
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.a3c import a3c, a3c_torch_policy


class VIModel():

    pass

class VIAgent(Agent):
    def __init__(self, env):
        self.v = defaultdict(lambda: 0)
        self.k = 0
        self.buffer = ReplayBuffer(capacity=1000, storage_unit=StorageUnit.TIMESTEPS)
        super().__init__(env)

    def pi(self, s, k=None):
        # take random actions and randomize the value function V for visualization.
        # also compute logrpob
        self._log_prob = 0
        return self.env.action_space.sample()

    def Phi(self, s):
        rout = s[:, :, 2]
        rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
        p = 1 - s[:, :, 0]
        return (rin, rout, p)

    def train(self, s, a, r, sp, done=False):
        # do training stuff here (save to buffer, call torch, whatever)
        self.buffer.add(SampleBatch({SampleBatch.OBS: [s], SampleBatch.ACTIONS: [a], SampleBatch.REWARDS: [r], SampleBatch.NEXT_OBS: [sp], SampleBatch.ACTION_LOGP: [self._log_prob], SampleBatch.T: [self.k]}))

        w = VIP(s, self.Phi(s))
        vv = w[:,:,min(self.k, w.shape[2]-1)]
        for i,j in self.v:
            self.v[i,j] = vv[j, i] # annoying transpose
        self.k += 1
        if done:
            self.k = 0
        pass

def VIP(s, Phi):
    (rin, rout, p) = Phi
    K = 20 # plan of depth of K.
    h, w = s.shape[0], s.shape[1]
    v = np.zeros((h,w, K+1))
    for k in range(K):
        for i in range(h):
            for j in range(w):
                v[i,j, k+1] = v[i,j,k]
                for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ip = i + di
                    jp = j + dj
                    if min(ip, jp) < 0 or ip >= h or jp >= w:
                        continue
                    nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
                    v[i,j,k+1] = max( v[i,j,k+1], nv)

    # plt.imshow(np.concatenate([v[:, :, -1], s[:, :, 0]]))

    return v


if __name__ == "__main__":
    env = MazeEnvironment(size=10)
    agent = VIAgent(env)
    agent = PlayWrapper(agent, env, autoplay=False)
    experiment = "experiments/q1_value_iteration"
    env = VideoMonitor(env, agent=agent, fps=100, continious_recording=True, agent_monitor_keys=('v', ), render_kwargs={'method_label': 'VI-K'})
    train(env, agent, experiment_name=experiment, num_episodes=10, max_steps=100)
    env.close()
