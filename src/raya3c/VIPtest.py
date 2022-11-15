import gym
from mazeenv import maze_register
import matplotlib.pyplot as plt
import numpy as np
import torch



env = gym.make("MazeDeterministic_empty4-v0")
s = env.reset()


def VIP(Phi, K=20):#k=20 default, 
    (rin, rout, p) = Phi
    h, w = p.shape[0], p.shape[1]
    v = torch.from_numpy(np.zeros((h,w, K+1)))
    for k in range(K):
        for i in range(h):
            for j in range(w):
                v[i,j, k+1] = v[i,j,k]
                for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if di + i < 0 or dj + j < 0:
                        continue
                    if di + i >= h or dj + j >= w:
                        continue
                    ip = i + di
                    jp = j + dj
                    nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
                    v[i,j,k+1] = max( v[i,j,k+1], nv)
    return v



def Phi(s):
    rout = s[:, :, 2]
    rin = s[:, :, 2] - 0.05 
    p = 1 - s[:, :, 0] 
    return (rin, rout, p)


v = VIP(Phi(s))
v = v[:,:,-1]

plt.imshow(v)
plt.show()