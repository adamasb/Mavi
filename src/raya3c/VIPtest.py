import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))
import gym
from mazeenv import maze_register
from a3c import A3CConfig
# import farmer
#from dtufarm import DTUCluster
from irlc import Agent, train, VideoMonitor
import numpy as np
from ray import tune
from ray.tune.logger import pretty_print
from raya3c.my_callback import MyCallbacks

from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer

import copy



# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
#import torchvision
from torch import nn
import matplotlib.pyplot as plt

import wandb


phi = SlimFC(3,3,activation_fn="relu")

# for i in range(10):
env = gym.make("MazeDeterministic_empty4-v0")

s = env.reset()

print(s[:,:,0])
print(s[:,:,1])
print(s[:,:,2], "\n")


test = torch.tensor(np.arange(16)).view(4,4)

def shift(v,dir):

    if (dir == 1):
        v_shift = torch.roll(v, -1, 1) # shift left
    elif (dir == -1):
        v_shift = torch.roll(v, -1, 0) # shift down

    return v_shift




shift(test,)



    # print("")
    # print(s[1,1,0])
    # input = (int(s[1,1,0]),int(s[1,1,1]),int(s[1,1,2]))
    # t = torch.tensor(input)
    # print(t)
    # #print(phi(s[1,1,0],s[1,1,0],s[1,1,0]))





# def VIP(Phi, K=20):#k=20 default, 
#     (rin, rout, p) = Phi
#     h, w = p.shape[0], p.shape[1]
#     v = torch.from_numpy(np.zeros((h,w, K+1)))
#     for k in range(K):
#         for i in range(h):
#             for j in range(w):
#                 v[i,j, k+1] = v[i,j,k]
#                 for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
#                     if di + i < 0 or dj + j < 0:
#                         continue
#                     if di + i >= h or dj + j >= w:
#                         continue
#                     ip = i + di
#                     jp = j + dj
#                     nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
#                     v[i,j,k+1] = max( v[i,j,k+1], nv)
#     return v



# def Phi(s):
#     rout = s[:, :, 2]
#     rin = s[:, :, 2] - 0.05 
#     p = 1 - s[:, :, 0] 
#     return (rin, rout, p)


# v = VIP(Phi(s))
# v = v[:,:,-1]

# plt.imshow(v)
# plt.show()