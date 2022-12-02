
import sys, os

"""export /zhome/8b/7/122640/Mavi/src/raya3c PYTHONPATH='$PYTHONPATH:/zhome/8b/7/122640/Mavi/src' """

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



from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn
import matplotlib.pyplot as plt



s = torch.zeros((4,4,3))
s[:,:,0][0,0] = 1 #create a wall
s[:,:,1][3,0] = 1 #create a goal
s[:,:,2][1,3] = 1 #create an agent

Phi = SlimFC(3, 3, activation_fn = "relu") # input 3 output 3
phi = []
dim4 = []



obs = torch.rand((3,4,4,3))
a_index = torch.randint(0,4,(32,2))
dim4 = torch.ones((32,4,4,1))






def VP_batch(self, phi, K=10):
    p,rin,rout = phi[:,:,:,0],phi[:,:,:,1],phi[:,:,:,2]

    v = torch.zeros((p.shape[0],p.shape[1],p.shape[2]))  

    r_in_pad = torch.nn.functional.pad(rin,  (1,1) + (1,1), 'constant', 0)

    for tt in range(K):
        vm = []
        if tt > 0:
            vm.append(v)

        v_pad = torch.nn.functional.pad(v,  (1,1) + (1,1), 'constant', 0)
        for ax, shift in [ (1, -1), (1, 1), (2, -1), (2, 1)]:
            # torch.pad(v, )
            v_shift = torch.roll(v_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
            r_shift = torch.roll(r_in_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
            vm.append(p[:,:,:] * v_shift + r_shift - rout[:,:,:] )

        v, _ = torch.stack(vm).max(axis=0)

    return v




def VP_new(self,s,phi,K=10):
    p,rin,rout = torch.unsqueeze(phi[:,:,0],dim=2), torch.unsqueeze(phi[:,:,1],dim=2), torch.unsqueeze(phi[:,:,2],dim=2)
    v = torch.zeros((p.shape[0],p.shape[1],1))  # wanna pad or roll over this, i think

    r_in_pad = torch.nn.functional.pad(rin,  (1,1) + (1,1), 'constant', 0)

    for tt in range(K):
        vm = []
        if tt > 0:
            vm.append(v)

        v_pad = torch.nn.functional.pad(v,  (1,1) + (1,1), 'constant', 0)
        for ax, shift in [ (1, -1), (1, 1), (2, -1), (2, 1)]:
            # torch.pad(v, )
            v_shift = torch.roll(v_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
            r_shift = torch.roll(r_in_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
            vm.append(p[:,:,:] * v_shift + r_shift - rout[:,:,:] )
        self.debug_vin = False #debug flag

        v, _ = torch.stack(vm).max(axis=0)
        if self.debug_vin:
            # diff = np.abs( V_db[:,:,:,tt+1] - v.detach().numpy() ).sum() #only used for comparing with the "bad VP"
            diff = np.abs( self.VP_simple(s) - v.detach().numpy() ).sum() #this isnt actually worth anything yet (fix)
            if diff > 1e-4:
                print(tt, "large diff", diff)
                assert False    
    return v




