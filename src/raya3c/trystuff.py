
# import sys, os

# """export /zhome/8b/7/122640/Mavi/src/raya3c PYTHONPATH='$PYTHONPATH:/zhome/8b/7/122640/Mavi/src' """

# sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))
# import gym
# from mazeenv import maze_register
# from a3c import A3CConfig
# # import farmer
# #from dtufarm import DTUCluster
# from irlc import Agent, train, VideoMonitor
# import numpy as np
# from ray import tune
# from ray.tune.logger import pretty_print
# from raya3c.my_callback import MyCallbacks

# from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer

# import copy



# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# import torch
# from torch import nn
# import matplotlib.pyplot as plt



# s = torch.zeros((4,4,3))
# s[:,:,0][0,0] = 1 #create a wall
# s[:,:,1][3,0] = 1 #create a goal
# s[:,:,2][1,3] = 1 #create an agent

# Phi = SlimFC(3, 3, activation_fn = "relu") # input 3 output 3
# phi = []
# dim4 = []
# for ii in range(2):

#     phi.append(Phi(s.squeeze()))
#     phi_v = torch.stack(phi)
#     phi_vals = phi[ii].detach().numpy()
#     dim4.append(VP_nn(phi_vals))

# def VP_nn(phi_v,K=10):
    
#     h, w = phi[:, :, 0].shape[0], phi[:, :, 0].shape[1] #height and width of map
            
#     v = torch.zeros((h,w,K+1))  # wanna pad or roll over this, i think

#     for k in range(K): #number of "convolutions", or times the algorithm is applied
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
                    
#                     p_ij,rij_in,rij_out = phi[i,j,:]
#                     p_p,rp_in,rp_out = phi[ip,jp,:]

#                     nv = p_ij * v[ip, jp,k] + rp_in - rij_out
#                     v[i,j,k+1] = max( v[i,j,k+1], nv) 


#     dim4 = torch.unsqueeze(v[:,:,-1],dim=2)
#     return dim4


# def get_neighborhood(obs,dim4,a_index):

#         neighborhood = []
#         v_matrix, w_matrix, a_matrix,g_matrix = [], [], [], []

#         for ii in range(obs.shape[0]):
#             v_matrix.append(torch.nn.functional.pad(dim4[ii].squeeze(),(1,1,1,1))) #dont wanna override tensors
#             w_matrix.append(torch.nn.functional.pad(obs[ii][:,:,0],(1,1,1,1))) #change padding to 1's (or invert 1s and 0s all over)
#             a_matrix.append(torch.nn.functional.pad(obs[ii][:,:,1],(1,1,1,1)))
#             g_matrix.append(torch.nn.functional.pad(obs[ii][:,:,2],(1,1,1,1)))

#             rowNumber = a_index[ii][0] +1 #numpy array so okay to override
#             colNumber = a_index[ii][1] +1 #plus 1 to account for padding
#             v_result, w_result, a_result, g_result = [], [], [], []

#             for rowAdd in range(-1, 2):
#                 newRow = rowNumber + rowAdd
#                 if newRow >= 0 and newRow <= len(v_matrix[ii])-1:
#                     for colAdd in range(-1, 2):
#                         newCol = colNumber + colAdd
#                         if newCol >= 0 and newCol <= len(v_matrix[ii])-1:
#                             if newCol == colNumber and newRow == rowNumber:
#                                 pass# this is the agent location itself
#                                 #continue
#                             v_result.append(v_matrix[ii][newRow][newCol])                      
#                             w_result.append(w_matrix[ii][newRow][newCol])
#                             a_result.append(a_matrix[ii][newRow][newCol])
#                             g_result.append(g_matrix[ii][newRow][newCol])
            
#             neighborhood.append(torch.tensor([w_result, a_result, g_result, v_result]).flatten())
    
            
#         return torch.stack(neighborhood)


# dim4 = VP_nn(phi_vals)


# obs = torch.rand((32,4,4,3))
# a_index = torch.randint(0,4,(32,2))
# dim4 = torch.ones((32,4,4,1))


# nb = get_neighborhood(obs,dim4,a_index)


# from functorch import vmap

# phi_v = torch.tensor(phi_vals)

# batch_vp = vmap(VP_nn)

# batch_vp(phi_v)
