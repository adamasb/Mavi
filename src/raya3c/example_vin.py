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



# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
#import torchvision
from torch import nn
import matplotlib.pyplot as plt

import wandb

#dont know where this should be defined
torch.autograd.set_detect_anomaly(True)


vin_label = "vin_network_model"
# Kig paa: FullyConnectedNetwork som er den Model-klassen bruger per default.
# alt. copy-paste FullyConnectedNetwork-koden ind i denne klasse og modififer gradvist (check at den virker paa simple gridworld)
class VINNetwork(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # super().__init__(obs_space, action_space, num_outputs, model_config, name)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        self.num_outputs = 5 #int(np.product(self.obs_space.shape))
        self._last_batch_size = None
        



        #not being used right now
        self.nn = SlimFC(3*3*4, self.num_outputs)#  generalize input dimensions


        #is this really all it is?
        # consider using SlimConv2d instead (from misc.py as well)
        self.Phi = SlimFC(3, 3, activation_fn = "relu") # input 3 output 3
        



        #lets try to remove all debug_vin stuff
        #if model_config['debug_vin']: #changed from model_conf, think that was a typo
        #    self.debug_vin = model_config['debug_vin']

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )


        layers = []
                         #can hardcode the value because it will always be 3x3*4
        prev_layer_size = 36# int(np.product(obs_space.shape)*4/3) #hacky way to get the right input size
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    #activation_fn=activation,
                )
            )
            prev_layer_size = size

        if len(hiddens) > 0:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=hiddens[-1],
                    initializer=normc_initializer(1.0),
                    #activation_fn=activation,
                )
            )
            prev_layer_size = hiddens[-1]
        if num_outputs:
            self._logits = SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                #activation_fn=None,
            )
        else:
            self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                -1
            ]

        self._hidden_layers = torch.nn.Sequential(*layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            #activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None


    """What the heck is this?"""
    # def value_function_for_env_state(self, state):
    #     """WHAR IS THIS SUPPOSED TO BE?"""
    #     stats = {}
    #     # value_function_for_env_state should include p,rin,rout,v,
    #     # self.value_function_for_env_state = {}
    #     stats["p"] = np.ones((4,4))
    #     stats["rin"] = np.ones((4,4))
    #     stats["rout"] = np.ones((4,4))
    #     stats["v"] = np.ones((4,4))
    #     # value_function_for_env_state
    #     # self.value_function_for_env_state["p"] = phi_vals[0,0,0]
    #     # self.value_function_for_env_state["rin"] = phi_vals[0,0,1]
    #     # self.value_function_for_env_state["rout"] = phi_vals[0,0,2]
    #     # self.value_function_for_env_state["v"] = dim4[0].detach().numpy()

    #     return stats



    def VP_simple(self,s):
        #s[:, :, 0] = walls, s[:, :, 1] = agent, s[:, :, 2] = goal
        rout = s[:, :, 2]
        rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
        p = 1 - s[:, :, 0] 
        K=20
        h, w = p.shape[0], p.shape[1]
        #we get issues with wandb showing the v plot when using tensors instead of np array
        v = torch.zeros((h,w, K+1)) 
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
                        
        s[:,:,0] = p
        dim4 = v[:,:,-1]
        dim4 = np.expand_dims(dim4,axis=2) 
        vp = np.concatenate((s,dim4),axis=2)
        vp = torch.from_numpy(vp.flatten())
        vp = vp.type(torch.FloatTensor)
    
        return vp


    # def phi(self, w,a,g): #could use funciton to clean it up a bit
    #     #w,a,g = s[i, j, 0], s[i, j, 1], s[i, j, 2] #walls, agent, goal
    #     rin,rout,p = self.Phi(w,a,g)
    #     return rin,rout,p

    def VP_nn(self,s,phi,K=10):
        
        #can also be defined based on phi
        h, w = s[:, :, 0].shape[0], s[:, :, 0].shape[1] #height and width of map
        
        #vp = torch.zeros((h,w,4)) #maybe i dont need to instantiate this
        
        v = torch.zeros((h,w,K+1))  # wanna pad or roll over this, i think

        for k in range(K): #number of "convolutions", or times the algorithm is applied
            for i in range(h):
                for j in range(w):
                    #do a version of this without assignments
                    v[i,j, k+1] = v[i,j,k]
                    for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if di + i < 0 or dj + j < 0:
                            continue
                        if di + i >= h or dj + j >= w:
                            continue  
                        
                        
                        ip = i + di
                        jp = j + dj
                        

                        # fixes issue of overriding tensor with gradients
                        # p_ij,rij_in,rij_out = float(phi[i,j,0]),float(phi[i,j,1]),float(phi[i,j,2])
                        # p_p,rp_in,rp_out = float(phi[ip,jp,0]), float(phi[ip,jp,1]), float(phi[ip,jp,2])

                        p_ij,rij_in,rij_out = phi[i,j,:]
                        p_p,rp_in,rp_out = phi[ip,jp,:]

                        #nv should be calculated with a roll from the previous v

                        nv = p_ij * v[ip, jp,k] + rp_in - rij_out
                        v[i,j,k+1] = max( v[i,j,k+1], nv) 

                        pass

        s[:,:,0]= 1 - s[:, :, 0] #walls
        dim4 = torch.unsqueeze(v[:,:,-1],dim=2)
        vp = torch.flatten(torch.cat((s,dim4),dim=2)) #concatenating the 3d tensor with the 2d tensor, and flattening it
        return dim4
        return vp


    
    def get_neighborhood(self, obs,dim4,a_index):

        neighborhood = []
        v_matrix, w_matrix, a_matrix,g_matrix = [], [], [], []

        for ii in range(obs.shape[0]):
            v_matrix.append(torch.nn.functional.pad(dim4[ii].squeeze(),(1,1,1,1))) #dont wanna override tensors
            w_matrix.append(torch.nn.functional.pad(obs[ii][:,:,0],(1,1,1,1))) #change padding to 1's (or invert 1s and 0s all over)
            a_matrix.append(torch.nn.functional.pad(obs[ii][:,:,1],(1,1,1,1)))
            g_matrix.append(torch.nn.functional.pad(obs[ii][:,:,2],(1,1,1,1)))

            rowNumber = a_index[ii][0] +1 #numpy array so okay to override
            colNumber = a_index[ii][1] +1 #plus 1 to account for padding
            v_result, w_result, a_result, g_result = [], [], [], []

            for rowAdd in range(-1, 2):
                newRow = rowNumber + rowAdd
                if newRow >= 0 and newRow <= len(v_matrix[ii])-1:
                    for colAdd in range(-1, 2):
                        newCol = colNumber + colAdd
                        if newCol >= 0 and newCol <= len(v_matrix[ii])-1:
                            if newCol == colNumber and newRow == rowNumber:
                                pass# this is the agent location itself
                                #continue
                            v_result.append(v_matrix[ii][newRow][newCol])                      
                            w_result.append(w_matrix[ii][newRow][newCol])
                            a_result.append(a_matrix[ii][newRow][newCol])
                            g_result.append(g_matrix[ii][newRow][newCol])
            
            neighborhood.append(torch.tensor([w_result, a_result, g_result, v_result]).flatten())
    
            
        return torch.stack(neighborhood)


    def forward(self, input_dict, state, seq_lens): #dont think this is currently being used
        obs = input_dict["obs_flat"]


        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        


        """ Consider turning this into a tensor, and using mapv to parrelelize it"""
        #vp = torch.zeros((obs.shape[0],int(obs.shape[1]/3*4))) # generalize dimensions
        vp = [] # np.array([])




        phi=[]
        dim4 = []
        a_index = []

        for ii in range(obs.shape[0]):

            phi.append(self.Phi(input_dict["obs"][ii].squeeze())) #only use the first obs, as it is the same for all (for now)
            # fixes issue of overriding tensor with gradients
            phi_vals = phi[ii].detach().numpy() #convert to np array to remove gradients
    
            
            
            #do this with append instead of this assignment
            # vp[ii] = self.VP_nn(obs[ii].reshape((4,4,3)),phi_vals)
            #vp.append(self.VP_nn(obs[ii].reshape((4,4,3)),phi_vals))
            
            width = len(input_dict["obs"][0][:,:,0])
            #generalize dimensions
            dim4.append(self.VP_nn(obs[ii].reshape((width,width,3)),phi_vals))
            

            """ removed these lines because using the get_neighborhood function instead"""
            s = input_dict["obs"][ii]
            s[:,:,0]= 1 - s[:, :, 0] #walls
            vp.append(torch.flatten(torch.cat((s,dim4[-1]),dim=2))) #concatenating the 3d tensor with the 2d tensor, and flattening it
            # np.append(vp,self.VP_nn(obs[ii].reshape((4,4,3)),phi_vals))
            #vp[ii] = self.VP_nn(copy.deepcopy(obs)[ii].reshape((4,4,3)))

            #self.VP_nn(copy.deepcopy(obs)[ii].reshape((4,4,3)))
            if obs[ii].any() !=0:
                a_index.append(input_dict["obs"][ii][:,:,1].nonzero().detach().numpy()[0]) #get the index of the agent
                # if len(a_index) > 1:
                #     print('more than one batch')
            else:
                a_index.append([1,1]) #this doesnt really matter
            
        #    V_np = []
        #    assert( (V_np - V_torch.numpy())< 1e-8 )

            pass




        


        #global plot_this
        #plot_this  = dim4[-1].detach().numpy()
        # if plot_this.any() != 0:
        #     #wandb.init()
        #     #images = wandb.Image(plot_this, caption="Top: Output, Bottom: Input")
        #     #wandb.log({"image of v": images})
        #     plt.imshow(plot_this)
        #     plt.show()


    
        """ consider passing just the values of the 3x3 neighbourhood around the agent into the network"""
                
        # should be a list of tensors of size (5*3) (maybe * batch size)

        # print(input_dict["obs"].shape)
        # print(dim4.shape)
        #print(type(a_index))
        #neighborhood = self.get_neighborhood(input_dict["obs"],dim4,a_index)

        # print(self.get_neighborhood(input_dict["obs"],dim4,a_index))
        # print("a")
        #neighborhood[0][:9].reshape(3,3)

        
        # if obs.shape[0] > 1:
        #     if obs.any() !=0:
        #         #print(neighbourhood[-1])
        #         pass
        
        """has shape (batch size, 3*3*4)"""
        # vp = self.get_neighborhood(input_dict["obs"],dim4,a_index)
        
        
        self._last_flat_in = self.get_neighborhood(input_dict["obs"],dim4,a_index)
        # self._last_flat_in = torch.stack(vp)
        
        #.reshape(vp.shape[0], -1) 
        #mat1 and mat2 shapes cannot be multiplied (32x64 and 48x24)
        # get i,j locations for agent as I, J


        # self.value_cache = tensor of size B x 1 corresponding to v[b, I, J], b = [0, 1, 2, 3, ..., B]

        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        
        #logits = self.nn(vp)
        return logits, state #from fcnet, state is []
        
    
    def value_function(self): #dont think this is currently being used
        #consider pass value function through a neural network
        # return self.value_cache
        return self._value_branch(self._features).squeeze(1) #torch.Size([32])
        

    #pi from agent.py
    def pi(self, s, k=None): #we never enter this (except with the irlc-visualise stuff)
        # return self.env.action_space.sample() #!s
        return self.obs_space.action_space.sample()


#Tue's "Scrap file" code:
ModelCatalog.register_custom_model(vin_label, VINNetwork)
def my_experiment(a):
    
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    mconf = dict(custom_model=vin_label, use_lstm=False)
    # mconf = {}
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)

    config = config.framework('torch')


    config.min_train_timesteps_per_iteration = 200
    config.min_sample_timesteps_per_iteration = 200
   

    env_name = "MazeDeterministic_empty4-v0"
    #env_name = 'Maze_empty4-v0' # i dont have this one
    config = config.callbacks(MyCallbacks)

    # Set up alternative model (gridworld).
    # config = config.model(custom_model="my_torch_model", use_lstm=False)
    #print(config.to_dict())
    config.model['fcnet_hiddens'] = [24, 24] #why 24, just arbitrary?
    # config.model['debug_vin'] = True

    config.model['custom_model_config'] = {}
    config.model['custom_model_config']['env_name'] = env_name
    from ray.tune.registry import _global_registry # EXAMPLE: Creating an environment.
    env =     _global_registry.get("env_creator", env_name)
    hw = env(env_config=config.env_config).game.height, env(env_config=config.env_config).game.width
    # env(env_config=config.env_config).observation_space
    config.model['custom_model_config']['env_shape'] = hw
    config.evaluation_interval = 1

    def my_eval_fun(a3c, worker_set, **kwargs):
        # print("Evaluating...", args, kwargs)
        # s = args[1].local_worker().sample()
        # a = 234]
        print("Running custom evaluation function...")
        def my_pol_fun(a3policy, *args, a3c=None, **kwargs):
            print(args, kwargs)
            # cb = a3policy.callbacks
            a3c.callbacks.evaluation_call(a3policy ) #my evaluation_call calls a funciton that needs a variable to be defined
            # value_function_for_env_state should include p,rin,rout,v,

            # a3policy.callbacks
            # a = 234
            return {}

        #this gives me an error: functools is not defined
        import functools #maybe this is it? -> causes all kinds of weird problems, leave it out for now
        #worker_set.foreach_policy(functools.partial(my_pol_fun, a3c=a3c) )
        
        return dict(my_eval_metric=123)

    config.custom_evaluation_function = my_eval_fun
    # config['evaluation_interval'] = 1


    # config.model['debug_vin'] = True
    # env = gym.make("MazeDeterministic_empty4-v0")


    # if False:
    #     ray.init()
    #     # from raya3c.a3c import A3C
    #     # trainer = A3C(env=env_name, config=config.to_dict())
    #     while True:
    #         print(trainer.train())
    # else:

    trainer = config.build(env="MazeDeterministic_empty4-v0")
    for t in range(600): #Seems to converge to 2.5 after 500-600 iterations
        print("Main training step", t)
        result = trainer.train()
        rewards = result['hist_stats']['episode_reward']
        try:
            int(result['episode_reward_mean'])
        except:
            print("whoops")#just want to catch when the result is nan

        print("training epoch", t, len(rewards), max(rewards) if len(rewards) > 0 else -1, result['episode_reward_mean'], result["episode_len_mean"])
        
        
        
        

    # config.save

    #plt.imshow(v)
    #plt.show()

    # images = wandb.Image(plot_this, caption="Top: Output, Bottom: Input")
    # wandb.log({"image of v": images})



if __name__ == "__main__":
    global plot_this

    res = []
    DISABLE = True
    
    my_experiment(1)
   
    print("Job done")
    sys.exit()

