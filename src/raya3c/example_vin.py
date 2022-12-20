import sys, os

"""export PYTHONPATH='$PYTHONPATH:/zhome/8b/7/122640/Mavi/src' """

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

from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer, SlimConv2d

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
        
        # consider using SlimConv2d instead (from misc.py as well)
        self.Phi = SlimFC(3, 3, activation_fn = "relu") # input 3 output 3


        #create simple NN for speaker agent - what should be input and output size?
        self.speaker = SlimFC(3, 3, activation_fn = "relu")




        # self.Phi = SlimConv2d(3, 3, kernel, stride, activation_fn = "relu") #define kernel and stride


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

    def VP_simple(self,s, phi = None, K=10):
        #s[:, :, 0] = walls, s[:, :, 1] = agent, s[:, :, 2] = goal
        phi2 = copy.deepcopy(phi)
        try:
            p,rin,rout = phi2(s)
            p = p.detach().numpy()
            rin = rin.detach().numpy()
            rout = rout.detach().numpy()
        except:
            rout = s[:, :, 2]
            rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
            p = 1 - s[:, :, 0]
        
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
                        
        # s[:,:,0] = p
        dim4 = v[:,:,-1]
        # dim4 = np.expand_dims(dim4,axis=2) 
        # vp = np.concatenate((s,dim4),axis=2)
        # vp = torch.from_numpy(vp.flatten())
        # vp = vp.type(torch.FloatTensor)
    
        return dim4


    def VP_nn(self,s,phi,K=10):
        
        #can also be defined based on phi
        h, w = phi[:, :, 0].shape[0], phi[:, :, 0].shape[1] #height and width of map


        #consider initializing this as a minimum of something, so that its not just 0s. Force it to calculate some gradients        
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
                        
                        p_ij,rij_in,rij_out = phi[i,j,:]
                        p_p,rp_in,rp_out = phi[ip,jp,:]

                        #nv should be calculated with a roll from the previous v

                        nv = p_ij * v[ip, jp,k] + rp_in - rij_out
                        v[i,j,k+1] = max( v[i,j,k+1], nv) 

                        pass

        # s[:,:,0]= 1 - s[:, :, 0] #walls
        dim4 = torch.unsqueeze(v[:,:,-1],dim=2)
        # vp = torch.flatten(torch.cat((s,dim4),dim=2)) #concatenating the 3d tensor with the 2d tensor, and flattening it
        return dim4
        # return vp

    def VP_new(self,s,phi,K=10):
        # p,rin,rout = phi[:,:,0],phi[:,:,1],phi[:,:,2]
        p,rin,rout = torch.unsqueeze(phi[:,:,0],dim=2), torch.unsqueeze(phi[:,:,1],dim=2), torch.unsqueeze(phi[:,:,2],dim=2)

        # p2,rin2,rout2 = torch.cat((p,p),dim=2), torch.cat((rin,rin),dim=2), torch.cat((rout,rout),dim=2)

        v = torch.zeros((p.shape[0],p.shape[1],1))  # wanna pad or roll over this, i think

        r_in_pad = torch.nn.functional.pad(rin,  (1,1) + (1,1), 'constant', 0)

        """ Tues code starts here"""
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
                #RuntimeError: The size of tensor a (4) must match the size of tensor b (32) at non-singleton dimension 1


            self.debug_vin = False #debug flag

            v, _ = torch.stack(vm).max(axis=0)

            # if self.debug_vin: #never really got it working
            #     vp = self.VP_simple(s, phi, K = tt+1)
            #     # diff = np.abs( V_db[:,:,:,tt+1] - v.detach().numpy() ).sum() #only used for comparing with the "bad VP"
            #     diff = np.abs(vp - v.detach().numpy() ).sum() #this isnt actually worth anything yet (fix)
            #     print("==================V=====================")
            #     print(v)
            #     print("==================VP=====================")
            #     print(vp)
            #     if diff > 1e-4:
            #         print(tt, "large diff", diff)
            #         assert False

        # II = oa.nonzero().numpy()
        # v_pad = torch.nn.functional.pad(v, (1, 1) + (1, 1), 'constant', 0)
        # o_pad = torch.nn.functional.pad(o, (0,0) + (1, 1) + (1, 1), 'constant', 0)
        # self.v_raw = v
        """" tues code^^"""
        
        return v




    def VP_batch(self, phi, K=10):
        # p,rin,rout = phi[:,:,0],phi[:,:,1],phi[:,:,2]
        #p,rin,rout = torch.unsqueeze(phi[:,:,0],dim=2), torch.unsqueeze(phi[:,:,1],dim=2), torch.unsqueeze(phi[:,:,2],dim=2)
        p,rin,rout = phi[:,:,:,0],phi[:,:,:,1],phi[:,:,:,2]

        # all have shape ([32,4,3])

        # p2,rin2,rout2 = torch.cat((p,p),dim=2), torch.cat((rin,rin),dim=2), torch.cat((rout,rout),dim=2)

        v = torch.zeros((p.shape[0],p.shape[1],p.shape[2]))  

        r_in_pad = torch.nn.functional.pad(rin,  (1,1) + (1,1), 'constant', 0)

        for tt in range(K):
            vm = []
            if tt > 0: # i think this might be the reason why its all black?, maybe force positive values in another way
                vm.append(v)

            v_pad = torch.nn.functional.pad(v,  (1,1) + (1,1), 'constant', 0)
            for ax, shift in [ (1, -1), (1, 1), (2, -1), (2, 1)]:
                # torch.pad(v, )
                v_shift = torch.roll(v_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
                r_shift = torch.roll(r_in_pad, dims=ax-0, shifts=shift)[:,1:-1, 1:-1]
                vm.append(p[:,:,:] * v_shift + r_shift - rout[:,:,:] )

            v, _ = torch.stack(vm).max(axis=0)

            # test = self.VP_simple(np.zeros((4,4,3), phi, K = tt+1), phi, K=20) #
            # diff = np.abs( test - v.detach().numpy() ).sum()
            # if diff > 1e-4:
            #     print(tt, "large diff", diff)
            #     assert False




            
        return v
        #  # Create a tensor with size (32, 4, 4)
        # tensor = torch.randn(32, 4, 4)

        # # Split the tensor into a list of tensors with size (4, 4)
        # tensors = tensor.split(4, dim=0)

        # # Apply the function to each tensor in the list using PyTorch's map function
        # results = torch.stack(list(map(VP_new, tensors)))

       
    
    def get_neighborhood(self, obs,dim4,a_index):

        neighborhood = []
        v_matrix, w_matrix, a_matrix,g_matrix = [], [], [], []

        for ii in range(obs.shape[0]):
            v_matrix.append(torch.nn.functional.pad(dim4[ii].squeeze(),(1,1,1,1))) #dont wanna override tensors
            w_matrix.append(torch.nn.functional.pad(obs[ii][:,:,0],(1,1,1,1), 'constant', 1)) #pad walls with ones
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
                           
                            v_result.append(v_matrix[ii][newRow][newCol])                      
                            w_result.append(w_matrix[ii][newRow][newCol])
                            a_result.append(a_matrix[ii][newRow][newCol])
                            g_result.append(g_matrix[ii][newRow][newCol])
            
            neighborhood.append(torch.tensor([w_result, a_result, g_result, v_result]).flatten())
        return torch.stack(neighborhood)


    def forward(self, input_dict,state,seq_lens):
        try:
            obs = input_dict["obs"]
        except:
            obs = input_dict #this is for when using the get_env_state_value_function method thing        


        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        
        
        
        """ Consider turning this into a tensor, and using mapv to parrelelize it"""
        phi=[]
        dim4 = []
        a_index = []

        v_c = []
        v_new = []
        v_test = []
       
        try:
            v_batch = self.VP_batch(self.Phi(obs))

        except Exception as e:
            print(e)

        for ii in range(obs.shape[0]):

            # try:
                # phi.append(self.Phi(obs[ii].squeeze())) #only use the first obs, as it is the same for all (for now)
            # fixes issue of overriding tensor with gradients


            # Not used after implementing shift-vprop-function
            #probably dont want to detach this
            # phi_vals = phi[ii].detach().numpy() #convert to np array to remove gradients
                # width = len(obs[0][:,:,0])
            # dim4.append(self.VP_nn(obs[ii].reshape((width,width,3)),phi_vals))
            # print(obs.shape)
            
                # dim4.append(self.VP_new(obs[ii].reshape((width,width,3)),phi[ii],K = 5)) #trying without the detach.numpu
                # if (obs.shape[0] > 1):
                #     test = self.VP_simple(obs[ii].reshape((width,width,3)),phi[ii], K = 5) #test the simple vprop function
                    
                #     diff = np.abs(test - dim4[-1].detach().numpy() ).sum() #this isnt actually worth anything yet (fix)
                #     if diff > 1e-4:
                #         # print("large diff", diff)
                #         assert False
            # except:
            #     pass                
            # v_new.append(self.VP_new(obs[-1],phi[-1]))
            
            if obs[ii].any() !=0:
                a_index.append(obs[ii][:,:,1].nonzero().detach().numpy()[0]) #get the index of the agent
            else:
                a_index.append([0,0]) #this doesnt really matter
            
            # v_c.append(dim4[ii][a_index[ii][0],a_index[ii][1]])
            # v_c.append(v_new[ii][a_index[ii][0],a_index[ii][1]])
            
            # v_c.append(v_new[ii][a_index[ii][0],a_index[ii][1]])
            v_test.append(v_batch[ii][a_index[ii][0],a_index[ii][1]])


            self.v_raw = v_batch[-1]

            # assert(v_c[ii] == torch.stack(v_test).any())

        #     if obs.any()!= 0:
        #         if ii > 1:
        #             print(ii)
        # # if v_c[ii] != 0:
        #     if ii == 31:
        #         # assert False
        #         print(torch.stack(v_new).squeeze())
        #         print(v_batch)
        #         print(v_c[ii] == v_test[ii])
        #         assert(torch.stack(v_new).squeeze() == v_batch).all()
        #         assert(v_c[ii] == v_test[ii])
                
        #             # assert(v_c[ii] == v_test[ii])
        # #    V_np = []
        # #    assert( (V_np - V_torch.numpy())< 1e-8 )
        
            """loop ends here"""



         # try:
        #     diff2 = np.abs(torch.sum(v_batch).detach().numpy() - torch.sum(dim4.detach().numpy()) ).sum()
        #     # print(diff2)
        #     if diff2 > 1e-4:
        #         print("large diff", diff2)
        #         assert False
        # except:
        #     pass
        # self.value_cache = tensor of size B x 1 corresponding to v[b, I, J], b = [0, 1, 2, 3, ..., B]
        # self.value_cache = torch.stack(v_c) # if i squeeze i get an error
        self.value_cache = torch.unsqueeze(torch.stack(v_test),dim=1)

        # self._last_flat_in = self.get_neighborhood(input_dict["obs"],v_new,a_index) 
        self._last_flat_in = self.get_neighborhood(input_dict["obs"],v_batch,a_index) 



        # self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        
        #logits = self.nn(vp)
        return logits, []# state #from fcnet, state is []
        
    
    def value_function(self): #dont think this is currently being used
        #consider pass value function through a neural network
        return self.value_cache.squeeze(1) 
        # return self._value_branch(self._features).squeeze(1) #torch.Size([32])
        

  
    def value_function_for_env_state(self, env_state):
        
        obs = env_state[np.newaxis, :]
        info_dict = torch.tensor(env_state)# {"obs": torch.tensor(env_state)}
        #self.forward(info_dict, [],[])
        
        phi_vals = self.Phi(info_dict.float()).detach().numpy() #convert to np array to remove gradients

        # v.append(self.VP_nn(obs[0].reshape((info_dict[:,:,0].shape[0],info_dict[:,:,0].shape[1],3)),phi_vals))
        # v = self.VP_nn(obs[0].reshape((info_dict[:,:,0].shape[0],info_dict[:,:,0].shape[1],3)),phi_vals).detach().numpy() 

        #alternatively define v like this
        v = self.v_raw.detach().numpy()
        
        stats = {"v": v.squeeze(), "phi": phi_vals, "p": phi_vals[:,:,0], "rin": phi_vals[:,:,1], "rout": phi_vals[:,:,2]}
        return stats





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
        worker_set.foreach_policy(functools.partial(my_pol_fun, a3c=a3c) )
        
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
    
    # """ Multi agent stuff - mostly inspired by tictactoe example"""
    # multi_agent = False
    
    # if multi_agent:
    #     trained_policy = trainer + '_policy' #this is exactly from tictactoe
    #     speaker_policy = trainer + '_speaker_policy' #this is just a placeholder for now
        
    #     def policy_mapping_fn(agent_id): #this is a function that maps agent id to policy id
    #         mapping = {env.listener: trained_policy,
    #                 env.speaker: speaker_policy}
    #         return mapping[agent_id]

    #     # add a multiagent config
    #     config = config.multi_agent(
    #         { # This is placeholder code

    #             "policies_to_train": [trained_policy, speaker_policy], #not sure if we need a policy for the speaker, other than repeat goal location
    #             "policies": {
    #                 "policy_0": (None, env.obs_space, env.act_space, mconf), #seperate policy for speaker and listenener
    #                 "policy_1": (None, env.obs_space, env.act_space, mconf),
    #             },
    #             "policy_mapping_fn": policy_mapping_fn # lambda agent_id: "policy_0", #function that maps listener id to listener policy etc.
    #         }
            #Alternatively something like this:
            #    {
    #             "policies": set(env.agents),
    #             "policy_mapping_fn": (
    #                 lambda agent_id, episode, worker, **kwargs: agent_id
    #           }
    #      )

    for t in range(3000): #Seems to converge to 2.5 after 500-600 iterations
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

