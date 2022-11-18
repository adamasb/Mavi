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
        
        self.nn = SlimFC(4*4*4, self.num_outputs)# used when we take state @ vp
        #not at all used, yet
        self.Phi = SlimFC(3, 3) # input 3 output 3

        #lets try to remove all debug_vin stuff
        #if model_config['debug_vin']: #changed from model_conf, think that was a typo
        #    self.debug_vin = model_config['debug_vin']

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )


        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
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


        """ Everythin above (until comment) is stolen straight from fcnet"""

    def VP(self,s):
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
                        
        s[:,:,0],s[:,:,2] = p,v[:,:,-1]
        # print("s",s.shape)
        dim4 = v[:,:,-1]
        #print(dim4.shape)
        dim4 = np.expand_dims(dim4,axis=2) 
        vp = np.concatenate((s,dim4),axis=2)
        # print("vp",vp.shape)
        vp = torch.from_numpy(vp.flatten())
        # print("vp",vp.shape)
        vp = vp.type(torch.FloatTensor)
        #vp =1

        return vp


    def forward(self, input_dict, state, seq_lens): #dont think this is currently being used
        obs = input_dict["obs_flat"]

        # if obs.any() != 0:
        #     #print(obs[0].reshape((4,4,3))[:,:,-1])
        #     if (obs[0] != torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])).all() == torch.tensor(False):
        #         if (obs[0] != torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.])).all() == torch.tensor(False):
        #             if (obs[0] != torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])).all():
        #                 if (obs[0] != torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.])).all():
        #                     #print(obs[0].reshape((4,4,3))[:,:,-1])
        #                     print("they dont match")


        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        
        #vp = torch.tensor(np.zeros((obs.shape)))
        vp = torch.zeros((obs.shape[0],int(obs.shape[1]/3*4)))
        for ii in range(obs.shape[0]):

            vp[ii] = self.VP(copy.deepcopy(obs)[ii].reshape((4,4,3)))
        #self.VP(torch.zeros((4,4,3)))
        #mat1 and mat2 shapes cannot be multiplied (32x48 and 64x5)
        
            #vp = self.VP(torch.zeros((4,4,3)))
            #HOW THE FRICK DOES THIS BREAK EVERYTHING!?
        
              
        #self._last_flat_in = vp     

        #plt.imshow(vp)
        #plt.show()
                    
        #    V_np = []
        #    assert( (V_np - V_torch.numpy())< 1e-8 )


        #stolen right from FCNet        
        

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        #logits = self._logits(self._features) if self._logits else self._features
        
        logits = self.nn(vp)
   
        return logits, state #from fcnet
        
    
    def value_function(self): #dont think this is currently being used
        #pass value function through a neural network

        return self._value_branch(self._features).squeeze(1) #
        

    #pi from agent.py
    def pi(self, s, k=None): #we never enter this (except with the irlc-visualise stuff i think)
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

    #figure out where these should be defined


    #might be here where we break the wandb??

    #THIS IS WHERE MY ISSUE IS!!
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
    # config.model['saf'] = True
    # config.model['asdfasdf'] = 234

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
            a3c.callbacks.evaluation_call(a3policy )


            # a3policy.callbacks
            # a = 234
            return {}

        #this gives me an error: functools is not defined
        #import functools #maybe this is it? -> causes all kinds of weird problems, leave it out for now
        #worker_set.foreach_policy(functools.partial(my_pol_fun, a3c=a3c) )

        return dict(my_eval_metric=123)

    config.custom_evaluation_function = my_eval_fun
    # config['evaluation_interval'] = 1


    # config.model['debug_vin'] = True
    # env = gym.make("MazeDeterministic_empty4-v0")


    if False:
        ray.init()
        # from raya3c.a3c import A3C
        # trainer = A3C(env=env_name, config=config.to_dict())
        while True:
            print(trainer.train())
    else:

        trainer = config.build(env="MazeDeterministic_empty4-v0")
        for t in range(400): #200 is not enough for good performance
            print("Main training step", t)
            result = trainer.train()
            rewards = result['hist_stats']['episode_reward']
            try:
                int(result['episode_reward_mean'])
            except:
                print("whoops")#just want to catch when the result is nan
                
            print("training epoch", t, len(rewards), max(rewards) if len(rewards) > 0 else -1, result['episode_reward_mean'])

    # config.save

    #plt.imshow(v)
    #plt.show()

    #images = wandb.Image(v, caption="Top: Output, Bottom: Input")
    #wandb.log({"image of v": images})



if __name__ == "__main__":
   
    res = []
    DISABLE = True
    
    my_experiment(1)
   
    print("Job done")
    sys.exit()

