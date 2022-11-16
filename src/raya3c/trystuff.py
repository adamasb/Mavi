# import sys, os
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



# # The custom model that will be wrapped by an LSTM.
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# import torch
# #import torchvision
# from torch import nn
# import matplotlib.pyplot as plt

# import wandb
# import time

# #can be used to create a custom policy or custom trainer
# """CustomPolicy = A3CTorchPolicy.with_updates(
#     name="MyCustomA3CTorchPolicy",
#     loss_fn=some_custom_loss_fn)

# CustomTrainer = A3C.with_updates(
#     default_policy=CustomPolicy)"""


# vin_label = "vin_network_model"
# # Kig paa: FullyConnectedNetwork som er den Model-klassen bruger per default.
# # alt. copy-paste FullyConnectedNetwork-koden ind i denne klasse og modififer gradvist (check at den virker paa simple gridworld)
# class VINNetwork(TorchModelV2, torch.nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         # super().__init__(obs_space, action_space, num_outputs, model_config, name)
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         torch.nn.Module.__init__(self)
#         self.num_outputs = 5 #int(np.product(self.obs_space.shape))
#         self._last_batch_size = None
        
#         self.n_forwards = 0
#         self.t_forwards = 0
        
#         self.t_total = 0
#         self.tc = time.time()
#         self.nn = SlimFC(3*4*4, self.num_outputs)#input = 3n^2 

#         #self.nn = SlimFC(4*4*4, self.num_outputs)# used when we take state @ vp
#         # perhaps change this to 4*4*4 so we can use s @ vp input

#         #linear(): argument 'input' (position 1) must be Tensor, not NoneType
        
#         #not at all used, yet
#         self.Phi = SlimFC(3, 3) # input 3 output 3

#         #lets try to remove all debug_vin stuff
#         #if model_config['debug_vin']: #changed from model_conf, think that was a typo
#         #    self.debug_vin = model_config['debug_vin']

#     # Implement your own forward logic, whose output will then be sent
#     # through an LSTM.



#         """everything down to comment is stolen from fcnet.py"""
#         #Stolen right from FCNet
    
#         hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
#             model_config.get("post_fcnet_hiddens", [])
#         )
#         #activation = model_config.get("fcnet_activation")
#         #if not model_config.get("fcnet_hiddens", []):
#         #    activation = model_config.get("post_fcnet_activation")
#         #no_final_linear = model_config.get("no_final_linear")
#         #self.vf_share_layers = model_config.get("vf_share_layers")
        

#         layers = []
#         prev_layer_size = int(np.product(obs_space.shape))
#         self._logits = None

#         # Create layers 0 to second-last.
#         for size in hiddens[:-1]:
#             layers.append(
#                 SlimFC(
#                     in_size=prev_layer_size,
#                     out_size=size,
#                     initializer=normc_initializer(1.0),
#                     #activation_fn=activation,
#                 )
#             )
#             prev_layer_size = size

#         # The last layer is adjusted to be of size num_outputs, but it's a
#         # layer with activation.
#         # if no_final_linear and num_outputs:
#         #     layers.append(
#         #         SlimFC(
#         #             in_size=prev_layer_size,
#         #             out_size=num_outputs,
#         #             initializer=normc_initializer(1.0),
#         #             #activation_fn=activation,
#         #         )
#         #     )
#         #     prev_layer_size = num_outputs
#         # Finish the layers with the provided sizes (`hiddens`), plus -
#         # iff num_outputs > 0 - a last linear layer of size num_outputs.
#         #else:
#         if len(hiddens) > 0:
#             layers.append(
#                 SlimFC(
#                     in_size=prev_layer_size,
#                     out_size=hiddens[-1],
#                     initializer=normc_initializer(1.0),
#                     #activation_fn=activation,
#                 )
#             )
#             prev_layer_size = hiddens[-1]
#         if num_outputs:
#             self._logits = SlimFC(
#                 in_size=prev_layer_size,
#                 out_size=num_outputs,
#                 initializer=normc_initializer(0.01),
#                 #activation_fn=None,
#             )
#         else:
#             self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
#                 -1
#             ]

#         self._hidden_layers = torch.nn.Sequential(*layers)


#         #self._value_branch_separate = None
#         # if not self.vf_share_layers:
#         #     # Build a parallel set of hidden layers for the value net.
#         #     prev_vf_layer_size = int(np.product(obs_space.shape))
#         #     vf_layers = []
#         #     for size in hiddens:
#         #         vf_layers.append(
#         #             SlimFC(
#         #                 in_size=prev_vf_layer_size,
#         #                 out_size=size,
#         #                 #activation_fn=activation,
#         #                 initializer=normc_initializer(1.0),
#         #             )
#         #         )
#         #         prev_vf_layer_size = size
#         #     self._value_branch_separate = nn.Sequential(*vf_layers)

#         self._value_branch = SlimFC(
#             in_size=prev_layer_size,
#             out_size=1,
#             initializer=normc_initializer(0.01),
#             #activation_fn=None,
#         )
#         # Holds the current "base" output (before logits layer).
#         self._features = None
#         # Holds the last input, in case value branch is separate.
#         self._last_flat_in = None


#         """ Everythin above (until comment) is stolen straight from fcnet"""

#     def VP(self,s):
#         rout = s[:, :, 2]
#         rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
#         p = 1 - s[:, :, 0]
#         K=20

#         h, w = p.shape[0], p.shape[1]
#         # h = 4
#         # w = 4
#         #we get issues with wandb showing the v plot when using tensors instead of np array
#         #v = torch.from_numpy(np.zeros((h,w, K+1))) #overly simple way to use tensors
#         v = torch.zeros((h,w, K+1)) #overly simple way to use tensors
        
#         # return
#         #v = np.zeros((h,w, K+1)) #overly simple way to use tensors 
#         for k in range(K):
#             for i in range(h):
#                 for j in range(w):
#                     # continue
#                     v[i,j, k+1] = v[i,j,k]
#                     #continue
#                     for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
#                         if di + i < 0 or dj + j < 0:
#                             continue
#                         if di + i >= h or dj + j >= w:
#                             continue
#                         ip = i + di
#                         jp = j + dj
#                         nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
#                         v[i,j,k+1] = max( v[i,j,k+1], nv)
#         s[:,:,0],s[:,:,2] = p,v[:,:,-1]
        
#         #dim4 = np.expand_dims(v[:,:,-1],axis=2) 
#         #vp = np.concatenate((s,dim4),axis=2)
#         vp = s.flatten()
#         vp = vp.type(torch.FloatTensor)
#         return vp


#     def forward(self, input_dict, state, seq_lens): #dont think this is currently being used
#         obs = input_dict["obs_flat"]
#         self.n_forwards = self.n_forwards + 1
#         self.t_total = self.t_total + time.time() - self.tc
#         self.tc = time.time()
        
#         # Store last batch size for value_function output.
#         self._last_batch_size = obs.shape[0]        
#         vp = torch.tensor(np.zeros((obs.shape)))

#         for ii in range(obs.shape[0]):      
#             vp = self.VP(torch.zeros((4,4,3)))
            
#         self._last_flat_in = obs.reshape(obs.shape[0], -1)
#         self._features = self._hidden_layers(self._last_flat_in)
#         logits = self._logits(self._features) if self._logits else self._features
#         logits = self.nn(obs) #this seems to be just a much simpler version of the above (single layer)
        
#         self.t_forwards = self.t_forwards + time.time() - self.tc        
#         # print(f"> fwd. {self.n_forwards}: Time per iteration {self.t_total / self.n_forwards} TPI (forward stuff) {self.t_forwards / self.n_forwards} ")
#         return logits, state #from fcnet

    
#     def value_function(self): #dont think this is currently being used

#         #return torch.tensor([0]*32) # a little test, causes weird error
#         # if self._value_branch_separate:
#         #     return self._value_branch(
#         #         self._value_branch_separate(self._last_flat_in)
#         #     ).squeeze(1)
#         # else:

#         #if i can get current location, then i can use vp module here
#         return self._value_branch(self._features).squeeze(1) #
        
#         #return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))
#         #this is just zeroes?
        
#         #v = VIP(env.reset(),Phi(env.reset()))[:,:,-1] # get last layer of the value prop

#         #return v
    


#     #pi from agent.py
#     def pi(self, s, k=None): #we never enter this (except with the irlc-visualise stuff i think)
#         # return self.env.action_space.sample() #!s

#         #hacky way of getting location of agent
#         s = env.reset() #dont use env.reset when changing to "real environment"
#         a_map = s[:,:,1]
#         a_loc = np.where(a_map==1)
#         a_loc = (a_loc[0][0],a_loc[1][0])

#         return self.obs_space.action_space.sample()



# """
# def VIP(Phi, K=20):#k=20 default, 
#     (rin, rout, p) = Phi
#     h, w = p.shape[0], p.shape[1]
#     #we get issues with wandb showing the v plot when using tensors instead of np array
#     v = torch.from_numpy(np.zeros((h,w, K+1))) #overly simple way to use tensors
#     #v = np.zeros((h,w, K+1)) #overly simple way to use tensors 
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
#     """





# def Phi(s):
#     # THIS SHOULD BE TRAINED IN A NN, for now we just experiment with this though
#     """ Is this supposed to be a linear NN or a CNN, paper suggests CNN? """

#     rout = s[:, :, 2]
#     rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
#     p = 1 - s[:, :, 0] # what exactly does this represent, currently
#     #Phi = (rin,rout,p)
#     #print(Phi)

#     return (rin, rout, p)
    
# """ #"my own" experiment, maybe outdated by tues version (below)
# ModelCatalog.register_custom_model(vin_label, VINNetwork)
# def my_experiment(a):
#     print("Hello world")
#     # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    
#     mconf = dict(custom_model=vin_label, use_lstm=False)#, debug_vin=True) 
    
    
#     config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
#     #config from example_maze.py
#     #config = A3CConfig().training(lr=0.01/10, grad_clip=30.0).resources(num_gpus=0).rollouts(num_rollout_workers=1)
#     config = config.framework('torch')


#     # config.
#     config = config.callbacks(MyCallbacks) #not sure i understand callbacks
#     #config = config.model(custom_model="my_torch_model", use_lstm=False)

#     #print(config.to_dict()) #
#     config.model['fcnet_hiddens'] = [24, 24]  
#     # lets try to make use of our own custom_net somewhow
#     # env = gym.make("MazeDeterministic_empty4-v0")


#     #something in here needs to be a tensor, but is nonetype
#     trainer = config.build(env="MazeDeterministic_empty4-v0") # juump into a3c setup()
#     #trainer = config.build(env="CartPole-v1") 


#     for t in range(10): #150
#         print("Main training step", t)
#         result = trainer.train() #calls this line, enters worker.py and crashes. crashes in trainer?
#         rewards = result['hist_stats']['episode_reward']
#         print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])
    



#     #config.save 
#     # #crash "A3CConfig has no attribute 'save'"
# """

# #Tue's "Scrap file" code:
# ModelCatalog.register_custom_model(vin_label, VINNetwork)
# def my_experiment(a):
#     print("Hello world")
#     # see https://docs.ray.io/en/latest/rllib/rllib-training.html
#     mconf = dict(custom_model=vin_label, use_lstm=False)
#     # mconf = {}
#     config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
#     config = config.framework('torch')

#     env_name = "MazeDeterministic_empty4-v0"
#     #env_name = 'Maze_empty4-v0' # i dont have this one
#     config = config.callbacks(MyCallbacks)

#     # Set up alternative model (gridworld).
#     # config = config.model(custom_model="my_torch_model", use_lstm=False)
#     print(config.to_dict())
#     config.model['fcnet_hiddens'] = [24, 24] #why 24, just arbitrary?
#     # config.model['debug_vin'] = True
#     # config.model['saf'] = True
#     # config.model['asdfasdf'] = 234

#     config.model['custom_model_config'] = {}
#     config.model['custom_model_config']['env_name'] = env_name
#     from ray.tune.registry import _global_registry # EXAMPLE: Creating an environment.
#     env =     _global_registry.get("env_creator", env_name)
#     hw = env(env_config=config.env_config).game.height, env(env_config=config.env_config).game.width
#     # env(env_config=config.env_config).observation_space
#     config.model['custom_model_config']['env_shape'] = hw
#     config.evaluation_interval = 1

#     def my_eval_fun(a3c, worker_set, **kwargs):
#         # print("Evaluating...", args, kwargs)
#         # s = args[1].local_worker().sample()
#         # a = 234]
#         print("Running custom evaluation function...")
#         def my_pol_fun(a3policy, *args, a3c=None, **kwargs):
#             print(args, kwargs)
#             # cb = a3policy.callbacks
#             a3c.callbacks.evaluation_call(a3policy )


#             # a3policy.callbacks
#             # a = 234
#             return {}

#         #this gives me an error: functools is not defined
#         #import functools #maybe this is it? -> causes all kinds of weird problems, leave it out for now
#         #worker_set.foreach_policy(functools.partial(my_pol_fun, a3c=a3c) )

#         return dict(my_eval_metric=123)

#     config.custom_evaluation_function = my_eval_fun
#     # config['evaluation_interval'] = 1


#     # config.model['debug_vin'] = True
#     # env = gym.make("MazeDeterministic_empty4-v0")


#     if False:
#         ray.init()
#         # from raya3c.a3c import A3C
#         # trainer = A3C(env=env_name, config=config.to_dict())
#         while True:
#             print(trainer.train())
#     else:

#         trainer = config.build(env="MazeDeterministic_empty4-v0")
#         for t in range(200): #150
#             print("Main training step", t)
#             result = trainer.train()
#             rewards = result['hist_stats']['episode_reward']
#             try:
#                 int(result['episode_reward_mean'])
#             except:
#                 print("whoops")#just want to catch when the result is nan
                
#             print("training epoch", t, len(rewards), max(rewards) if len(rewards) > 0 else -1, result['episode_reward_mean'])

#     # config.save





#     #only needed for printing the wandb image of V
#     #env = gym.make("MazeDeterministic_empty4-v0")

#     #s = env.reset() # 
#     """ s looks like: 
# array([[1., 0., 0., 1.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [1., 0., 0., 1.]])

# array([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 1., 0.],
#        [0., 0., 0., 0.]])

# array([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [1., 0., 0., 0.]])
#        """
#     #env = VideoMonitor(env)
    
#     #this is only needed if i wanna visualize the trainer
#     #train(env, VINNetwork(env, trainer,num_outputs=env.action_space.n, model_config=dict(custom_model=vin_label, use_lstm=False), name='hardcoded'), num_episodes=10) #train() is a function in agent.py that crashes


#     #create an image in wandb (of last layer of V)
#     #v = VIP(Phi(s))
#     #v = v[:,:,-1]

#     #print(s[:,:,0])
#     #print(s[:,:,1])
#     #print(s[:,:,2])

#     #p = 1 - s[:, :, 0] 
#     #images = wandb.Image(p, caption="Top: Output, Bottom: Input")
#     #wandb.log({"image of p": images})

#     #plt.imshow(v)
#     #plt.show()

#     #images = wandb.Image(v, caption="Top: Output, Bottom: Input")
#     #wandb.log({"image of v": images})




# if __name__ == "__main__":
#     #
#     # import pickle
#     #
#     # with open('mydb', 'wb') as f:
#     #     pickle.dump({'x': 344}, f)
#     #
#     # with open('mydb', 'rb') as f:
#     #     s = pickle.load(f)
#     #     # pickle.dump({'x': 344}, f)
#     #
#     # sys.exit()
#     res = []
#     DISABLE = True
#     # key = "04ff52c3923a648c9c263246bf44cb955a8bf56d"

#     # Optional
#     # wandb.watch(model)

#     # sys.exit()
#     # my_experiment(34)
#     # sys.exit()
    
    
#     # with DTUCluster(job_group="myfarm/job0", nuke_group_folder=True, rq=False, disable=False, dir_map=['../../../mavi'],
#     #                 nuke_all_remote_folders=True) as cc:
#     #     # my_experiment(2)
#     #     wfun = cc.wrap(my_experiment) if not DISABLE else my_experiment
#     #     for a in [1]:
#     #         res.append(wfun(a))
#     # print(res)

#     my_experiment(1)
#     # wfun = my_experiment
#     # for a in [1]:
#     #     res.append(wfun(a))
#     #     a = 123
    

#     # res = cc.wrap(myfun)(args1, args2)
#     # val2 = myexperiment(1,2)
#     # wait_to_finish()
#     print("Job done")
#     sys.exit()