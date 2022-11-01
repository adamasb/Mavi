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

# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import matplotlib.pyplot as plt

import wandb



#can be used to create a custom policy or custom trainer
"""CustomPolicy = A3CTorchPolicy.with_updates(
    name="MyCustomA3CTorchPolicy",
    loss_fn=some_custom_loss_fn)

CustomTrainer = A3C.with_updates(
    default_policy=CustomPolicy)"""


vin_label = "vin_network_model"
# Kig paa: FullyConnectedNetwork som er den Model-klassen bruger per default.
# alt. copy-paste FullyConnectedNetwork-koden ind i denne klasse og modififer gradvist (check at den virker paa simple gridworld)
class VINNetwork(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # super().__init__(obs_space, action_space, num_outputs, model_config, name)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        self.num_outputs = env.action_space.n#int(np.product(self.obs_space.shape)) #now we crash,: 'VideoMonitor' object has no attribute 'shape'
        self._last_batch_size = None
        
        #self.Phi = torch.nn.Linear() # dont think we can do this yet
        
        #lets try to remove all debug_vin stuff
        #if model_config['debug_vin']: #changed from model_conf, think that was a typo
        #    self.debug_vin = model_config['debug_vin']

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens): #dont think this is currently being used
        obs = input_dict["obs_flat"]
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        # Return 2x the obs (and empty states).
        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        #if self.debug_vin:
            # Brug VIN-koden fra VI-agent til at udregne value-funktionen for dette Phi.

            # udregn V(s) vha. numpy-VI-AGENT

            
        #    V_np = []
        #    assert( (V_np - V_torch.numpy())< 1e-8 )

        return obs * 2.0, []
    
    
    def value_function(self): #dont think this is currently being used

        """ Needs to consider VI values"""

        v = VIP(env.reset(),Phi(env.reset()))[:,:,-1] # get last layer of the value prop

        return v
        #
        #return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))
        #this is just zeroes?
        #return
    
    #i dont know man, needed for train()
    #def pi(self, s, k=None):
    #    return self.trainer.compute_action(s) #who knows man
    


    #pi from agent.py
    def pi(self, s, k=None):
        # return self.env.action_space.sample() #!s
        
        
        #move = max(self.VIP(s, self.Phi(s))) 
        #call value function here

        #v = VIP(s, Phi(s))[:,:,-1]  #last layer ofv
        #sm = torch.nn.Softmax(self.value_function())

        

        #hacky way of getting location of agent
        s = env.reset() #dont use env.reset when changing to "real environment"
        a_map = s[:,:,1]
        a_loc = np.where(a_map==1)
        a_loc = (a_loc[0][0],a_loc[1][0])

        
        neighbours = []
        
                    
                    

        """ t        
        what we wanna do is:
         - Should be argmax(value function), only considering neighbouring values
         - How to get understanding of current position?
            - Why does the state (s) have multiple 1's
            s = env.reset()
            a_map = s[:,:,1]
            a_loc = np.where(a_map==1)
            a_loc = (a_loc[0][0],a_loc[1][0])


        learn phi (first just use the "stupid algo") 

        """
        #probably dont need to run this so often, maybe just once?
        #print(self.VIP(s, self.Phi(s)))

        #to plot p
        #_,_, p = self.Phi(s)
        #plt.imshow(p)

        #v = self.VIP(s, self.Phi(s))

        #move = 

        #print(move)


        #maybe use compute_action
        #something like:
        #return self.trainer.compute_action(s) #VINNetwork has no attribute traienr
        return self.obs_space.action_space.sample()




def VIP(s, Phi, K=20):#k=20 default, 
    #print(Phi)
    (rin, rout, p) = Phi
    h, w = s.shape[0], s.shape[1]
    v = np.zeros((h,w, K+1))
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
    # THIS SHOULD BE TRAINED IN A NN, for now we just experiment with this though
    """ Is this supposed to be a linear NN or a CNN, paper suggests CNN? """

    rout = s[:, :, 2]
    rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
    p = 1 - s[:, :, 0] # what exactly does this represent, currently
    #Phi = (rin,rout,p)
    #print(Phi)

    return (rin, rout, p)
    

ModelCatalog.register_custom_model(vin_label, VINNetwork)

# class DummyAgent(Agent):
#     def __init__(self, env, trainer):
#         super().__init__(env)
#         self.trainer = trainer
#
#     def pi(self, s, k=None):
#         return self.trainer.compute_action(s)
#
# class MyClass:
#     pass

def my_experiment(a):
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    
    mconf = dict(custom_model=vin_label, use_lstm=False)#, debug_vin=True) 
    # maybe this is the model_config that i need
    
    
    #config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    #config from example_maze.py
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0).resources(num_gpus=0).rollouts(num_rollout_workers=1)


    config = config.framework('torch')


    # config.
    config = config.callbacks(MyCallbacks) #not sure i understand callbacks
    # Set up alternative model (gridworld).
    # config = config.model(custom_model="my_torch_model", use_lstm=False)

    #print(config.to_dict()) #
    config.model['fcnet_hiddens'] = [24, 24]  #does this refer to fcnet.py? doesnt seem like it
    # lets try to make use of our own custom_net somewhow
    # env = gym.make("MazeDeterministic_empty4-v0")

    trainer = config.build(env="MazeDeterministic_empty4-v0") 
    #trainer = config.build(env="CartPole-v1") 


    for t in range(2): #150
        print("Main training step", t)
        result = trainer.train()
        rewards = result['hist_stats']['episode_reward']
        print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])
    



    #config.save 
    # #crash "A3CConfig has no attribute 'save'"

    # return
    # print(pretty_print(result1))
    # import matplotlib.pyplot as plt
    #import matplotlib
    #matplotlib.use('TkAgg')

    # plt.plot(rewards)
    # plt.show()

    # print( rewards )
    # env = gym.make("CartPole-v1")
    # env.reset()
    env = gym.make("MazeDeterministic_empty4-v0")
    #env = gym.make("CartPole-v1")
    s = env.reset() # 
    """ 
array([[1., 0., 0., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [1., 0., 0., 1.]])

array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 0.]])

array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [1., 0., 0., 0.]])
       """
    #env = VideoMonitor(env)
    
    
    #this is only needed if i wanna visualize the trainer
    #train(env, VINNetwork(env, trainer,num_outputs=env.action_space.n, model_config=dict(custom_model=vin_label, use_lstm=False), name='hardcoded'), num_episodes=10) #train() is a function in agent.py that crashes


    #create an image in wandb (of last layer of V)
    v = VIP(s, Phi(s))
    v = v[:,:,-1]
    #p = 1 - s[:, :, 0] 
    #images = wandb.Image(p, caption="Top: Output, Bottom: Input")
    #wandb.log({"image of p": images})
    images = wandb.Image(v, caption="Top: Output, Bottom: Input")
    wandb.log({"image of v": images})


    

#model_config=MODEL_DEFAULTS

    a = 234
    #
    # config = A3CConfig()
    # # Print out some default values.
    # print(config.sample_async)
    # # Update the config object.
    # config.training(lr=tune.grid_search([0.001, 0.0001]), use_critic=False)
    # # Set the config object's env.env.action_space
    # config.environment(env="CartPole-v1")
    # # Use to_dict() to get the old-style python config dict
    # # when running with tune.
    # tune.run(
    #     "A3C",
    #     stop = {"episode_reward_mean": 200},
    #     config = config.to_dict(),
    # )




if __name__ == "__main__":
    #
    # import pickle
    #
    # with open('mydb', 'wb') as f:
    #     pickle.dump({'x': 344}, f)
    #
    # with open('mydb', 'rb') as f:
    #     s = pickle.load(f)
    #     # pickle.dump({'x': 344}, f)
    #
    # sys.exit()
    res = []
    DISABLE = True
    # key = "04ff52c3923a648c9c263246bf44cb955a8bf56d"

    # Optional
    # wandb.watch(model)

    # sys.exit()
    # my_experiment(34)
    # sys.exit()
    
    
    # with DTUCluster(job_group="myfarm/job0", nuke_group_folder=True, rq=False, disable=False, dir_map=['../../../mavi'],
    #                 nuke_all_remote_folders=True) as cc:
    #     # my_experiment(2)
    #     wfun = cc.wrap(my_experiment) if not DISABLE else my_experiment
    #     for a in [1]:
    #         res.append(wfun(a))
    # print(res)

    wfun = my_experiment
    for a in [1]:
        res.append(wfun(a)) # i think this call the function that crashes
        a = 123
    

    # res = cc.wrap(myfun)(args1, args2)
    # val2 = myexperiment(1,2)
    # wait_to_finish()
    print("Job done")
    sys.exit()

