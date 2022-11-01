import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))

from a3c import A3CConfig
# import farmer
#from dtufarm import DTUCluster
from irlc import Agent, train, VideoMonitor
import gym
import numpy as np
from ray import tune
from ray.tune.logger import pretty_print
#apparently this cant be found
#from raya3c.my_callback import MyCallbacks


class DummyAgent(Agent):
    def __init__(self, env, trainer):
        super().__init__(env)
        self.trainer = trainer

    def pi(self, s, k=None):
        return self.trainer.compute_action(s)

class MyClass:
    pass

def my_experiment(a):
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    config = config.framework('torch')
    
    #cant use callbacks idk
    #config = config.callbacks(MyCallbacks)
    # Set up alternative model (gridworld).

    #print(config.to_dict())
    config.model['fcnet_hiddens'] = [24, 24]

    #lets figure out how to insert our own network


    
    #do we keep this in?
    env = gym.make("CartPole-v1")
    #trainer = config.build(env=env)

    trainer = config.build(env="CartPole-v1")

    for t in range(1):
        #print("Main training step", t)
        result = trainer.train()
        rewards = result['hist_stats']['episode_reward']
        print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])


    # i think this return is meant to just ignore the rest of the code
    #return
    # print(pretty_print(result1))
    
    
    #i have added this to try:
    import matplotlib
 #   matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    

    plt.plot(rewards)
    plt.show()
    print( rewards )
    env = gym.make("CartPole-v1")
    # env.reset()
    # trainer.compute_action(env.reset())
    env = VideoMonitor(env)
    train(env, DummyAgent(env, trainer), num_episodes=10)
    a = 234
    #
    # config = A3CConfig()
    # # Print out some default values.
    # print(config.sample_async)
    # # Update the config object.
    # config.training(lr=tune.grid_search([0.001, 0.0001]), use_critic=False)
    # # Set the config object's env.
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


    #i threw in this section
#    import wandb
#    wandb.init(project="my-test-project", name="run1")
#    wandb.config = {
#        "learning_rate": 0.001,
#        "epochs": 100,
#        "batch_size": 128
#    }
#    # for loss in range(10):
#    #     wandb.log({"loss": np.sqrt(loss)})
#    # wandb.watch(model, log_freq=100)
#    self.wandb = wandb
    
    
    # Optional
    #wandb.watch(model)

    # sys.exit()
    # my_experiment(34)
    # sys.exit()

    #cant use dtucluster
    #with DTUCluster(job_group="myfarm/job0", nuke_group_folder=True, rq=False, disable=False, dir_map=['../../../mavi'],
    #                nuke_all_remote_folders=True) as cc:
    #    wfun = cc.wrap(my_experiment) if not DISABLE else my_experiment
    #    for a in [1]:
    #        res.append(wfun(a))
    #    a = 123

    wfun = my_experiment
    for a in [1]:
        res.append(wfun(a))
        a = 123



    print(res)
    # res = cc.wrap(myfun)(args1, args2)
    # val2 = myexperiment(1,2)
    # wait_to_finish()
    print("Job done")
    sys.exit()

