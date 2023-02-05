import gym
import numpy as np
from mazeenv.MazeListenerSpeakerEnv import MazeListenerSpeakerEnv
from ray.rllib.env.multi_agent_env import make_multi_agent


gym.envs.register(
     id="MA_Maze-v0",
     # entry_point='mazeenv.MazeListenerSpeakerEnv:MazeListenerSpeakerEnv',
     entry_point='mazeenv.MazeListenerSpeakerEnv:SimpleMAEnv',

     max_episode_steps=200,
     kwargs=dict(size=4, blockpct=0, seed = 0),  #size=4 is default, change this to change size of map, remove seed = 0 to randomize
)

from ray.tune.registry import register_env

def env_creator(env_config):
     env = gym.make("MA_Maze-v0")
     from gym.wrappers.flatten_observation import FlattenObservation
     #env = FlattenObservation(env) # flat as fuck to avoid rllib interpreting it as an image.
     #removed that line because tue said so 
     return env

register_env('MA_Maze-v0', env_creator)

# Apparently needed because #reasons.
# from ray import tune
# tune.register_env('MazeDeterministic_empty4-v0', lambda cfg: gym.make("MazeDeterministic_empty4-v0"))
# gym.register("MazeDeterministic_simple3-v0", MazeEnvironment)