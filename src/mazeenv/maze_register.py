import gym
import numpy as np
from mazeenv.maze_environment import MazeEnvironment
gym.envs.register(
     id="MazeDeterministic_empty4-v0",
     # entry_point='mazeenv.maze_environment:MazeEnvironment',
     entry_point='mazeenv.envstufftemp:MazeEnvironment', 

     max_episode_steps=200,
     kwargs=dict(size=8, blockpct=0),  #size=4 is default, change this to change size of map# removed seed = 0
)

from ray.tune.registry import register_env

def env_creator(env_config):
     env = gym.make("MazeDeterministic_empty4-v0")
     from gym.wrappers.flatten_observation import FlattenObservation
     #env = FlattenObservation(env) # flat as fuck to avoid rllib interpreting it as an image.
     #removed that line because tue said so 
     return env

register_env('MazeDeterministic_empty4-v0', env_creator)

# Apparently needed because #reasons.
# from ray import tune
# tune.register_env('MazeDeterministic_empty4-v0', lambda cfg: gym.make("MazeDeterministic_empty4-v0"))
# gym.register("MazeDeterministic_simple3-v0", MazeEnvironment)