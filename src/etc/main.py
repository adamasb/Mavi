# from farmer import *
# import torch
# from irlc import Agent


def main():
    import minigrid
    import gym
    #
    # import gym
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
    #
    # env = gym.make('MiniGrid-Empty-8x8-v0')
    # env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    # env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    # obs, _ = env.reset()  # This now produces an RGB tensor only

    from minigrid.envs.empty import EmptyEnv
    from random_grid import RandomMinigridEnv

    env = RandomMinigridEnv(size=10, fraction=0.2)
    from minigrid.wrappers import FullyObsWrapper
    env = FullyObsWrapper(env)
    
    x, _ = env.reset()

    env = RGBImgObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    obs, _ = env.reset(return_info=True)  # This now produces an RGB tensor only
    import matplotlib.pyplot as plt
    plt.imshow(obs)
    plt.show()
    a = 234
    pass

if __name__ == '__main__':
    main()

    print("Buy world")
