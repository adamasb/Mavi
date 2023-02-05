import random
from irlc.gridworld.gridworld_environments import BookGridEnvironment, BridgeGridEnvironment, FrozenLake
from irlc.utils.video_monitor import VideoMonitor
from irlc.ex01.agent import train
from irlc.gridworld.hidden_agents import ValueIterationAgent2
import numpy as np
from collections import defaultdict
from pyglet.window import key
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym.spaces.discrete import Discrete
from irlc.ex09.mdp import MDP2GymEnv
from irlc.gridworld.gridworld_mdp import GridworldMDP, FrozenGridMDP
from irlc.gridworld import gridworld_graphics_display
from irlc import Timer



from irlc.gridworld.gridworld_environments import GridworldEnvironment
import matplotlib.pyplot as plt
from gym import Env
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
from random import choice, randint
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np
from mazebase.games.goal_based_games import MovingAgent
from mazebase.items.terrain import Goal
import mazebase.games as games
from mazebase.games import featurizers
from mazebase.games import curriculum
import logging


logging.getLogger().setLevel(logging.DEBUG)

pp = PrettyPrinter(indent=2, width=160)
from gym.spaces.discrete import Discrete

from ray.rllib.env.multi_agent_env import make_multi_agent
from mazeenv.maze_environment import MazeEnvironment

from gym.spaces import Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# class MazeListenerSpeakerEnv(Env): #argument used to be: 'MultiEnv' -> not sure why we have this as an argument

class MazeListenerSpeakerEnv(MazeEnvironment, MultiAgentEnv):
    # def __init__(self): #im not sure if the arguments should be passed in here or only in the base env
    def __init__(self, size=10, blockpct=0.0, living_reward=-0.05, seed=None, render_mode='human'):
        self.listener_name = "listener"
        self.speaker_name = "speaker"
        self.base_env = MazeEnvironment()
        self.goals = ["red", "blue"]
        self.speakers_last_action = 0.5


        #define action space for each agent
        self.listener_action_space = Discrete(5)
        self.speaker_action_space = Discrete(2) #number of goals?? how does the communication happen
        #override observation space for each agent (kinda)


        #following code is from the original maze environment - not sure if i need it
        sg = curriculum.CurriculumWrappedGame(
            games.SingleGoal,
            blockpct=blockpct,
            waterpct=0,
            living_reward=living_reward,
            curriculums={
                'map_size': games.curriculum.MapSizeCurriculum(
                     (size,) * 4,  (size,) * 4,  (size,) * 4 #initial, minimum, maximum
                )
            }
        )

        self.game = None
        self.all_games = [sg]
        self.action_space = Discrete(5) # up, down, left, right, stand still.

        self.render_mode = render_mode
        # Rendering related stuff.
        self.zoom = 0.5
        self.view_mode = 0
        self.display = None
        self.resets = 0


        self.game = games.MazeGame(
            self.all_games,
            featurizer=featurizers.GridFeaturizer()
        )
  



    def reset(self):
        self.current_turn = self.speaker_name
        # raise Exception("RESET EXCEPTION HIT. ")
        # Reset og valgt korrekt maal.
        s = self.base_env.reset()
        self.goal_index = np.random.choice(2)
        # goal_color = self.goals[ ]
        self.s = s
        # Fjern information fra s om hvor
        return {self.speaker_name: np.asarray([self.goal_index]), 
                self.listener_name: self.observation_space.sample()} #temp sample. 


        #some of this code is needed to run the program, but maybe they should be included differenly
        #the following is from the original maze environment reset function 
        self.game.reset()
        # else:
        if self.seed is not None:
            # seed = np.random.seed()
            r_state = random.getstate()
            np_state = np.random.get_state()

            np.random.seed(self.seed)
            random.seed(self.seed)

            self.game.reset()
            if self.render_mode == 'human':
                self.render_as_text()
            np.random.set_state(np_state)
            random.setstate(r_state)

            # np.random.seed(seed)
        else:
            self.game.reset()
        # if self.seed is not None:
        # else:
        #     self.game.reset()

        # max_w, max_h = game.get_max_bounds()
        # self.game = game

        if self.render_mode == 'human':
            self.mdp = GridworldMDP(self._get_grid(), living_reward=self.living_reward)
        # s = self._state()
        # self.state = s
        return self._state()




    def step(self, action_dict):
        # udregn reward alt efter om der er vundet eller ej. Dvs. reward = 0 hvis vi finder maal med forkert farve, og 1 hvis rigtig farve, og evt. -0.01 hvis ingen har vundet.
        # udregn ogsaa done.
        raise Exception("Took a single step")
        reward = 0
        done = False
        rewards = {self.listener_name: reward, self.speaker_name: reward}
        dones = {self.listener_name: done, self.speaker_name: done}
        infos = {self.listener_name: {}, self.speaker_name: {}}

        if self.speaker_name in action_dict:
            sp, r, done, info = self.base_env.step(action_dict[self.speaker_name])



            rewards = {self.listener_name: r, self.speaker_name: r}
            dones = {self.listener_name: done, self.speaker_name: done}
            infos = {self.listener_name: {}, self.speaker_name: {}}

            s0 = {self.speaker_name: np.asarray([self.goal_index])}
            return s0, rewards, dones, infos
            # listeners turn
        else:
            # speakers turn
            action_speaker = action_dict[self.speaker_name]
            s = np.stack([self.s, np.zeros( self.s.shape + (1,), ) + action_speaker ])
            s0 = {self.listener_name: s}
            return s0, rewards, dones, infos




    def _state(self): # Multi agent/goal version - 5 dimensions (+2, 1 for goal2 and one for the "correct goal")
            game = self.game
            x = np.zeros((game.height, game.width, 5)) 
            #random.choice([0,1]) #decide if the "correct goal" is dim 2 or 3
            x[:,:,4] = 0 #initially just force it to be 0 
            for i in range(len( game._map )):
                for j in range(len(game._map[i])):
                    for t in game._map[i][j]:
                        if isinstance(t, MovingAgent):
                            x[j,i,1] = 1
                            x[j,i,0] = 0
                        elif isinstance(t, Goal):
                            x[j,i,2] = 1
                            x[j,i,0] = 0
                            x[j,i,3] = 1 #this is the "second goal"
                        else:
                            x[j,i,0] = 1 #1 originally, what happens now: No major change, i think. (no corner walls tho)
            return x




""" original """


class MazeEnvironment(Env):
    # My attempt at making a maze-environment
    metadata = {
        'render.modes': ['native', 'human', 'rgb_array'],
        'video.frames_per_second': 10,
    }
    def get_keys_to_action(self):
        #      amap = {0: 'down', 1: 'left', 2:'right', 3:'up', 4:'pass'}
        return {(key.LEFT,): 1, (key.RIGHT,): 2, (key.UP,): 3, (key.DOWN,): 0, (key.S,): 4}

    def __init__(self, size=10, blockpct=0.0, living_reward=-0.05, seed=None, render_mode='native'):
        self.living_reward = living_reward
        sz = (size,) * 4
        self.seed = seed
        # if self.seed is not None:
        #     np.random.seed(seed)
        sg = curriculum.CurriculumWrappedGame(
            games.SingleGoal,
            blockpct=blockpct,
            waterpct=0,
            living_reward=living_reward,
            curriculums={
                'map_size': games.curriculum.MapSizeCurriculum(
                    sz, sz, sz # not planning on using variable map sizes for multi-agent
                )
            }
        )
        self.game = None
        self.all_games = [sg]
        self.action_space = Discrete(5) # up, down, left, right, stand still.

        self.render_mode = render_mode
        # Rendering related stuff.
        self.zoom = 0.5
        self.view_mode = 0
        self.display = None
        self.reset()
        self.observation_space = Box(low=0, high=1, shape=self.state.shape, dtype=np.float)

        
        self.game = games.MazeGame(
            self.all_games,
            featurizer=featurizers.GridFeaturizer()
        )

    def reset(self):
        # if self.seed is None or self.game is None:
        self.game = games.MazeGame(
            self.all_games,
            featurizer=featurizers.GridFeaturizer()
        )
        # else:
        if self.seed is not None:
            # seed = np.random.seed()
            r_state = random.getstate()
            np_state = np.random.get_state()

            np.random.seed(self.seed)
            random.seed(self.seed)

            self.game.reset()
            if self.render_mode == 'human':
                self.render_as_text()
            np.random.set_state(np_state)
            random.setstate(r_state)

            # np.random.seed(seed)
        else:
            self.game.reset()
        # if self.seed is not None:
        # else:
        #     self.game.reset()

        # max_w, max_h = game.get_max_bounds()
        # self.game = game

        if self.render_mode == 'human':
            self.mdp = GridworldMDP(self._get_grid(), living_reward=self.living_reward)
        # s = self._state()
        # self.state = s
        return self._state()

    @property
    def state(self):
        return self._state() # visualization purposes.

   
    def _state(self):
        game = self.game
        x = np.zeros((game.height, game.width, 3))
        for i in range(len( game._map )):
            for j in range(len(game._map[i])):
                for t in game._map[i][j]:
                    if isinstance(t, MovingAgent):
                        x[j,i,1] = 1
                        x[j,i,0] = 0
                    elif isinstance(t, Goal):
                        x[j,i,2] = 1
                        x[j,i,0] = 0
                    else:
                        x[j,i,0] = 1 #1 originally, what happens now: No major change, i think. (no corner walls tho)
        return x

    

    def step(self, action):
        game = self.game
        # actions = game.all_possible_actions()
        amap = {0: 'down', 1: 'left', 2:'right', 3:'up', 4:'pass'}

        # action = action_func(actions)
        game.act(amap[action])
        reward = game.reward()
        sp = self._state()
        done = game.is_over()
        info = {}


        if done and self.render_mode == 'human':
            self._reset_display()
        return sp, reward, done, info


    def render(self, mode='human', state=None, agent=None, v=None, Q=None, pi=None, policy=None, v2Q=None, gamma=0, method_label="", label=None):
        # Render function from the course. State is a tuple (i,j)
        # if state is not None:

        state = self.state
        I,J = state[:,:,1].nonzero()
        if len(I) == 0:
            print(I, state)
        state = (J[0], I[0])
        print(state)
        self.render_as_text()

        self.agent = agent
        self.render_steps = 0 # for vizualization
        if label is None:
            label = f"{method_label} AFTER {self.render_steps} STEPS"
        speed = 1
        if self.display is None:
            self._reset_display()
        # print("In environment - render")

        if state is None:
            state = self.state

        avail_modes = []
        if agent != None:
            label = (agent.label if hasattr(agent, 'label') else method_label) if label is None else label
            v = agent.v if hasattr(agent, 'v') else None
            Q = agent.Q if hasattr(agent, 'Q') else None
            policy = agent.policy if hasattr(agent, 'policy') else None
            v2Q = agent.v2Q if hasattr(agent, 'v2Q') else None
            avail_modes = []
            if Q is not None:
                avail_modes.append("Q")
                avail_modes.append("v")
            elif v is not None:
                avail_modes.append("v")

        if len(avail_modes) > 0:
            self.view_mode = self.view_mode % len(avail_modes)
            if avail_modes[self.view_mode] == 'v':
                preferred_actions = None

                if v == None:
                    preferred_actions = {}
                    v = {s: Q.max(s) for s in self.mdp.nonterminal_states}
                    for s in self.mdp.nonterminal_states:
                        acts, values = Q.get_Qs(s)
                        preferred_actions[s] = [a for (a,w) in zip(acts, values) if np.round(w, 2) == np.round(v[s], 2)]

                if v2Q is not None:
                    preferred_actions = {}
                    for s in self.mdp.nonterminal_states:
                        q = v2Q(s)
                        mv = np.round( max( q.values() ), 2)
                        preferred_actions[s] = [k for k, v in q.items() if np.round(v, 2) == mv]

                if agent != None and hasattr(agent, 'policy') and agent.policy is not None and state in agent.policy and isinstance(agent.policy[state], dict):
                    for s in self.mdp.nonterminal_states:
                        preferred_actions[s] = [a for a, v in agent.policy[s].items() if v == max(agent.policy[s].values()) ]

                if hasattr(agent, 'returns_count'):
                    returns_count = agent.returns_count
                else:
                    returns_count = None
                if hasattr(agent, 'returns_sum'):
                    returns_sum = agent.returns_sum
                else:
                    returns_sum = None
                self.display.displayValues(mdp=self.mdp, v=v, preferred_actions=preferred_actions, currentState=state, message=label, returns_count=returns_count, returns_sum=returns_sum)

            elif avail_modes[self.view_mode] == 'Q':

                if hasattr(agent, 'e') and isinstance(agent.e, defaultdict):
                    eligibility_trace = defaultdict(float)
                    for k, v in agent.e.items():
                        eligibility_trace[k] = v

                else:
                    eligibility_trace = None
                    # raise Exception("bad")
                # print(eligibility_trace)
                self.display.displayQValues(self.mdp, Q, currentState=state, message=label, eligibility_trace=eligibility_trace)
            else:
                raise Exception("No view mode selected")
        else:
            self.display.displayNullValues(self.mdp, currentState=state)

        self.display.end_frame()
        render_out = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return render_out

    def _reset_display(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

            # self.display.close()

        # if self.display is None:
        gridSize = int(150 * self.zoom)
        self.display = gridworld_graphics_display.GraphicsGridworldDisplay(self.mdp, gridSize) #cant visualize ( env.render() ) #
        self.viewer = self.display.ga.gc.viewer


    def render_as_text(self):
        return self.game.display()

    def _get_grid(self):
        # Turn this into an MDP environment.
        x = self._state()
        grid = []
        for i in range(x.shape[0]):
            g = []
            for j in range(x.shape[1]):
                if x[i, j, 0] == 1:
                    g.append('#')
                elif x[i, j, 1] == 1:
                    g.append('S')
                elif x[i, j, 2] == 1:
                    g.append(1)
                else:
                    g.append(' ')
                    # print("very odd")
            grid = [g] + grid
        return grid

    def mk_mdp(self):
        grid = self._get_grid()

        menv = GridworldEnvironment(grid=grid, living_reward=-0.001)
        return menv



# class

# agent = Agent(cenv)
# cenv = VideoMonitor(cenv)
# agent = PlayWrapper(agent, cenv)
# train(cenv, agent)
if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import CliffGridEnvironment, BookGridEnvironment
    cenv = CliffGridEnvironment()
    from irlc import PlayWrapper, VideoMonitor
    from irlc import Agent, train

    # env = BookGridEnvironment()
    # agent = PlayWrapper(Agent(env), env)
    # agent.label = "Random agent"
    # env = VideoMonitor(env, agent=agent, fps=30, agent_monitor_keys=("label",))
    # train(env, agent, num_episodes=100)
    # env.close()

    env = MazeListenerSpeakerEnv(size=10, blockpct=.3, render_mode='human')
    # env = MazeEnvironment(size=10, blockpct=.3, render_mode='human')
    # agent = ValueIterationAgent2(env, gamma=0.99)
    agent = Agent(env)
    agent = PlayWrapper(agent, env)
    experiment = "experiments/q1_value_iteration"
    env = VideoMonitor(env, agent=agent, fps=100, continious_recording=True, # agent_monitor_keys=('v', 'v2Q'),
                       render_kwargs={'method_label': 'VI'})
    # env.reset()
    train(env, agent, experiment_name=experiment, num_episodes=10, max_steps=100)
    env.close()



    s = env.reset()
    env.step(1)
    env.display()

