
from minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace
import numpy as np
from minigrid.envs.empty import EmptyEnv

class RandomMinigridEnv(EmptyEnv):
    def __init__(self, *args, fraction=0.3, **kwargs):
        self.fraction = fraction
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # TODO: Check if traversible.
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        for _ in range(int((width - 2) * (height - 2) * self.fraction)):
            j = np.random.randint(height - 2)+1
            i = np.random.randint(width - 2)+1
            from minigrid import Wall
            if (i,j) == self.agent_pos: # don't put on top of agent.
                continue

            self.grid.set(i, j, Wall())
            # self.grid
            a = 234

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.mission = "get to the green goal square"

    def reset(self):
        return super().reset(return_info=True)