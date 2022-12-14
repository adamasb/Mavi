from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex04.model_pendulum import ContiniousPendulumModel
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc import VideoMonitor, train
from irlc import Agent
import numpy as np


class DirectAgent(Agent):
    def __init__(self, env, guess=None, options=None, simple_bounds=None):
        cmod = env.discrete_model.continuous_model  # Get the continuous-time model for planning

        if guess is None:
            guess = cmod.guess()

        if options is None:
            options = [get_opts(N=10, ftol=1e-3, guess=guess, verbose=False),
                       get_opts(N=60, ftol=1e-6, verbose=False)
                       ]
        if simple_bounds is not None:
            cmod.set_simple_bounds(simple_bounds)

        solutions = direct_solver(cmod, options)

        # The next 3 lines are for plotting purposes. You can ignore them.
        self.x_grid = np.stack([env.discrete_model.continious_states2discrete_states(x) for x in solutions[-1]['grid']['x']])
        self.u_grid = np.stack([env.discrete_model.continious_actions2discrete_actions(x) for x in solutions[-1]['grid']['u']])
        self.ts_grid = np.stack(solutions[-1]['grid']['ts'])
        # set self.ufun equal to the solution (policy) function. You can get it by looking at `solutions` computed above
        self.solutions = solutions
        self.ufun = solutions[-1]['fun']['u'] #!b #!b set self.ufun = solutions[....][somethingsomething] (insert a breakpoint, it should be self-explanatory).
        super().__init__(env)

    def pi(self, x, t=None): #!f
        """ Return the action given x and t. As a hint, you will only use t, and self.ufun computed a few lines above"""
        if t > self.ts_grid[-1]:
            print("Bad time!", t, self.ts_grid[-1])
            raise Exception("Time exceed agents planning horizon")

        u = self.ufun(t)
        u = np.asarray(self.env.discrete_model.continious_actions2discrete_actions(u))
        return u

def train_direct_agent(animate=True, plot=False):
    env = ContiniousPendulumModel()
    """
    Test out implementation on a fairly small grid. Note this will work fairly terribly.
    """
    guess = {'t0': 0,
             'tF': 4,
             'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
             'u': [np.asarray([0]), np.asarray([0])]}

    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=60, ftol=1e-6)
               ]

    dmod = DiscretizedModel(model=env, dt=0.1) # Discretize the pendulum model. Used for creatingthe environment.
    denv = ContiniousTimeEnvironment(discrete_model=dmod, Tmax=4)
    agent = DirectAgent(denv, guess=guess, options=options)
    denv.Tmax = agent.solutions[-1]['fun']['tF'] # Specify max runtime of the environment. Must be based on the Agent's solution.

    if animate:
        denv = VideoMonitor(denv)
    stats, traj = train(denv, agent=agent, num_episodes=1, return_trajectory=True)

    if plot:
        from irlc import plot_trajectory
        plot_trajectory(traj[0], env=denv)
        import matplotlib.pyplot as plt
        from irlc import savepdf
        savepdf("direct_agent_pendulum")
        plt.show()

    return stats

if __name__ == "__main__":
    stats = train_direct_agent(animate=True, plot=True)
    print("Obtained cost", -stats[0]['Accumulated Reward'])

