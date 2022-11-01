from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex04.model_pendulum import ContiniousPendulumModel
from irlc.ex05.direct_plot import plot_solutions
import numpy as np

def compute_pendulum_solutions():
    model = ContiniousPendulumModel()
    """
    Test out implementation on a fairly small grid. Note this will work fairly terribly.
    """
    guess = {'t0': 0,
             'tF': 4,
             'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
             'u': [np.asarray([0]), np.asarray([0])]}

    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=20, ftol=1e-3),
               get_opts(N=80, ftol=1e-6)
               ]

    solutions = direct_solver(model, options)
    return model, solutions

if __name__ == "__main__":
    model, solutions = compute_pendulum_solutions()
    plot_solutions(model, solutions, animate=True, pdf="direct_pendulum_real")
