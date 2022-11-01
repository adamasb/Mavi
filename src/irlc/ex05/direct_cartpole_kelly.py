import matplotlib.pyplot as plt
from irlc.ex04.model_cartpole import ContiniousCartpole, kelly_swingup
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex05.direct import direct_solver, get_opts
import numpy as np
from scipy.optimize import Bounds
from irlc import savepdf

def make_cartpole_kelly17():
    """
    Creates Cartpole problem. Details about the cost function can be found in \cite[Section 6]{kelly}
    and details about the physical parameters can be found in \cite[Appendix E, table 3]{kelly}.
    """
    # this will generate a different carpole environment with an emphasis on applying little force u.
    duration = 2.0
    Q = np.zeros((4, 4)) #!b
    cost = SymbolicQRCost(Q=Q, R=np.asarray([[1.0]]) ) #!b
    # Initialize the cost-function above. You should do so by using a call of the form:
    # cost = SymbolicQRCost(Q=..., R=...) # Take values from Kelly
    # The values of Q, R can be taken from the paper.

    _, bounds, _ = kelly_swingup(maxForce=20, dist=1.0) # get a basic version of the bounds (then update them below).
    bounds['tF'] = Bounds([duration], [duration])  # #!b #!b Update the bounds so the problem will take exactly tF=2 seconds.

    # Instantiate the environment as a ContiniousCartpole environment. The call should be of the form:
    # env = ContiniousCartpole(...)
    # Make sure you supply all relevant physical constants (maxForce, mp, mc, l) as well as the cost and bounds. Check the
    # ContiniousCartpole class definition for details.
    model = ContiniousCartpole(maxForce=20, mp=0.3, mc=1.0, g=9.81, l=0.5, cost=cost, simple_bounds=bounds) #!b #!b
    guess = model.guess()
    guess['tF'] = duration # Our guess should match the constraints.
    return model, guess

def compute_solutions():
    model, guess = make_cartpole_kelly17()
    print("cartpole mp", model.mp)
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=40, ftol=1e-6)]
    solutions = direct_solver(model, options)
    return model, solutions

def direct_cartpole():
    model, solutions = compute_solutions()
    from irlc.ex05.direct_plot import plot_solutions
    print("Did we succeed?", solutions[-1]['solver']['success'])
    plot_solutions(model, solutions, animate=True, pdf="direct_cartpole_force")
    model.close()

if __name__ == "__main__":
    direct_cartpole()