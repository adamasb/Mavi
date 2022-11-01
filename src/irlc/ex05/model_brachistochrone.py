"""
The Brachistochrone problem. See
https://apmonitor.com/wiki/index.php/Apps/BrachistochroneProblem
and \cite{betts2010practical}
"""
from gym.spaces import Box
from scipy.optimize import Bounds
import sympy as sym
import numpy as np
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost

class ContiniouBrachistochrone(ContiniousTimeSymbolicModel): #!s=a #!s
    state_labels= ["$x$", "$y$", "bead speed"]
    action_labels = ['Tangent angle']

    def __init__(self, g=9.82, h=None, x_dist=1, simple_bounds=None, cost=None): #!s=a #!s
        self.g = g
        self.h = h
        self.x_dist = x_dist
        c0, b0, guess0 = brachistochrone(x_B=x_dist)

        if simple_bounds is None:
            simple_bounds = b0
        if cost is None:
            cost = c0

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))  #!s=a
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,))  #!s
        super(ContiniouBrachistochrone, self).__init__(cost=cost, simple_bounds=simple_bounds)
        if simple_bounds is None:
            x_B = 1
            simple_bounds = {'t0': Bounds([0], [0]),  #!s
                             'tF': Bounds([0], [np.inf]),
                             'x': Bounds([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                             'u': Bounds([-np.inf], [np.inf]),
                             'x0': Bounds([0, 0, 0], [0, 0, 0]),
                             'xF': Bounds([x_B, -np.inf, -np.inf], [x_B, np.inf, np.inf]) #!s
                             }
        if cost is None:
            cost = SymbolicQRCost(Q=np.zeros((3,3)), R = np.zeros((1,1)), qc=1.0)  # !b #!b Instantiate cost=SymbolicQRCost(...) here corresponding to minimum time. See the cartpole for hints

        super().__init__(cost=cost, simple_bounds=simple_bounds)


    def sym_f(self, x, u, t=None): #!f
        v = x[2]
        uu = u[0]
        xp = [v * sym.sin(uu), -v * sym.cos(uu), self.g * sym.cos(uu)]
        return xp

    def sym_h(self, x, u, t):
        '''
        Add a dynamical constraint of the form

        h(x, u, t) <= 0
        '''
        if self.h is None:
            return []
        else:
            # compute a single dynamical constraint as in \cite[Example (4.10)]{betts2010practical} (Note y-axis is reversed in the example)
            return [ -x[1] - x[0]/2 - self.h ] #!b #!b

def brachistochrone(x_B, g=9.82):
    '''

    i.e. simple_bounds={'t0': Bounds([0], [0]),
                        'tF': ...,
                        ...
                        }
    '''
    simple_bounds = {'t0': Bounds([0], [0]), #!s
                     'tF': Bounds([0], [np.inf]),
                     'x': Bounds([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf ]),
                     'u': Bounds([-np.inf], [np.inf]),
                     'x0': Bounds([0, 0, 0], [0, 0, 0]), #!s
                     }
    xF2 = Bounds([x_B, -np.inf, -np.inf], [x_B, np.inf, np.inf])
    simple_bounds['xF'] = xF2
    # cost = SymbolicQRCost(c=1.0) #!b #!b Instantiate cost=SymbolicQRCost(...) here corresponding to minimum time. See the cartpole for hints
    cost = None
    # Set up a guess
    guess = {'t0': 0,
             'tF': 2,
             'x': [np.asarray(simple_bounds['x0'].lb), np.asarray([x_B, x_B, 2])],
             'u': [np.asarray( [0] ), np.asarray( [1] )]}

    return cost, simple_bounds, guess


