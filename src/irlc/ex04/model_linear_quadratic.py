import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from gym.spaces.box import Box
import numpy as np

class LinearQuadraticModel(ContiniousTimeSymbolicModel):
    """
    Implements a model with update equations

    dx/dt = Ax + Bx + d
    Cost = integral_0^{t_F} (1/2 x^T Q x + 1/2 u^T R u + q' x + qc) dt
    """
    def __init__(self, A, B, Q, R, q=None, qc=None, d=None):  #!s=a #!s
        cost = SymbolicQRCost(R=R, Q=Q, q=q, qc=qc)
        self.A, self.B, self.d = A, B, d
        self.observation_space = Box(high=np.inf, low=-np.inf, shape=(A.shape[0],), dtype=float)
        self.action_space = Box(high=np.inf, low=-np.inf, shape=(B.shape[1],), dtype=float)
        super().__init__(cost=cost)

    def sym_f(self, x, u, t=None):  #!s=a
        xp = sym.Matrix(self.A) * sym.Matrix(x) + sym.Matrix(self.B) * sym.Matrix(u)
        if self.d is not None:
            xp += sym.Matrix(self.d)
        return [x for xr in xp.tolist() for x in xr]  #!s

    def reset(self):
        x0 = np.zeros_like(self.observation_space.low)
        return x0
