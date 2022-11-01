import sympy as sym
from irlc.ex04.cost_discrete import DiscreteQRCost
import numpy as np
from irlc.ex04.cost_discrete import targ2matrices

class SymbolicMayerLagrangeCost:
    """
    Symbolic MayerLagrange cost function. The environment is assumed to terminate at time t=t_f
    and have cost:

    J(x_0, t_0) = cf(x(t_f), t_f) + \int_{t_0}^{t_F} c(x(t), u(t), t) dt

    (if the environment does not terminate, simply let cf=0). We specify these as symbolic expressions
    to allow us to compute derivatives later.
    """
    def sym_cf(self, t0, tF, x0, xF):
        # compute Mayer term
        raise NotImplementedError()

    def sym_c(self, x, u, t):
        # compute Lagrange term
        raise NotImplementedError()

def mat(x):
    return sym.Matrix(x) if x is not None else x

# def sym2np(x):
#     if x is None:
#         return x
#     f = sym.lambdify([], x)
#     return f()


# class SymbolicQRCost2(SymbolicMayerLagrangeCost):
#     def __init__(self, env=None, state_size=-1, action_size=-1, Q=None,R=None,H=None,q=None,r=None,qc=0, QN=None, qN=None,qcN=0):
#
#         self._x = QuadraticForm(state_size, Q=Q, q=q, qc=qc)
#         self._u = QuadraticForm(action_size, Q=R, q=r, qc=0)
#         self.H = H
#
#         pass
#     pass

class SymbolicQRCost(SymbolicMayerLagrangeCost):
    def __init__(self, Q, R, q=None, qc=None, r=None, H=None, QN=None, qN=None, qcN=None):

        # assert Q.shape[0] == Q.shape[1]
        # assert R.shape[0] == R.shape[1]

        n = Q.shape[0]
        d = R.shape[0]
        self.Q = Q
        self.R = R
        self.q = np.zeros( (n,)) if q is None else q
        self.qc = 0 if qc == None else qc
        self.r = np.zeros( (d,)) if r is None else r
        self.H = np.zeros((d,n)) if H is None else H
        self.QN = np.zeros((n,n)) if QN is None else QN
        self.qN = np.zeros((n,)) if qN is None else qN
        self.qcN = 0 if qcN == None else qcN
        self.flds = ('Q', 'R', 'q', 'qc', 'r', 'H', 'QN', 'qN', 'qcN')
        self.flds_term = ('QN', 'qN', 'qcN')


    @classmethod
    def zero(cls, state_size, action_size):
        return cls(Q=np.zeros( (state_size,state_size)), R=np.zeros((action_size,action_size)) )

    def sym_cf(self, t0, tF, x0, xF):
        xF = sym.Matrix(xF)
        c = 0.5 * xF.transpose() @ self.QN @ xF + xF.transpose() @ sym.Matrix(self.qN) + sym.Matrix([[self.qcN]])
        assert c.shape == (1,1)
        return c[0,0]


    def sym_c(self, x, u, t):
        '''
        Implements:
        w = 0.5 * x' * Q * x + q' @ x + qc + 0.5 * u' * R * u + r' @ u
        '''
        u = sym.Matrix(u)
        x = sym.Matrix(x)
        c =  1 / 2 * (x.transpose() @ self.Q @ x) + 1 / 2 * (u.transpose() @ self.R @ u) + u.transpose() @ self.H @ x + sym.Matrix(self.q).transpose() @ x + sym.Matrix(self.r).transpose() @ u + sym.Matrix([[self.qc]])
        assert c.shape == (1,1)
        return c[0,0]


    def discretize(self, env, dt):
        """ Discreteize the cost function. Note not all terms are discretized; it is good enough for this course, but
        it would be worth re-visiting it later if the examples are extended. """
        return DiscreteQRCost(env=env, **{f: self.__getattribute__(f) * (1 if f in self.flds_term else dt) for f in self.flds} )


    def __add__(self, c):
        return SymbolicQRCost(**{k: self.__dict__[k] + c.__dict__[k] for k in self.flds})

    def __mul__(self, c):
        return SymbolicQRCost(**{k: self.__dict__[k] * c for k in self.flds})

    def goal_seeking_terminal_cost(self, xN_target, QN=None):
        if QN is None:
            QN = np.eye(xN_target.size)
        QN, qN, qcN = targ2matrices(xN_target, Q=QN)
        return SymbolicQRCost(Q=QN*0, R=self.R*0, QN=QN, qN=qN, qcN=qcN)

    def goal_seeking_cost(self, x_target, Q=None):
        if Q is None:
            Q = np.eye(x_target.size)
        Q, q, qc = targ2matrices(x_target, Q=Q)
        return SymbolicQRCost(Q=Q, R=self.R*0, q=q, qc=qc)

    def term(self, Q=None, R=None,r=None):
        dd = {}
        lc = locals()
        for f in self.flds:
            if f in lc and lc[f] is not None:
                dd[f] = lc[f]
            else:
                dd[f] = self.__getattribute__(f)*0
        return SymbolicQRCost(**dd)


# def targ2matrices(t, Q=None):
#     """
#     1/2 * (x - t)**2 = 1/2 * x' * x + 1/2 * t' * t - x * t
#     """
#     n = t.size
#     if Q is None:
#         Q = np.eye(n)
#     return Q, -1/2 * (Q @ t + t @ Q), 1/2 * t @ Q @ t


# def goal_seeking_continious_qr_cost(Q, R, x_target=None, QN=None, xN_target=None):
#     cost = SymbolicQRCost(Q=Q,R=R)
#     # R = np.zeros((env.action_size,env.action_size))
#     # cost = None
#     if x_target is not None:
#         Q,q,qc = targ2matrices(x_target,Q=Q)
#         cost = cost + SymbolicQRCost(Q = Q, R=R,q=q, qc=qc)
#
#     if xN_target is not None:
#         QN, qN, qcN = targ2matrices(xN_target, Q=QN)
#         cost += SymbolicQRCost(Q=Q*0, R=R, QN=QN, qN=qN, qcN=qcN)
#
#     return cost

""" 
class QuadraticForm:
    def __init__(self, d, Q=None, q=None, qc=None):
        self.Q = np.zeros( (d,d) ) if Q is None else Q
        self.q = np.zeros((d, )) if q is None else q
        self.qc = 0 if qc is None else qc
        self.d = d
        self.flds = {'Q', 'q', 'qc'}

    def sym_c(self, x):
        return self.qc + x @ sym.Matrix(self.q) + 0.5 * x.transpose() @ sym.Matrix(self.Q) @ x

    def c(self, x):
        return self.qc + x @ self.q + 0.5 * x.T @ self.Q @ x

    @classmethod
    def goal_seeking(cls, Q, x_target):
        return cls(len(x_target), Q=Q, q=-x_target @ Q, qc=0.5 * x_target.T @ Q @ x_target)

    def __add__(self, c):
        return QuadraticForm(d=self.d, **{k: self.__dict__[k] + c.__dict__[k] for k in self.flds})

    def __mul__(self, c):
        return QuadraticForm(d=self.d, **{k: self.__dict__[k] * c for k in self.flds})
"""