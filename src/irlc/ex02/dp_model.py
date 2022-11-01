import numpy as np

class DPModel: #!s=dse #!s=dse
    """ The Dynamical Programming model class

    The purpose of this class is to translate a dynamical programming problem, defined by the equations,

    .. math::
        x_{k+1} = f_k(x_k, u_k, w_k)
        cost = g_k(x_k, u_k, w_k).
        terminal cost = g_N(x_N)

    into a python object which we can then use for planning.
    The above corresponds to the first 3 methods below. The rest of the functions specify the available states and actions
    (see definition of the basic problem in the lecture notes).
    """
    def __init__(self, N): #!s=dse
        self.N = N

    def f(self, x, u, w, k: int):
        raise NotImplementedError("Return f_k(x,u,w)")

    def g(self, x, u, w, k: int) -> float:
        raise NotImplementedError("Return g_k(x,u,w)")

    def gN(self, x) -> float:
        raise NotImplementedError("Return g_N(x)")

    def S(self, k: int):
        raise NotImplementedError("Return state space as set S_k = {x_1, x_2, ...}")

    def A(self, x, k: int):
        raise NotImplementedError("Return action space as set A(x_k) = {u_1, u_2, ...}")

    def Pw(self, x, u, k: int):
        """
        At step k, given x_k, u_k, compute the set of random noise disturbances w
        and their probabilities as a dict {..., w_i: pw_i, ...} such that

        pw_i = P_k(w_i | x, u)
        """
        return {'w_dummy': 1/3, 42: 2/3}  # P(w_k="w_dummy") = 1/3, P(w_k =42)=2/3. #!s=dse

    def w_rnd(self, x, u, k): #!s=w_rnd
        """ generate random disturbances w ~ P_k(x, u) (useful for simulation) """
        pW = self.Pw(x, u, k)
        w, pw = zip(*pW.items())  # seperate w and p(w)
        return np.random.choice(a=w, p=pw) #!s
