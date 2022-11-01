import numpy as np
from irlc.ex02.dp_model import DPModel
from irlc.ex02.graph_traversal import symG
from irlc.ex03.search_problem import DP2SP
from irlc.ex03.dp_forward import dp_forward

Gtravelman = {("A", "B"): 5, ("A", "C"): 1, ("A", "D"): 15, ("B", "C"): 20, ("B", "D"): 4, ("C", "D"): 3}
symG(Gtravelman)  # make graph symmetric.

class TravelingSalesman(DPModel):
    """ Travelling salesman problem, see \nref{c5travelman}
    Visit all nodes in the graph with the smallest cost, and such that we end up where we started.
    The actions are still new nodes we can visit, however the states have to be the current path.

    I.e. the first state is s = ("A", ) and then the next state could be s = ("A", "B"), and so on.
    """
    def __init__(self):
        self.G = Gtravelman
        self.cities = {c for chord in self.G for c in chord}
        N = len(self.cities)
        super(TravelingSalesman, self).__init__(N)

    def f(self, x, u, w, k):
        assert((x[-1],u) in self.G)
        return x + (u,) #!b #!b

    def g(self, x, u, w, k): #!f
        return self.G[x[-1], u]

    def gN(self, x): #!f
        """
        Win condition is that:

        (1) We end up where we started AND
        (2) we saw all cities AND
        (3) all cities connected by path

        If these are met return 0 otherwise np.inf
        """
        path_ok = x[0] == x[-1] and len(set(x)) == len(self.cities) and all([(x[i],x[i+1]) in self.G for i in range(len(x)-1)])
        return 0 if path_ok else np.inf

    def A(self, x, k):
        return {b for (a, b) in self.G if x[-1] == a}

def main():
    tm = TravelingSalesman()
    s = ("A",)
    tm_sp = DP2SP(tm, s)
    J, actions, path = dp_forward(tm_sp, N=tm.N)
    #!o
    print("Cost of optimal path (should be 13):", J[-1][tm_sp.terminal_state])
    print("Optimal path:", path)
    print("(Should agree with \nref{c5travelman})")  #!o

if __name__ == "__main__":
    main()