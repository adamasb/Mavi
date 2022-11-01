"""
Implements the inventory-control problem from \cref{c2inventory}. See todays slides if you are stuck!
"""
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic

class InventoryDPModel(DPModel): #!s=a
    def __init__(self, N=3):
        super().__init__(N=N)

    def A(self, x, k): # Action space A_k(x) #!f
        return {0, 1, 2}

    def S(self, k): # State space S_k #!f
        return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w) #!f
        return u + (x + u - w) ** 2

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w) #!f
        return max(0, min(2, x + u - w ))

    def Pw(self, x, u, k): # Distribution over random disturbances #!f
        return {0:.1, 1:.7, 2:0.2}

    def gN(self, x): #!f
        return 0 #!s

def main():
    inv = InventoryDPModel() #!s=b #!o=a
    J,pi = DP_stochastic(inv)
    print(f"Inventory control optimal policy/value functions")
    for k in range(inv.N):
        print(", ".join([f" J_{k}(x_{k}={i}) = {J[k][i]:.2f}" for i in inv.S(k)] ) )
    for k in range(inv.N):
        print(", ".join([f"pi_{k}(x_{k}={i}) = {pi[k][i]}" for i in inv.S(k)] ) ) #!o #!s

if __name__ == "__main__":
    main()