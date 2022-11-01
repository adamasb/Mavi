import numpy as np
from irlc.ex02.dp_model import DPModel

"""
Graph of shortest path problem of \nref{c2smallgraph}
"""
G222 = {(1, 2): 6,  (1, 3): 5, (1, 4): 2, (1, 5): 2,  #!s=g222
        (2, 3): .5, (2, 4): 5, (2, 5): 7,
        (3, 4): 1,  (3, 5): 5, (4, 5): 3}  #!s

def symG(G):
    """ make a graph symmetric. I.e. if it contains edge (a,b) with cost z add edge (b,a) with cost c """
    G.update({(b, a): l for (a, b), l in G.items()})
symG(G222)

class SmallGraphDP(DPModel):
    """ Implement the small-graph example in \nref{c2smallgraph}. t is the terminal node. """
    def __init__(self, t, G=None):  #!s=init
        self.G = G.copy() if G is not None else G222.copy()  #!s=init
        self.G = self.G.copy()  # Copy G. This is good style since G is passed by reference & modified in place.
        self.G[(t,t)] = 0  # make target vertex absorbing  #!s=init
        self.t = t         # target vertex in graph
        self.nodes = {i for k in self.G for i in k} # set of all nodes
        super(SmallGraphDP, self).__init__(N=len(self.nodes)-1)  #!s=init

    def f(self, x, u, w, k):
        if (x,u) in self.G:  #!f
            return u
        else:
            raise Exception("Nodes are not connected")

    def g(self, x, u, w, k): #!f
        return self.G[(x,u)]

    def gN(self, x):  #!f
        return 0 if x == self.t else np.inf

    def S(self, k):   #!s=A
        return self.nodes

    def A(self, x, k):
        return {j for (i,j) in self.G if i == x} #!s

def pi_silly(x, k): #!s=a
    if x == 1:
        return 2
    else:
        return 1 #!s

def pi_inc(x, k): #!f
    return x+1

def pi_smart(x,k): #!f
    return 5

def policy_rollout(model, pi, x0):
    """
    Given an environment and policy, should compute one rollout of the policy and compute
    cost of the obtained states and actions. In the deterministic case this corresponds to

    J_pi(x_0)

    in the stochastic case this would be an estimate of the above quantity.

    Note I am passing a policy 'pi' to this function. The policy is in this case itself a function, that is,
    you can write code such as

    > u = pi(x,k)

    in the body below.
    """
    J, x, trajectory = 0, x0, [x0]
    for k in range(model.N):
        u = pi(x, k) #!b #!b Generate the action u = ... here using the policy
        w = model.w_rnd(x, u, k) # This is required; just pass them to the transition function
        J += model.g(x, u, w, k) #!b
        x = model.f(x, u, w, k) #!b Update J and generate the next value of x.
        trajectory.append(x) # update the trajectory
    J += model.gN(x) #!b #!b Add last cost term env.gN(x) to J.
    return J, trajectory

def main():
    t = 5  # target node
    model = SmallGraphDP(t=t)
    x0 = 1  # starting node
    print("Cost of pi_silly", policy_rollout(model, pi_silly, x0)[0]) #!o=a
    print("Cost of pi_inc", policy_rollout(model, pi_inc, x0)[0])
    print("Cost of pi_smart", policy_rollout(model, pi_smart, x0)[0])  #!o

if __name__ == '__main__':
    main()
