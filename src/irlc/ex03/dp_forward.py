from irlc.ex03.search_problem import GraphSP, DP2SP, EnsureTerminalSelfTransitionsWrapper
from irlc.ex02.graph_traversal import SmallGraphDP
import numpy as np

def dp_forward(sp, N):
    """
    Implement the forward DP algorithm as described in \nref{dpgraph4}
    """
    J = [{} for _ in range(N + 2)]
    pi = [{} for _ in range(N + 1)]
    J[0][sp.initial_state] = 0

    for k in range(0, N + 1):
        for xi in J[k].keys():
            for a, (xj, c) in sp.available_transitions(xi).items():  #!f
                """
                Implement the forward-DP check and update rule here. 
                There is one tricky thing, namely how to get the path from the policy; you can consider how you would do this, but 
                for now you will just get the solution: When the policy is updated (as in the algorithm) simply use this code

                pi[k][xj] = (a,xi). 
                The post-processing below will take care of the rest. 
                The J-function is updated as normal: 

                J[k+1][xj] = .... 
                """
                if xj not in J[k + 1] or c + J[k][xi] < J[k + 1][xj]:
                    J[k + 1][xj] = c + J[k][xi]
                    pi[k][xj] = (a, xi)

    # The final states obtained are the keys in J[N+1]. Find one of these which is also a terminal state for the search problem
    terminal_state = None
    best_terminal_cost = np.inf
    for xN, cost in J[N + 1].items():
        if sp.is_terminal(xN) and cost < best_terminal_cost:
            terminal_state = xN
            best_terminal_cost = cost

    if terminal_state is None:
        raise Exception("The sp problem is misspecified; S_{N+1} should contain a terminal state, but J[N+1] was" + str(J[N + 1]))

    # Create the path from start to end; this is the post-processing step.
    actions = []
    path = [terminal_state]
    for k in range(N + 1):
        a, x_prev = pi[N - k][path[0]]
        actions = [a] + actions
        path = [x_prev] + path
    return J, actions, path


def search_partA():
    """
    Part 1a: Test forward DP algorithm. Find optimal path from s=2 to t=5 in exactly N=4 steps.
    """
    t = 5  #!s=sp_run
    s = 2
    sp = GraphSP(start=s, goal=t)
    N = len(set([i for edge in sp.G for i in edge])) - 1  # N = (Unique vertices) - 1
    J_sp, pi_sp, path = dp_forward(sp, N)
    print(f"GraphSP> Optimal cost from {s} -> {t}:", J_sp[-1][t])
    print(f"GraphSP> Optimal path from {s} -> {t}:", path, "\n")  #!s=sp_run


def search_partB():
    """
       Part 1b: The above code should give a suboptimal cost of J[N][t] = 5.5; this is because
       the DP algorithm searches for a path of exactly length N=4, while the shortest path have length N=3.
       We can fix this by adding a terminal self-transition t->t of cost 0 (see \nref{c2s21}). However, rather than modifying the search
       problem, we can do this more generally using a wrapper class. Review the code to check out how it works.
       """
    s, t = 2, 5
    sp = GraphSP(start=s, goal=t)
    N = len(set([i for edge in sp.G for i in edge])) - 1  # N = (Unique vertices) - 1
    sp_wrapped = EnsureTerminalSelfTransitionsWrapper(sp)  #!s=wrapped
    J_wrapped, pi_wrapped, path_wrapped = dp_forward(sp_wrapped, N)
    print(f"GraphSP[Wrapped]> Optimal cost from {s} -> {t}:", J_wrapped[-1][t])
    print(f"GraphSP[Wrapped]> Optimal path from {s} -> {t}:", path_wrapped, "\n")  #!s


def search_partC():
    """
    Part 2: Convert a DP problem into search problem, then solve it with forward DP
    """
    s, t = 2, 5
    env = SmallGraphDP(t=t)  #!s=sp_run2
    sp_env = DP2SP(env, initial_state=s)
    J2, pi2, path = dp_forward(sp_env, env.N)
    print(f"DP2SP> Optimal cost from {s} -> {t}:", J2[-1][sp_env.terminal_state])  #!o=sp2
    print(f"DP2SP> Optimal path from {s} -> {t}:", path)  #!s=sp_run2  #!o


if __name__ == '__main__':
    search_partA()  #!o=sp
    search_partB()  #!o
    search_partC()
