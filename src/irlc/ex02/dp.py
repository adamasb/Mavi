from irlc.ex02.graph_traversal import SmallGraphDP
from irlc.ex02.graph_traversal import policy_rollout

def DP_stochastic(model):
    """
    Implement the stochastic DP algorithm. The implementation follows \nref{c4dpalg}.
    In case you run into problems, I recommend following the hints in \nref{c4sgraphex} and focus on the
    case without a noise term; once it works, you can add the w-terms. When you don't loop over noise terms, just specify
    them as w = None in env.f and env.g.
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        for x in model.S(k):
            """
            Update pi[k][x] and Jstar[k][x] using the general DP algorithm given in \nref{c4dpalg}.
            If you implement it using the pseudo-code, I recommend you define Q as a dictionary like the J-function such that
                        
            > Q[u] = Q_u (for all u in model.A(x,k))
            Then you find the u where Q_u is lowest, i.e. 
            > umin = arg_min_u Q[u]
            Then you can use this to update J[k][x] = Q_umin and pi[k][x] = umin.
            """
            Qu = {u: sum(pw * (model.g(x, u, w, k) + J[k + 1][model.f(x, u, w, k)]) for w, pw in model.Pw(x, u, k).items()) for u in model.A(x, k)} #!b
            umin = min(Qu, key=Qu.get)
            J[k][x] = Qu[umin]
            pi[k][x] = umin #!b
            """
            After the above update it should be the case that:

            J[k][x] = J_k(x)
            pi[k][x] = pi_k(x)
            """
    return J, pi


if __name__ == "__main__":  # Test dp on small graph given in \nref{c4sgraphex}
    print("Testing the deterministic DP algorithm on the small graph yafcport")
    model = SmallGraphDP(t=5)  # Instantiate the small graph with target node 5 #!s
    J, pi = DP_stochastic(model)
    # Print all optimal cost functions J_k(x_k) #!o=a
    for k in range(len(J)):
        print(", ".join([f"J_{k}({i}) = {v:.1f}" for i, v in J[k].items()]))
    s = 2  # start node
    J,xp = policy_rollout(model, pi=lambda x, k: pi[k][x], x0=s)
    print(f"Actual cost of rollout was {J} which should obviously be similar to J_0[{s}]")
    print(f"Path was", xp)  #!s #!o
    # Remember to check optimal path agrees with the the (self-evident) answer from the figure.
