from irlc.ex04.model_boing import BoingEnvironment
import numpy as np
from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent, boing_experiment
# cvxpy is only used here.
# It can be a little annoying to install on Windows, and you don't need it for the project.
import cvxpy as cp


class MPCLearningAgentLocalOptimize(MPCLocalLearningLQRAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _solve(self, env, x0, A, B, d):
        # Example taken from: https://colab.research.google.com/github/cvxgrp/cvx_short_course/blob/master/intro/control.ipynb#scrollTo=WeoGIbrpb7zC
        T = self.NH  # Horizon length
        n, m = B[0].shape
        x = cp.Variable((n, T))  # Define the variables in the optimization problem. See example in link above.
        u = cp.Variable((m, T))

        cost = env.discrete_model.cost
        """
        Construct the cost function E here using the cvx optimization library.
        You should follow the link above and use the same idea, just with the cost.R, cost.Q and cost.q matrices/vectors.
        Make sure the cost function is a sum of T terms. 

        """
        E = sum([cost.q @ x[:, t] + 0.5 * cp.quad_form(u[:, t], cost.R) + 0.5 * cp.quad_form(x[:, t], cost.Q) for t in range(T)])  #!b #!b
        constr = [x[:, t + 1] == A[t] @ x[:, t] + B[t] @ u[:, t] + d[t] for t in range(T - 1)]

        def addc(space, w):
            I, J = np.isfinite(space.low), np.isfinite(space.high)
            return ([w[I, k] >= space.low[I] for k in range(w.shape[1])] if any(I) else []) \
                   + ([w[J, k] <= space.high[J] for k in range(w.shape[1])] if any(J) else [])

        constr += [x[:, 0] == x0] + addc(env.observation_space, x[:, 1:]) + addc(env.action_space, u)
        problem = cp.Problem(cp.Minimize(E), constr)
        problem.solve()
        x_star = [x.value[:, k] for k in range(T)]
        u_star = [u.value[:, k] for k in range(T)]

        if np.max(np.abs(u_star[0])) > np.max(np.abs(env.action_space.low)):
            print("bad!")  # Should probably raise an Exception here.

        return x_star, u_star


class MPCLocalAgentExactDynamics(MPCLearningAgentLocalOptimize):
    """ Bonus agent that uses approximate system matrices obtained using exact dynamics, i.e. iLQR. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ABd(self):
        # obtain system matrices.
        A, B, d = [], [], []
        for l in range(self.NH):
            # build quadratic problem
            f, Jx, Ju = self.env.discrete_model.f(self.x_bar[l], self.u_bar[l], 0, compute_jacobian=True)
            dd = f - Jx @ self.x_bar[l] - Ju @ self.u_bar[l]
            dA = Jx
            dB = Ju
            A.append(dA), B.append(dB), d.append(dd)
        return A, B, d


def learning_optimization_mpc_local(env):
    # Learning the dynamics and apply LQR, but train on a short horizon. This method implements \nref{alg13estimD}
    lagent3 = MPCLearningAgentLocalOptimize(env, neighbourhood_size=50)
    boing_experiment(env, lagent3, pdf="ex7_D", num_episodes=4)


if __name__ == "__main__":
    env = BoingEnvironment(output=[10, 0])
    # Part D: Optimization+MPC and local regression
    learning_optimization_mpc_local(env)
