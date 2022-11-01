from irlc.ex06.dlqr import LQR
from irlc.ex07.regression import solve_linear_problem_simple
from irlc.ex04.model_boing import BoingEnvironment
from irlc.ex01.agent import train
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from irlc import Agent
from irlc import Timer

class Buffer: #!s=a #!s
    def __init__(self):
        self.x = []
        self.u = []
        self.xp = []

    def push(self, x, u, xp): #!s=a
        """ Add an observation of the form
        > xp = f(x, u) to the buffer.
        """
        self.x.append(x)
        self.u.append(u)
        self.xp.append(xp)

    def __len__(self):
        return len(self.x)

    def get_data(self):
        """ Return matrices (vertical dimension are number of samples) of the form

        > XP[i,:].T = f(X[i,:].T, U[i,:].T)

        The matrices will consist of all data in the buffer.
        """
        X = np.asarray(self.x)  # train new LQR
        XP = np.asarray(self.xp)
        U = np.asarray(self.u)
        return X, U, XP #!s

    def get_closest_observations(self, x, u, n=50):
        """ Given x, u (as vectors) the code finds the n closest observations to (x, u), i.e. observations of the form (x_k, u_k, x_{k+1}), such that
        the distances |(x_k, u_k)-(x_k', u_k')| is as small as possible. This can be used for local linear regression. """
        if len(self) < n:
            return self.get_data()
        X,U,XP = self.get_data()
        TT = 1
        NN_WITH_U = True
        for _ in range(TT):
            Z = np.concatenate([X,U],axis=1) if NN_WITH_U else X
            nbrs = NearestNeighbors(n_neighbors=n, metric='euclidean', algorithm='auto').fit(Z)
            z = (np.concatenate([x,u],axis=0) if NN_WITH_U else x ).reshape( (1,-1))
            distances, indices = nbrs.kneighbors(  z)
            indices = indices.squeeze()
            xx, uu,xxp = X[indices], U[indices], XP[indices]
        return xx, uu, xxp


class LearningLQRAgent(Agent):
    def __init__(self, env):
        self.buffer = Buffer()
        self.L = None
        self.l = None
        self.dt = env.dt
        self.lamb = 0.001 # Lambda regularization parameter for regression.
        super().__init__(env)

    def pi(self, x, t=None):
        if t == 0 and len(self.buffer) > 0: # Don't plan if the buffer is empty.
            N = int(self.env.Tmax / self.env.dt) # Horizon length for LQR planning (i.e. problem length)
            """ Re-plan self.L, self.l using LQR. To do so:
            > Get data from buffer
            > Fit the linear problem (A, B, d) using the function solve_linear_problem_simple(..., lambd=self.lambd) (see regression.py for comments). The data is assumed to be in the format [dims x samples]. 
            > (self.L, self.l), (V,v,vc) = LQR(...) # Apply LQR to get control matrices. 
            When you apply LQR, you only need to supply the cost-terms cost.Q, cost.q, cost.qc and cost.R. 
            """
            X = np.asarray(self.buffer.x) #!b
            Y = np.asarray(self.buffer.xp)
            U = np.asarray(self.buffer.u)
            cost = self.env.discrete_model.cost
            A, B, C = solve_linear_problem_simple(Y=Y.T, X=X.T, U=U.T, lamb=self.lamb)
            (self.L, self.l), (V, v, vc) = LQR(A=[A] * N, B=[B] * N, d=[C]*N, Q=[cost.Q] * N, R=[cost.R] * N, q=[cost.q] * N, qc=[cost.qc] * N) #!b

        if self.L is None:
            return self.env.action_space.sample() # There are no control matrices. Use a random action.
        else:
            # Compute action u based on control matrices self.L, self.l (see the LQR agent)
            k = int(t / self.dt)  # current timepoint #!b
            u = self.L[k] @ x + self.l[k] #!b Compute action u here using control matrices stored in self.L, self.l.
            return u

    def train(self, x, u, reward, xp, done=False):
        # Push the  current observation into the buffer. See buffer documentation for details.
        self.buffer.push(x=x, u=u, xp=xp) #!b #!b

class MPCLearningAgent(Agent):
    def __init__(self, env, horizon_length=30):
        self.buffer = Buffer()
        self.horizon_length = horizon_length
        self.neighbourhood_size = 100
        self.dt = env.dt
        self.lamb = 0.00001 # Very small regularization parameter for the linear regression.
        super().__init__(env)

    def pi(self, x, t=None):
        if len(self.buffer) < 10: # If buffer is very small do random actions.
            return self.env.action_space.sample()
        else:
            """ Compute control matrices self.L, self.l here by 
            (1) getting data from buffer
            (2) fitting a regression model to the data (as before)
            (3) apply LQR to the system matrices obtained in (2). 
            """
            X,U,XP = self.buffer.get_data() #!b
            N = self.horizon_length
            cost = self.env.discrete_model.cost
            A, B, C = solve_linear_problem_simple(Y=XP.T, X=X.T, U=U.T, lamb=self.lamb)
            (self.L, self.l), (V, v, vc) = LQR(A=[A] * N, B=[B] * N, d=[C]*N, Q=[cost.Q] * N, R=[cost.R] * N, q=[cost.q]*N, qc=[cost.qc]*N) #!b
            u = self.L[0] @ x + self.l[0]
            return u

    def train(self, x, u, cost, xp, done=False, metadata=None):
        self.buffer.push(x=x, u=u, xp=xp) #!b #!b Push current observation into the buffer. See buffer documentation for details.

class MPCLocalLearningLQRAgent(Agent):
    def __init__(self, env, horizon_length=30, neighbourhood_size=50, min_buffer_size=40):
        self.buffer = Buffer()
        self.NH = horizon_length
        self.neighbourhood_size = neighbourhood_size
        self.x_bar, self.u_bar = None, None
        self.min_buffer_size = min_buffer_size
        self.timer = Timer()
        super().__init__(env)

    def _solve(self, env, x0, A, B, d):
        """
        Helper function 'solve' in which solves for L, l using LQR, then computes x_bar, u_bar (see \nref{alg13estimC}).

        """
        cost = env.discrete_model.cost # use the LQR cost. You only need the terms Q, q and R (no terminal terms) when you call LQR.
        # When you call LQR to get the control matrices L and l, set mu=1e-6 (Note completely sure this is required).
        (L, l), (V, v, vc) = LQR(A=A, B=B, d=d, Q=[cost.Q] * self.NH, R=[cost.R] * self.NH, q=[cost.q]*self.NH, qc=[cost.qc]*self.NH,mu=1e-6)
        x_bar = [] # Compute x_bar, u_bar as lists.
        u_bar = []
        xl = x0 #!b
        for k in range(self.NH):
            x_bar.append(xl)
            u_bar.append(L[k] @ xl + l[k])
            xl = A[k] @ x_bar[k] + B[k] @ u_bar[k] + d[k] #!b
        return x_bar, u_bar

    def get_ABd(self):
        A, B, d = [], [], []
        for l in range(self.NH):
            self.timer.tic("Nearest observations")
            # Get the nearest observations to x_bar[l], u_bar[l] here.
            X, U, XP = self.buffer.get_closest_observations(self.x_bar[l], self.u_bar[l], n=self.neighbourhood_size)  #!b #!b
            self.timer.toc()
            self.timer.tic("Solve for A,B,d")
            lamb = 0.00001
            # Perform local linear problem using the solve_linear_problem_simple helper method as before. This should give you the matrices relevant for time
            # x_bar[l] as dA, dB, dd
            dA, dB, dd = solve_linear_problem_simple(Y=XP.T, X=X.T, U=U.T, lamb=lamb) #!b #!b
            self.timer.toc()
            A.append(dA), B.append(dB), d.append(dd)
        return A,B,d

    def pi(self, x, t=None):
        if len(self.buffer) < self.min_buffer_size:
            return self.env.action_space.sample()
        else:
            if self.x_bar is None:
                # Initialize x_bar
                self.x_bar, self.u_bar = [x] * self.NH, [self.env.action_space.sample()] * self.NH

            # Perform the shuffle-step for self.x_bar, self.u_bar.
            """
            self.x_bar = ...
            self.u_bar = ...
            """
            self.x_bar = self.x_bar[1:] + [self.x_bar[-1]] #!b
            self.u_bar = self.u_bar[1:] + [self.u_bar[-1]] #!b
            # Perform the rest of the planning operations.
            A,B,d = self.get_ABd()
            self.timer.tic("Solve for x-bar, u-bar")
            self.x_bar, self.u_bar = self._solve(self.env, x0=x, A=A, B=B, d=d)
            self.timer.toc()
            u = self.u_bar[0] # the optimal action.
            return u

    def train(self, x, u, reward, xp, done=False, metadata=None):
        self.buffer.push(x=x, u=u, xp=xp)


def boing_experiment(env, agent, num_episodes=2, plot=True, pdf=None):
    """ Train the agent for num_episodes of data and plot the result on each trajectory """
    stats, trajectories= train(env,agent,num_episodes=num_episodes, return_trajectory=True)
    def plot_trajectory(t):
        ss = t.state
        P = env.discrete_model.continuous_model.P
        airspeed =(P @ ss.T)[0]
        climbrate = (P @ ss.T)[1]

        plt.plot(t.time, airspeed, label='airspeed u')
        plt.plot(t.time, climbrate, label='climb rate')
        plt.plot(t.time[:-1], t.action[:,0], label="Elevator e")
        plt.plot(t.time[:-1], t.action[:,1], label="Throttle t")

        plt.xlabel("Time/s")
        plt.grid()
        plt.legend()

    if hasattr(agent, '_t_solver'):
        tt = [agent._t_nearest, agent._t_linearizer, agent._t_solver]
        print("Nearest, linear, solver: ", [t/sum(tt) for t in tt])
    if plot:
        f,axs = plt.subplots(len(trajectories), 1, sharey=True, figsize = (10, 10))

        for k, t in enumerate(trajectories):
            plt.sca(axs[k])
            plot_trajectory(t)
            plt.title(f"Trajectory {k}")
        if pdf is not None:
            from irlc import savepdf
            savepdf(pdf)
        plt.show()
    return stats, trajectories

def learning_lqr(env):
    # Learn the dynamisc and apply LQR.
    lagent = LearningLQRAgent(env)
    boing_experiment(env, lagent, pdf="ex7_A", num_episodes=3)

def learning_lqr_mpc(env):
    # Learning the dynamics and apply LQR, but train on a short horizon. This method implements \nref{alg13estimB}
    lagent2 = MPCLearningAgent(env)
    boing_experiment(env, lagent2, num_episodes=3, pdf="ex7_B")

def learning_lqr_mpc_local(env):
    # Learning the dynamics and apply LQR, but train on a short horizon. This method implements \nref{alg13estimC}
    lagent3 = MPCLocalLearningLQRAgent(env, neighbourhood_size=50)
    boing_experiment(env, lagent3, pdf="ex7_C", num_episodes=4)


if __name__ == "__main__":
    env = BoingEnvironment(output=[10, 0])

    # Part A: LQR and global regression
    learning_lqr(env)

    # Part B: LQR+MPC
    learning_lqr_mpc(env)

    # Part C: LQR+MPC and local regression
    learning_lqr_mpc_local(env)
