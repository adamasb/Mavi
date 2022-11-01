from scipy.optimize import Bounds
import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from gym.spaces.box import Box
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np
from irlc import train, Agent, savepdf

"""
This class implements the Kuramoto model given by the differential equation:

> dx/dt = u + cos(x)

"""
class ContiniousKuramotoModel(ContiniousTimeSymbolicModel): #!s=m1
    def __init__(self):
        self.action_space = Box(low=np.array([-2]), high=np.array([2]), dtype=float)
        self.observation_space = Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=float) #!s=m1
        """ 
        Create a cost-object. It is less important what this does in details; from a user-perspective,
        it sets up a quadratic cost (with the given matrices) and allows easy computation of derivatives, etc.
        There are automatic ways to discretize the cost so you don't have to bother with that.
        You can see the environments in the toolbox for specific examples.
        """
        cost = SymbolicQRCost(R=np.ones( (1,1) ), Q=np.zeros((1,1))) #!s=m1 #!s=m1
        """  
        This call is important: This is where the symbolic function self.sym_f is turned into a numpy function
        Automatically. Although what happens in the super class is a little more complicated than most of the code
        we have seen so far, I still recommend looking at it for reference. 
        """
        super().__init__(cost=cost, simple_bounds=None) #!s=m1

    def reset(self):
        """ Return the starting position as a numpy ndarray. In this x=0 """
        return np.asarray([0])

    def sym_f(self, x, u, t=None): #!s=m1
        """ Return a symbolic expression representing the Kuramoto model.
        The inputs x, u are themselves *lists* of symbolic variables (insert breakpoint and check their value).
        you have to use them to create a symbolic object representing f, and return it as a list. That is, you are going to return
        > return [f(x,u)]
        where f is the symbolic expression. Note you can use trigonometric functions like sym.cos. 
        """
        symbolic_f_expression = [u[0] + sym.cos(x[0])] #!b #!b Implement symbolic expression as a singleton list here
        return symbolic_f_expression #!s=m1 #!s=m1

class DiscreteKuramotoModel(DiscretizedModel): #!s=m2
    """ Create a discrete version of the Kuramoto environment.
    The superclass will automatically Euler discretize the continious model (time constant 0.5) and set up useful functionality.
    Note many classes overwrite part of the functionality here to suit the needs of the environment. """
    def __init__(self, dt=0.5):
        model = ContiniousKuramotoModel()
        super().__init__(model=model, dt=dt) #!s=m2

class KuramotoEnvironment(ContiniousTimeEnvironment): #!s=m3
    """ Turn the whole thing into an environment. The step()-function in the environment will use *exact* RK4 simulation.
    and automaticcally compute the cost using the cost-function.
    """
    def __init__(self, Tmax=5, dt=0.5):
        discrete_model = DiscreteKuramotoModel(dt)
        super().__init__(discrete_model, Tmax=Tmax) #!s=m3


def f(x, u):
    """ Implement the kuramoto osscilator model's dynamics, i.e. f such that dx/dt = f(x,u).
    The answer should be returned as a singleton list. """
    cmodel = ContiniousKuramotoModel()
    f_value = cmodel.f(x, u, t=0) #!b #!b
    # Use the ContiniousKuramotoModel to compute f(x,u). If in doubt, insert a breakpoint and let pycharms autocomplete
    # guide you. See my video to Exercise 2 for how to use the debugger. Don't forget to specify t (for instance t=0).
    # Note that sympys error messages can be a bit unforgiving.
    return f_value

def fk(x,u):
    """ Computes the discrete (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,i.e.

    x_{k+1} = f_k(x,u).

    Look at dmodel.f for inspiration. As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a numpy ndarray.
    """
    dmodel = DiscreteKuramotoModel(dt=0.5)
    f_euler = dmodel.f(x, u, k=0)  #!b #!b Compute Euler discretized dynamics here using the dmodel.
    return f_euler

def dfk_dx(x,u):
    """ Computes the derivative of the (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,
    i.e. if

    x_{k+1} = f_k(x,u)

    this function should return

    df_k/dx

    (i.e. the Jacobian with respect to x) as a numpy matrix.
    Look at dmodel.f for inspiration, and note it has an input argument that is relevant.
    As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a two-dimensional numpy ndarray.

    """
    dmodel = DiscreteKuramotoModel(dt=0.5)
    # the function dmodel.f accept various parameters. Perhaps their name can give you an idea?
    f_euler_derivative = dmodel.f(x, u, k=0, compute_jacobian=True)[1]  #!b #!b Compute derivative here using the dmodel.
    return f_euler_derivative


if __name__ == "__main__":
    # Part 1: A sympy warmup. This defines a fairly nasty sympy function:
    z = sym.symbols('z')    # Create a symbolic variable #!o=a #!s=a
    g = sym.exp( sym.cos(z) ** 2) * sym.sin(z) # Create a nasty symbolic expression.
    print("z is:", z, " and g is:", g) #!s
    dg_dz = sym.diff(g, z) #!b #!b Compute the derivative of g here (symbolically)
    print("The derivative of the nasty expression is dg/dz =", dg_dz)

    g_as_a_function = sym.lambdify(z, dg_dz) #!b #!b Turn the symbolic expression into a function using sym.lambdify. Check the notes for an example (or sympys documentation).

    print("dg/dz (when z=0) =", g_as_a_function(0))
    print("dg/dz (when z=pi/2) =", g_as_a_function(np.pi/2))
    print("(Compare these results with the symbolic expression)") #!o=a


    # Part 2: Create a symbolic model corresponding to the Kuramoto model:
    cmodel = ContiniousKuramotoModel()

    # xdot = cmodel.f(x, u, k=0) # This is an example of using the environment. It may be of help later.
    print("Value of f(x,u) in x=2, u=0.3", f([2], [0.3])) #!o=b
    print("Value of f(x,u) in x=0, u=1", f([0], [1])) #!o=b

    """ We use cmodel.simulate(...) to simulate the environment, starting in x0 =0, from time t0=0 to tF=10, using a constant action of u=1.5. 
    Note u_fun in the simulate function can be set to a constant. 
    Use this compute compute numpy ndarrays corresponding to the time, x and u values. 
    
    To make this work, you have to implement RK4 integration. 
    """
    x0 = cmodel.reset() # Get the starting state x=0.
    u = 1.3
    xs, us, ts = cmodel.simulate(x0, u_fun=u, t0=0, tF=20)

    # Plot the exact simulation of the environment
    import matplotlib.pyplot as plt
    plt.plot(ts, xs, 'k-', label='RK4 state sequence x(t)')
    plt.plot(ts, us, 'r-', label='RK4 action sequence u(t)')
    plt.legend()
    savepdf('kuramoto_rk4')
    plt.show()

    plt.figure(2)
    # Part 3: The discrete environment
    dmodel = DiscreteKuramotoModel()  # Create a *discrete* model

    print("The Euler-discretized version, f_k(x,u) = x + Delta f(x,u), is") #!o=c
    print("f_k(x=0,u=0) =", fk([0], [0]))
    print("f_k(x=1,u=0.3) =", fk([1], [0.3]))

    print("The derivative of the Euler discretized version wrt. x is:")
    print("df_k/dx(x=0,u=0) =", dfk_dx([0], [0])) #!o=c

    # Part 4: The environment and simulation:

    env = KuramotoEnvironment(Tmax=20)  # An environment that runs for 5 seconds.

    ts_step = []  # Current time (according to the environment, i.e. in increments of dt.
    xs_step = []  # x_k using the env.step-function in the enviroment.
    xs_euler = [] # x_k using Euler discretization.

    x = env.reset()          # Get starting state.
    ts_step.append(env.time) # env.time keeps track of the clock-time in the environment.
    xs_step.append(x)        # Initialize with first state
    xs_euler.append(x)       # Initialize with first state

    # Use
    # > next_x, cost, done, metadata = env.step([u])
    # to simulate a single step.
    for _ in range(10000):
        next_x, cost, done, metadata = env.step([u]) #!b
        xs_step.append(next_x)
        ts_step.append(env.time)
        xs_euler.append(dmodel.f( xs_euler[-1], [u], 0 ))

        if done:
            break #!b Use the step() function to simulate the environment. Note that the step() function uses RK4.

    plt.plot(ts, xs, 'k-', label='RK4 (nearly exact)')
    plt.plot(ts_step, xs_step, 'b.', label='RK4 (step-function in environment)')
    plt.plot(ts_step, xs_euler, 'r.', label='Euler (dmodel.f(last_x, action, k)')

    # Train and plot a random agent.
    env = KuramotoEnvironment(Tmax=20) #!s=d
    stats, trajectories = train(env, Agent(env), return_trajectory=True)
    plt.plot(trajectories[0].time, trajectories[0].state, label='x(t) when using a random action sequence from agent') #!s=d
    plt.legend()
    savepdf('kuramoto_step')
    plt.show()
    print("The total cost obtained using random actions", -stats[0]['Accumulated Reward'])

