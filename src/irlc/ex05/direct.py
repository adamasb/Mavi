import numpy as np
import sympy as sym
import sys
from scipy.optimize import Bounds, minimize
from scipy.interpolate import interp1d
from irlc.ex04.continuous_time_model import symv
from irlc.ex04.continuous_time_discretized_model import sympy_modules_
from irlc import Timer
from tqdm import tqdm

def bounds2fun(t0, tF, bounds):
    """
    Given start and end times [t0, tF] and a scipy Bounds object with upper/lower bounds on some variable x, i.e. so that:

    > bounds.lb <= x <= bounds.ub

    this function returns a new function f such that f(t0) equals bounds.lb and f(tF) = bounds.ub and
    f(t) interpolates between the uppower/lower bounds linearly, i.e.

    > bounds.lb <= f(t) <= bounds.ub

    The function will return a numpy nd.array.
    """
    return interp1d(np.asarray([t0, tF]), np.stack([np.reshape(b, (-1,)) for b in bounds], axis=1))

def direct_solver(model, options):
    """
    Main direct solver method, see \nref{alg11directB}. Given a list of options of length S, the solver performers collocation
    using the settings found in the dictionary options[i], and use the result of options[i] to initialize collocation on options[i+1].

    This iterative refinement scheme is required to obtain good overall solutions.

    :param model: A ContinuousTimeModel instance
    :param options:  An options-structure. This is a list of dictionaries of options for each collocation iteration
    :return: A list of solutions, one for each collocation step. The last will be the 'best' solution (highest N)

    """
    if isinstance(options, dict):
        options = [options]
    solutions = []  # re-use result of current solutions to initialize next with a higher value of N
    for i, opt in enumerate(options):
        optimizer_options = opt['optimizer_options']  # to be passed along to minimize()
        if i == 0 or "guess" in opt:
            # No solutions-function is given. Re-calculate by linearly interpreting bounds (see \nref{s11guess})
            guess = opt['guess']
            guess['u'] = bounds2fun(guess['t0'],guess['tF'],guess['u']) if isinstance(guess['u'], list) else guess['u']
            guess['x'] = bounds2fun(guess['t0'],guess['tF'],guess['x']) if isinstance(guess['x'], list) else guess['x']
        else:
            """ For an iterative solver (\nref{s11guess}), initialize the guess at iteration i to be the solution at iteration i-1.
            The guess consists of a guess for t0, tF (just numbers) as well as x, u (state/action trajectories),
            the later two being functions. The format of the guess is just a dictionary (you have seen several examples)
            i.e. 
            
            > guess = {'t0': (number), 'tF': (number), 'x': (function), 'u': (function)}
            
            and you can get the solution by using solutions[i - 1]['fun']. (insert a breakpoint and check the fields)
            """
            guess = {k: solutions[i - 1]['fun'][k] for k in ['t0', 'tF', 'x', 'u'] } #!b #!b Define guess = {'t0': ..., ...} here.
        N = opt['N']
        print(f"{i}> Collocation starting with grid-size N={N}")
        sol = collocate(model, N=N, optimizer_options=optimizer_options, guess=guess, verbose=opt.get('verbose', False))
        solutions.append(sol)

    print("Was collocation success full at each iteration?")
    for i, s in enumerate(solutions):
        print(f"{i}> Success? {s['solver']['success']}")
    return solutions

def collocate(model, N=25, optimizer_options=None, guess=None, verbose=True):
    """
    Performs collocation by discretizing the model using a grid-size of N and optimize to find the optimal solution.
    The 'model' should be a ContinuosTimeModel instance, optimizer_options contains options for the optimizer, and guess
    is a dictionary used to initialize the optimizer containing keys:

    guess = {'t0': Start time (float),
             'tF': Terminal time (float),
             'x': A *function* which takes time as input and return a guess for x(t),
             'u': A *function* which takes time as input and return a guess for u(t),
            }

    So for instance

    > guess['x'](0.5)

    will return the state 'x(0.5)' as a numpy ndarray.
    The functions you need to use from the model are:

    > model.state_size # Dimension of x(t)
    > model.action_size # Dimension of u(t)
    > model.sym_f(x,u,t) # Symbolic expression of f(x,u,t)
    > model.sym_c(x,u,t) # The part of the cost-function we are integrating over. i.e. \int c(x,u,t) dt
    > model.sym_cf(x0,t0,xF,tF) # The constant term in the cost-function
    > model.simple_bounds() # A dictionary of all simple (constant) bounds on t0, tF, x, u, etc.

    For Brachistochrone (later) you need
    > model.sym_h(x,u,t) # Non-linear inequality constraints of the form h(x,u,t) <= 0


    ---
    The overall structure of the optimization procedure is as follows:

    (1) Define the following variables. They will all be lists:

    z: Variables to be optimized over. Each element z[k] is a symbolic variable (sym.symbols('variable_name')). This will allow us to compute derivatives.
    z0: A list of numbers representing the initial guess. Computed using 'guess' (above)
    z_lb, z_ub: Lists of numbers representting the upper/lower bounds on z. Derived from model.simple_bounds()

    The variables has the same order as in the notes. We use symbolic variables for z to be able to compute derivatives.

    (2)  Create a symbolic expression representing the cost-function J
    This is defined using the symbolic variables similar to the toy-problem we saw last week. This allows us to compute derivatives of the cost

    (3) Create *symbolic* expressions representing all constraints
    The lists 'ineqC' and 'eqC' contains *lists* of constraints. The solver will ensure that for any i:

    > eqC[i] == 0

    and

    > ineqC[i] <= 0

    This allows us to just specify each element in 'eqC' and 'ineqC' as a single symbolic expression. Once more, we use symbolic expressions so
    derivatives can be computed automatically. The most important constraints are in 'eqC', as these must include the collocation-constraints (see algorithm in notes)

    (4) Compile all symbolic expressions into a format useful for the optimizer
    The optimizer accepts numpy functions, so we turn all symbolic expressions and derivatives into numpy (similar to the example last week).
    It is then fed into the optimizer and, fingers crossed, the optimizer spits out a value 'z*', which represents the optimal values.

    (5) Unpack z
    The value 'z*' then has to be unpacked and turned into function u*(t) and x*(t) (as in the notes). These functions can then be put into the
    solution-dictionary and used to initialize the next guess (or assuming we terminate, these are simply our solution).

    :param model:
    :param N:
    :param optimizer_options:
    :param guess:
    :param verbose:
    :return:
    """
    timer = Timer(start=True)
    t0, tF = sym.symbols("t0"), sym.symbols("tF")
    ts = t0 + np.linspace(0, 1, N) * (tF-t0)   # N points linearly spaced between [t0, tF] TODO: Convert this to a list.
    xs, us = [], []
    for i in range(N):
        xs.append(list(symv("x_%i_" % i, model.state_size)))
        us.append(list(symv("u_%i_" % i, model.action_size)))

    ''' (1) Construct guess z0, all simple bounds [z_lb, z_ub] for the problem and collect all symbolic variables as z '''
    sb = model.simple_bounds()  # get simple inequality boundaries in problem (v_lb <= v <= v_ub)
    z = []  # list of all *symbolic* variables in the problem
    z0, z_lb, z_ub = [], [], []  # Guess z0 and lower/upper bounds (list-of-numbers): z_lb[k] <= z0[k] <= z_ub[k]
    ts_eval = sym.lambdify((t0, tF), ts.tolist(), modules='numpy')
    for k in range(N):
        x_bnd, u_bnd = sb['x'], sb['u']
        if k == 0:
            x_bnd = sb['x0']
        if k == N - 1:
            x_bnd = sb['xF']
        tk = ts_eval(guess['t0'], guess['tF'])[k]
        """ In these lines, update z, z0, z_lb, and z_ub with values corresponding to xs[k], us[k]. 
        The values are all lists; i.e. z[j] (symbolic) has guess z0[j] (float) and bounds z_lb[j], z_ub[j] (floats) """
        z, z0, z_lb, z_ub = z + xs[k], z0 + list(guess['x'](tk).flat), z_lb + list(x_bnd.lb), z_ub + list(x_bnd.ub) #!b
        z, z0, z_lb, z_ub = z + us[k], z0 + list(guess['u'](tk).flat), z_lb + list(u_bnd.lb), z_ub + list(u_bnd.ub) #!b Updates for x_k, u_k

    """ Update z, z0, z_lb, and z_ub with bounds/guesses corresponding to t0 and tF (same format as above). """
    z, z0, z_lb, z_ub = z+[t0], z0+[guess['t0']], z_lb+list(sb['t0'].lb), z_ub+list(sb['t0'].ub) #!b
    z, z0, z_lb, z_ub = z+[tF], z0+[guess['tF']], z_lb+list(sb['tF'].lb), z_ub+list(sb['tF'].ub) #!b Updates for t0, tF
    assert len(z) == len(z0) == len(z_lb) == len(z_ub)
    if verbose:
        print(f"z={z}\nz0={np.asarray(z0).round(1).tolist()}\nz_lb={np.asarray(z_lb).round(1).tolist()}\nz_ub={np.asarray(z_ub).round(1).tolist()}") #!o=v1 #!o
    print(">>> Trapezoid collocation of problem") # problem in this section
    fs, cs = [], []  # lists of symbolic variables corresponding to f_k and c_k, see \nref{alg11directA}.
    for k in range(N):
        """ Update both fs and cs; these are lists of symbolic expressions such that fs[k] corresponds to f_k and cs[k] to c_k in the slides. 
        Use the functions env.sym_f and env.sym_c """
        fs.append(model.sym_f(x=xs[k], u=us[k], t=ts[k])) #!b # fs.append( symbolic variable corresponding to f_k; see env.sym_f). similarly update cs.append(env.sym_c(...) ).
        cs.append(model.sym_c(x=xs[k], u=us[k], t=ts[k])) #!b Compute f[k] and c[k] here (see slides) and add them to above lists

    J = model.sym_cf(x0=xs[0], t0=t0, xF=xs[-1], tF=tF)  # cost, to get you started, but needs more work
    eqC, ineqC = [], []  # all symbolic equality/inequality constraints are stored in these lists
    for k in range(N - 1):
        # Update cost function (\nref{c11eq15}). Use the above defined symbolic expressions ts, hk and cs.
        hk = (ts[k + 1] - ts[k])  #!b
        J += .5 * hk * (cs[k] + cs[k + 1])  #!b Update J here
        # Set up equality constraints. See \nref{c11eq18}.
        for j in range(model.state_size):
            """ Create all collocation equality-constraints here and add them to eqC. I.e.  
            xs[k+1] - xs[k] = 0.5 h_k (f_{k+1} + f_k)
            Note we have to create these coordinate-wise which is why we loop over j. 
            """
            eqC.append((xs[k+1][j] - xs[k][j]) - 0.5*hk*(fs[k+1][j] + fs[k][j])) #!b #!b Update collocation constraints here
        """
        To solve problems with dynamical path constriants like Brachiostone, update ineqC here to contain the 
        inequality constraint env.sym_h(...) <= 0. For the other problems this can simply be left blank """
        ineqC += model.sym_h(x=xs[k], u=us[k], t=ts[k]) #!b #!b Update symbolic path-dependent constraint h(x,u,t)<=0 here

    print(">>> Creating objective and derivative...")
    timer.tic("Building symbolic objective")
    J_fun = sym.lambdify([z], J, modules='numpy')  # create a python function from symbolic expression
    # To compute the Jacobian, you can use sym.derive_by_array(J, z) to get the correct symbolic expression, then use sym.lamdify (as above) to get a numpy function.
    J_jac = sym.lambdify([z], sym.derive_by_array(J, z), modules='numpy') #!b #!b Jacobian of J. See how this is computed for equality/inequality constratins for help.
    if verbose:
        print(f"eqC={eqC}\nineqC={ineqC}\nJ={J}") #!o=v2 #!o
    timer.toc()
    print(">>> Differentiating equality constraints..."), timer.tic("Differentiating equality constraints")
    constraints = []
    for eq in tqdm(eqC, file=sys.stdout):  # dont' write to error output.
        constraints.append(constraint2dict(eq, z, type='eq'))
    timer.toc()
    print(">>> Differentiating inequality constraints"), timer.tic("Differentiating inequality constraints")
    constraints += [constraint2dict(ineq, z, type='ineq') for ineq in ineqC]
    timer.toc()

    c_viol = sum(abs(np.minimum(z_ub - np.asarray(z0), 0))) + sum(abs(np.maximum(np.asarray(z_lb) - np.asarray(z0), 0)))
    if c_viol > 0:  # check if: z_lb <= z0 <= z_ub. Violations only serious if large
        print(f">>> Warning! Constraint violations found of total magnitude: {c_viol:4} before optimization")

    print(">>> Running optimizer..."), timer.tic("Optimizing")
    z_B = Bounds(z_lb, z_ub)
    res = minimize(J_fun, x0=z0, method='SLSQP', jac=J_jac, constraints=constraints, options=optimizer_options, bounds=z_B) #!s=min #!s
    # Compute value of equality constraints to check violations
    timer.toc()
    eqC_fun = sym.lambdify([z], eqC)
    eqC_val_ = eqC_fun(res.x)
    eqC_val = np.zeros((N - 1, model.state_size))

    x_res = np.zeros((N, model.state_size))
    u_res = np.zeros((N, model.action_size))
    t0_res = res.x[-2]
    tF_res = res.x[-1]

    m = model.state_size + model.action_size
    for k in range(N):
        dx = res.x[k * m:(k + 1) * m]
        if k < N - 1:
            eqC_val[k, :] = eqC_val_[k * model.state_size:(k + 1) * model.state_size]
        x_res[k, :] = dx[:model.state_size]
        u_res[k, :] = dx[model.state_size:]

    # Generate solution structure
    ts_numpy = ts_eval(t0_res, tF_res)
    # make linear interpolant similar to \nref{c11eq22}
    ufun = interp1d(ts_numpy, np.transpose(u_res), kind='linear')
    # Evaluate function values fk points (useful for debugging but not much else):
    f_eval = sym.lambdify((t0, tF, xs, us), fs)
    fs_numpy = f_eval(t0_res, tF_res, x_res, u_res)
    fs_numpy = np.asarray(fs_numpy)

    """ make cubic interpolant similar to \nref{c11eq26} """
    x_fun = lambda t_new: trapezoid_interpolant(ts_numpy, np.transpose(x_res), np.transpose(fs_numpy), t_new=t_new)

    if verbose:
        newt = np.linspace(ts_numpy[0], ts_numpy[-1], len(ts_numpy)-1)
        print( x_fun(newt) ) #!o=v4 #!o

    sol = {
        'grid': {'x': x_res, 'u': u_res, 'ts': ts_numpy, 'fs': fs_numpy},
        'fun': {'x': x_fun, 'u': ufun, 'tF': tF_res, 't0': t0_res},
        'solver': res,
        'eqC_val': eqC_val,
        'inputs': {'z': z, 'z0': z0, 'z_lb': z_lb, 'z_ub': z_ub},
    }
    print(timer.display())
    return sol

def trapezoid_interpolant(ts, xs, fs, t_new=None):
    ''' Quadratic interpolant as in \nref{c11eq26}. Inefficient but works. '''
    I = []
    t_new = np.reshape(np.asarray(t_new), (-1,))
    for t in t_new:  # yah, this is pretty terrible..
        i = -1
        for i in range(len(ts) - 1):
            if ts[i] <= t and t <= ts[i + 1]:
                break
        I.append(i)

    ts = np.asarray(ts)
    I = np.asarray(I)
    tau = t_new - ts[I]
    hk = ts[I + 1] - ts[I]
    """
    Make interpolation here. Should be a numpy array of dimensions [xs.shape[0], len(I)]
    What the code does is that for each t in ts, we work out which knot-point interval the code falls within. I.e. 
    insert a breakpoint and make sure you understand what e.g. the code tau = t_new - ts[I] does.
     
    Given this information, we can recover the relevant (evaluated) knot-points as for instance 
    fs[:,I] and those at the next time step as fs[:,I]. With this information, the problem is simply an 
    implementation of  \nref{c11eq26}, i.e. 

    > x_interp = xs[:,I] + tau * fs[:,I] + (...)    
    
    """
    x_interp = xs[:, I] + tau * fs[:, I] + (tau ** 2 / (2 * hk)) * (fs[:, I + 1] - fs[:, I]) #!b #!b
    return x_interp


def constraint2dict(symb, all_vars, type='eq'):
    ''' Turn constraints into a dict with type, fun, and jacobian field. '''
    if type == "ineq": symb = -1 * symb  # To agree with sign convention in optimizer

    f = sym.lambdify([all_vars], symb, modules=sympy_modules_)
    # np.atan = np.arctan  # Monkeypatch numpy to contain atan. Passing "numpy" does not seem to fix this.
    jac = sym.lambdify([all_vars], sym.derive_by_array(symb, all_vars), modules=sympy_modules_)
    eq_cons = {'type': type,
               'fun': f,
               'jac': jac}
    return eq_cons

def get_opts(N, ftol=1e-6, guess=None, verbose=False): # helper function to instantiate options objet.
    d = {'N': N,
         'optimizer_options': {'maxiter': 1000,
                               'ftol': ftol,
                               'iprint': 1,
                               'disp': True,
                               'eps': 1.5e-8},  # 'eps': 1.4901161193847656e-08,
         'verbose': verbose}
    if guess:
        d['guess'] = guess
    return d

def run_direct_small_problem():
    from irlc.ex04.model_pendulum import ContiniousPendulumModel
    model = ContiniousPendulumModel()
    """
    Test out implementation on a VERY small grid. The overall solution will be failry bad, 
    but we can print out the various symbolic expressions
    
    We use verbose=True to get debug-information.
    """
    print("Solving with a small grid, N=5 (yikes)")
    options = [get_opts(N=5, ftol=1e-3, guess=model.guess(), verbose=True)]
    solutions = direct_solver(model, options)
    return model, solutions

if __name__ == "__main__":
    from irlc.ex05.direct_plot import plot_solutions
    model, solutions = run_direct_small_problem()
    plot_solutions(model, solutions, animate=False, pdf="direct_pendulum_small")
