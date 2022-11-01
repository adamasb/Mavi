from scipy.optimize import minimize
import numpy as np
ineq_cons = {}
z0 = None
J_fun = lambda x: 32
J_jac = None
ineq_cons['fun'][z0]  # return a 3 x 1 array #!s=b
ineq_cons['jac'][z0]  # return a 3 x 2 matrix (the Jacobian of fun) because we have 2 input variables
ineq_cons['fun'][z0] <= 0 # this will be true of the inequality constraint for a solution #!s


ineq_cons = {'type': 'ineq', #!s=a
             'fun': lambda x: np.array([1 - x[0] - 2 * x[1],
                                        1 - x[0] ** 2 - x[1],
                                        1 - x[0] ** 2 + x[1]]),
             'jac': lambda x: np.array([[-1.0, -2.0],
                                        [-2 * x[0], -1.0],
                                        [-2 * x[0], 1.0]])}
eq_cons = {'type': 'eq',
           'fun': lambda x: np.array([2 * x[0] + x[1] - 1]),
           'jac': lambda x: np.array([2.0, 1.0])}
from scipy.optimize import Bounds
z_lb, z_ub = [0, -0.5], [1.0, 2.0]
bounds = Bounds(z_lb, z_ub)  # Bounds(z_low, z_up)
z0 = np.array([0.5, 0])
res = minimize(J_fun, z0, method='SLSQP', jac=J_jac,
               constraints=[eq_cons, ineq_cons], bounds=bounds) #!s

# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
#, options={'ftol': 1e-9},
