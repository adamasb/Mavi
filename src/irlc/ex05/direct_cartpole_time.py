from irlc.ex04.model_cartpole import ContiniousCartpole, kelly_swingup
from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex05.direct_plot import plot_solutions

def compute_solutions():
    """
        See: https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minTime.m
        """
    cost, bounds, _ = kelly_swingup(maxForce=50, dist=1.0)
    model = ContiniousCartpole(maxForce=50, mp=0.5, mc=2.0, g=9.81, l=0.5, cost=cost, simple_bounds=bounds)
    options = [get_opts(N=8, ftol=1e-3, guess=model.guess()),
               get_opts(N=16, ftol=1e-6),                # This is a hard problem and we need gradual grid-refinement.
               get_opts(N=32, ftol=1e-6),
               get_opts(N=70, ftol=1e-6)
               ]
    solutions = direct_solver(model, options)
    return model, solutions

if __name__ == "__main__":
    model, solutions = compute_solutions()
    x_sim, u_sim, t_sim = plot_solutions(model, solutions[:], animate=True, pdf="direct_cartpole_mintime")
    model.close()
    print("Did we succeed?", solutions[-1]['solver']['success'])