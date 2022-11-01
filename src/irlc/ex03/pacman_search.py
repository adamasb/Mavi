import time
from irlc import Agent, train, VideoMonitor
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc.ex03.dp_forward import dp_forward
from irlc.ex03.search_problem import SearchProblem
from irlc.ex03.search_problem import EnsureTerminalSelfTransitionsWrapper

layout1 = """
%%%%%%%%%
% .    .%
%       %
% .     %
%P    %%%
%     %.%
%%%%%%%%%
"""

layout2 = """
%%%%%%%%%
% .    .%
%      .%
% .     %
%P    %%%
%     %.%
%%%%%%%%%
"""

class PacmanSearchProblem(SearchProblem):
    """
    Functions that I used:

    > state.getPacmanPosition()
    > state.getScore()
    > state.A()
    > state.f(action)

    """
    def __init__(self, env):
        initial_state = env.reset()
        super().__init__(initial_state)

    def is_terminal(self, state):
        # Returns true if this state is terminal, i.e. if Pacman's position is (1,1).
        is_in_position_11 = state.getPacmanPosition() == (1, 1) #!b #!b
        return is_in_position_11

    def available_transitions(self, state):       
        s0 = state.getScore()
        transitions = {}
        for a in state.A(): #!b
            sp = state.f(a)
            transitions[a] = (sp, s0-sp.getScore()) #!b Create the available transitions here
        return transitions


class ForwardDPSearchAgent(Agent):
    """
    This is an agent which plan using dynamical programming.
    """
    def __init__(self, env, N=10):
        super().__init__(env)
        search_problem = PacmanSearchProblem(env)
        search_problem = EnsureTerminalSelfTransitionsWrapper(search_problem)
        J, self.actions, path = dp_forward(search_problem, N)

    def pi(self, s, k=None):
        action = None
        action = self.actions[k] #!b #!b Compute the action here. Hint: Look at the __init__ function.
        # This code handle the case where the action is not set correctly.
        # I found this useful given that EnsureTerminalSelfTransitionsWrapper return a dummy action when the environment has terminated.
        if action not in s.A():
            return "Stop"
        else:
            return action

if __name__ == "__main__":
    # Make snapshots of the two layouts.
    N = 20  # Plan over a horizon of N steps.
    env1 = GymPacmanEnvironment(layout_str=layout1) # Create environment #!s=a
    problem1 = PacmanSearchProblem(env1)            # Transform into a search problem
    problem1 = EnsureTerminalSelfTransitionsWrapper(problem1) # This will make the terminal states absorbing as described in \nref{c2s21} #!s

    # Same for second layout
    env2 = GymPacmanEnvironment(layout_str=layout2)
    problem2 = PacmanSearchProblem(env2)
    problem2 = EnsureTerminalSelfTransitionsWrapper(problem2) # This will make the terminal states absorbing as described in \nref{c2s21}

    J1, _, _ = dp_forward(problem1, N)  # Compute optimal trajectory in layout 1
    J2, _, _ = dp_forward(problem2, N)  # Compute optimal trajectory in layout 2
    optimal_cost_1 = min( [cost for s, cost in J1[-1].items() if problem1.is_terminal(s)] )
    optimal_cost_2 = min( [cost for s, cost in J2[-1].items() if problem1.is_terminal(s)] )

    print("Optimal cost in layout 1", optimal_cost_1) #!o
    print("Optimal cost in layout 2", optimal_cost_2) #!o

    # show what pacman actually does in the two cases:
    env1 = VideoMonitor(env1)
    env2 = VideoMonitor(env2)
    t0 = time.time()
    agent1 = ForwardDPSearchAgent(env1, N=N) #!s=b
    train(env1, agent1, max_steps=N) #!s
    env1.close()
    print("Time elapsed", time.time() - t0)

    t0 = time.time()
    agent2 = ForwardDPSearchAgent(env2, N=N)
    train(env2, agent2, max_steps=N)
    print("Time elapsed", time.time()-t0)
    env2.close()
