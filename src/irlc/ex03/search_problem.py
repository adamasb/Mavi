from irlc.ex02.graph_traversal import G222


class SearchProblem: #!s=spp #!s=spp
    """
    An abstract search problem. Provides the functionality defined in \nref{searchp}.
    The search problem has a function to tell the user when a state is terminal, and what transitions are available in a state. 
    """
    def __init__(self, initial_state=None): #!s=spp
        if initial_state is not None:
            self.set_initial_state(initial_state)

    def set_initial_state(self, state):
        """ Re-set the initial (start) state of the search problem. """
        self.initial_state = state

    def is_terminal(self, state):
        """ Return true if and only if state is the terminal state. """
        raise NotImplementedError("Implement a goal test")

    def available_transitions(self, state):
        """ return the available set of transitions in this state
        as a dictionary {a: (s1, c), a2: (s2,c), ...}
        where a is the action, s1 is the state we transition to when we take action 'a' in state 'state', and
        'c' is the cost we will obtain by that transition.
        """
        raise NotImplementedError("Transition function not impelmented") #!s=spp

class EnsureTerminalSelfTransitionsWrapper(SearchProblem):
    def __init__(self, search_problem):
        self._sp = search_problem
        super().__init__(search_problem.__dict__.get('initial_state', None)) # Get initial state if set.

    def set_initial_state(self, state):
        self._sp.set_initial_state(state)

    @property
    def initial_state(self):
        return self._sp.initial_state

    def is_terminal(self, state):
        return self._sp.is_terminal(state)

    def available_transitions(self, state):
        return {0: (state, 0)} if self.is_terminal(state) else self._sp.available_transitions(state)

class DP2SP(SearchProblem):
    """ This class converts a Deterministic DP environment to a shortest path problem matching the description
    in \cref{c2:Jmin}.
    """
    def __init__(self, env, initial_state):
        self.env = env
        self.terminal_state = "terminal_state"
        super(DP2SP, self).__init__(initial_state=(initial_state, 0))

    def is_terminal(self, state):
        return state == self.terminal_state

    def available_transitions(self, state):
        """ Implement the dp-to-search-problem conversion described in \nref{dp2sp}. Keep in mind the time index is
        absorbed into the state; this means that state = (x, k) where x and k are intended to be used as
        env.f(x, <action>, <noise w>, k).
        As usual, you can set w=None since the problem is assumed to be deterministic.

        The output format should match SearchProblem, i.e. a dictionary with keys as u and values as (next_state, cost).
        """
        if state == self.terminal_state:
            return {0: (self.terminal_state, 0)}
        s, k = state
        if k == self.env.N: #!b
            return {0: ( self.terminal_state, self.env.gN(s))}
        return {u: ((self.env.f(s, u, None, k), k+1), self.env.g(s, u, None, k)) for u in self.env.A(s, k)} #!b return transtitions as dictionary


class SmallGraphSP(SearchProblem):
    G = G222


class GraphSP(SearchProblem): #!s=smallgraph_sp
    """ Implement the small graph graph problem in \nref{c2smallgraph} """
    G = G222

    def __init__(self, start=2, goal=5):
        self.goal = goal
        super().__init__(initial_state=start)

    def is_terminal(self, state): # Return true if the state is a terminal state
        return state == self.goal

    def available_transitions(self, i,k=None):
        # In vertex i, return available transitions i -> j and their cost.
        # This is encoded as a dictionary, such that the keys are the actions, and
        # the values are of the form (next_state, cost).
        return {j: (j, cost) for (i_, j), cost in self.G.items() if i_ == i} #!s

    @property
    def vertices(self):
        # Helper function: Return number of vertices in the graph. You can ignore this.
        return len(set([i for edge in self.G for i in edge]))
