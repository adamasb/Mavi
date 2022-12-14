import numpy as np
from gym.spaces.discrete import Discrete
from gym import Env

class ChessTournament(Env):
    """
       Environment is used to simulate a chess tournament which ends when a player wins two games in a row. The results
       of each game are -1, 0, 1 corresponding to a loss, draw and win for player 1. See:
       https://www.youtube.com/watch?v=5UQU1oBpAic

        To implement this, we define the step-function such that one episode of the environment corresponds to playing
        a chess tournament to completion. Once the environment completes, it returns a reward of +1 if the player won
        the tournament, and otherwise 0.

        Each step therefore corresponds to playing a single game in the tournament.
        To implement this, we use a state corresponding to the sequence of games in the tournament:

        >>> self.s = [0, -1, 1, 0, 0, 1]

        In the self.step(action)-function, we ignore the action, simulate the outcome of a single game,
        and append the outcome to self.s. We then compute whether the tournament has completed, and if so
        a reward of 1 if we won.
       """

    def __init__(self, p_draw=3 / 4, p_win=2 / 3):
        self.action_space = Discrete(1)
        self.p_draw = p_draw
        self.p_win = p_win
        self.s = []  # A chess tournament is a sequence of won/lost games s = [0, -1, 1, 0, ...]

    def reset(self): #!f
        """ After each episode is complete, reset self.s and return s (as dictated by openai Gyms interface)
        :return: Current state s
        """
        self.s = []
        return self.s

    def step(self, action):
        game_outcome = None # should be -1, 0, or 1 depending on outcome of single game.
        if np.random.rand() < self.p_draw: #!b
            game_outcome = 0
        else:
            if np.random.rand() < self.p_win:
                game_outcome = 1
            else:
                game_outcome = -1 #!b Compute game_outcome here
        self.s.append(game_outcome)

        #done = True if the tournament has ended otherwise false. Compute using s.
        done = len(self.s) >= 2 and self.s[-1] == self.s[-2] and self.s[-1] != 0 #!b #!b Compute 'done', whether the tournament has ended.
        # r = ... . Compute reward. Let r=1 if we won the tournament otherwise 0.
        r = self.s[-1] == 1 if done else 0   #!b #!b Compute the reward 'r' here.
        return self.s, r, done, {}

def main():
    T = 5000
    """
    Simulate tournamnet for T games and estimate average win probability for player 1 as p_win (answer to riddle) and also 
    the average length. Note the later should be a 1-liner, but would require non-trivial computations to solve
    analytically. See Environment class for details.    
    """

    from irlc import train, Agent
    env = ChessTournament()
    # Compute stats using the train function. Simulate the tournament for a total of T=10'000 episodes.
    stats, _ = train(env, Agent(env), num_episodes=T) #!b #!b Compute stats here using train(env, ...). Use num_episodes.
    p_win = np.mean([st['Accumulated Reward'] for st in stats])
    avg_length = np.mean([st['Length'] for st in stats])

    print("Agent: Estimated chance I won the tournament: ", p_win)  #!o
    print("Agent: Average tournament length", avg_length)  #!o


if __name__ == "__main__":
    main()
