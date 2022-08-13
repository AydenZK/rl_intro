"""A simple Connect-4 custom gym environment to train a reinforcement learning agent.
Author: Ayden Khalil
Date: 24/04/2022
"""

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

DISCRETE_ACTION_SPACE = True

class Connect4Random(Env):
    """Playing in a single player env where the opponent takes random actions"""
    def __init__(self):
        super(Connect4Random, self).__init__()
        if DISCRETE_ACTION_SPACE:
            self.action_space = Discrete(7) # discrete action space
        else:
            self.action_space = Box(low=0, high=7, shape=(1,)) # continuous action space
        
        self.observation_space = Box(low=-1, high=1, shape=(6,7), dtype=int)
        self.start_state = np.zeros((6,7))
        self.state = self.start_state

    def reset(self):
        return self.start_state

    def check_done(self) -> tuple:
        """Checks if game is done
        Returns
            (tuple): (game_done:bool, winner: int or None)
                (False, None) - Game is not done
                (True, -1) - Opponent won
                (True, 1) - RL Agent won
                (True, 0) - Game is a draw
        """
        # check horizontals
        for row in self.state:
            for i in range(4):
                hor_of_four = row[i:i+4]
                if all([k == 1 for k in hor_of_four]):
                    return (True, 1)
                elif all([k == -1 for k in hor_of_four]):
                    return (True, -1)

        # check verticals
        for top_row in range(3):
            for col in range(len(self.state[top_row])):
                col_of_4 = [self.state[row][col] for row in range(top_row, top_row + 4)]
                if all([k == 1 for k in col_of_4]):
                    return (True, 1)
                elif all([k == -1 for k in col_of_4]):
                    return (True, -1)

        # check diagonals left to right
        for start_row in range(3):
            for start_col in range(4):
                diag_4 = [self.state[row][col] for row, col in zip(
                    range(start_row, start_row + 4), range(start_col, start_col + 4))]
                if all([k == 1 for k in diag_4]):
                    return (True, 1)
                elif all([k == -1 for k in diag_4]):
                    return (True, -1)

        # check diagonals right to left
        for start_row in range(3):
            for start_col in range(3,7):
                r_diag_4 = [self.state[row][col] for row, col in zip(
                    range(start_row, start_row + 4), range(start_col, start_col - 4, -1))]
                if all([k == 1 for k in r_diag_4]):
                    return (True, 1)
                elif all([k == -1 for k in r_diag_4]):
                    return (True, -1)
        
        if all([i != 0 for i in self.state[0]]): # checks if top row is full
            return (True, 0) # 0 means draw

        return (False, None)

    def execute_action(self, action: float, player: int):
        """Executes an action in the games current state.
        Args:
            action (float): an action - which column to place a chip, continuous range (0,6)
            player (int): either 1 (the RL agent) or -1 (opponenent)
        """
        action = action if DISCRETE_ACTION_SPACE else round(action[0]) # converts to integer
        # The action specifies the column, iterates from the bottom row to find a clear space
        # to insert a chip.
        for i in range(5, -1, -1):
            if self.state[i][action] == 0:
                self.state[i][action] = player
                break

    def step(self, action: float) -> tuple:
        """Steps through the game given an action from the RL agent.
        Args:
            action (float): an action - which column to place a chip, continuous range (0,6)
        """
        # Reward is 0 unless the game is over
        reward = 0

        # Take my action, player = 1
        self.execute_action(action=action, player=1)
        
        done, winner = self.check_done() # checks if game is over and who won

        # If game not done, opponenent takes an action:
        if not done:
            # Opponent takes a random action, player = -1
            random_action = self.action_space.sample()
            self.execute_action(action=random_action, player=-1)
        else:
            # simple reward function
            if winner != 0: # not a draw
                reward = 10 * winner
            else:
                reward = -2

            # TODO: complex reward function that prioritises finishing the game quickly
        
        return self.state, reward, done, {}

    def render(self):
        """Renders the environment to the screen"""
        print(self.state)
        done, winner = self.check_done()
        print_dict = {
            -1: "The game is over and the opponent won!",
            0: "The game ended in a draw!",
            1: "The game is over and the RL agent won!"
        }
        if done:
            print(print_dict[winner])
        else:
            print("Game in progress...")