from __future__ import annotations
from copy import deepcopy

from typehints import *

class Game:
    def copy(self) -> Game:
        """
        RETURNS:
            Deep copy of the current game, including its internal state. If RNG is used, use the random.Random class instead of querying the random module directly so that environment remains deterministic.
        """
        return deepcopy(self)
    
    def render(self) -> None:
        """
        Displays the visual representation of the current state of the game. Returns nothing.
        """
        raise NotImplementedError

    def get_state(self) -> State:
        """
        RETURNS:
            Current state of the game, viewed from the perspective of the current player. State provided should be immutable and hashable if possible.
        """
        raise NotImplementedError
        
    def num_players(self) -> int:
        """
        RETURNS:
            Number of players, which must remain constant over an episode.
        """
        raise NotImplementedError
    
    def current_player(self) -> Player:
        """
        RETURNS:
            Player that is taking an action for the current state. Players are labelled from 0 to num_players - 1.
        """
        raise NotImplementedError

    def max_num_actions(self) -> int:
        """
        RETURNS:
            Maximum number of actions, which must remain constant over an episode.
        """
        raise NotImplementedError
    
    def possible_actions(self) -> Actions:
        """
        RETURNS:
            List of all possible actions that can be taken by current player. Actions are labelled from 0 to max_num_actions - 1. List of possible actions should be made immutable where possible, but does not need to be ordered.
        """
        raise NotImplementedError
    
    def take_action(self, action: Action) -> None:
        """
        Make the current player take action in the current state. Returns nothing.
        """
        raise NotImplementedError
    
    def has_outcome(self) -> bool:
        """
        RETURNS:
            True if state is terminal, else False.
        """
        raise NotImplementedError

    def final_scores(self) -> Scores:
        """
        RETURNS:
            List of scores achieved by each player at terminal state. List of scores is indexed by the player's ID.
        """
        raise NotImplementedError
