
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    
    def my_moves(self, gameState):
        
        if gameState.ply_count % 2 == 0:
            loc_player_1 = gameState.locs[0]
            loc_player_2 = gameState.locs[1]
        else:
            loc_player_1 = gameState.locs[1]
            loc_player_2 = gameState.locs[0]

        val_p_1 = abs((11-int(loc_player_1))) + abs((9-int(loc_player_1)))
        val_p_2 = abs((11-int(loc_player_2))) + abs((9-int(loc_player_2)))
        return val_p_1 - val_p_2

        
    
    def min_value(self, gameState, depth):
        """ Return the game state utility if the game is over,
        otherwise return the minimum value over all legal successors
        """
        if gameState.terminal_test():
            return gameState.utility(0)

        if depth <= 0:
            return self.my_moves(gameState)

        v = float("inf")
        for a in gameState.actions():
            # the depth should be decremented by 1 on each call
            v = min(v, self.max_value(gameState.result(a), depth - 1))
        return v

    def max_value(self, gameState, depth):
        """ Return the game state utility if the game is over,
        otherwise return the maximum value over all legal successors
        """
        if gameState.terminal_test():
            return gameState.utility(0)

        if depth <= 0:
            return self.my_moves(gameState)

        v = float("-inf")
        for a in gameState.actions():
            # the depth should be decremented by 1 on each call
            v = max(v, self.min_value(gameState.result(a), depth - 1))
        return v

    
    def minimax_decision(self, gameState, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        best_score = float("-inf")
        best_move = None
        for a in gameState.actions():
            if gameState.ply_count < 10:
                best_move = random.choice(gameState.actions())
                break
            # call has been updated with a depth limit
            v = self.min_value(gameState.result(a), depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def get_action_helper(self, gameState, depth_limit):
        # Turns out "iterative deepening" is just a for loop...
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = self.minimax_decision(gameState, depth)
        return best_move
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        next_action = self.get_action_helper(state, 1)
        print(next_action)
        self.queue.put(next_action)
