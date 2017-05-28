"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Number of moves left vs 2 x the number of opponent moves
    return float(len(game.get_legal_moves(player)) - 2*len(game.get_legal_moves(game.get_opponent(player))))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #w, h = game.width, game.height
    y, x = game.get_player_location(player)
    y_o, x_o = game.get_player_location(game.get_opponent(player))

    #Calculate player distances from centre
    #Middle square is (4,4)
    player_distance =  float(abs(4 - y) + abs(4 - x))
    opponent_distance =  float(abs(4 - y_o) + abs(4 - x_o))
    #return float(opponent_distance - player_distance)
    return float(opponent_distance - player_distance)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]

    w, h = game.width, game.height
    y, x = game.get_player_location(player)
    y_o, x_o = game.get_player_location(game.get_opponent(player))

    #check if position blocked the opponent
    for dr, dc in  directions:
        new_y_o = y_o + dr
        new_x_o = x_o + dc
        if new_y_o == y and new_x_o == x:
            #Create a score base on the number of moves we denied the opponent
            moves_player = game.get_legal_moves(player)
            moves_opponent = game.get_legal_moves(game.get_opponent(player))
            #Remove moves that the opponent could still make on its next turn
            denied_moves = [x for x in moves_player if x not in moves_opponent]
            return float(len(denied_moves))
    return float(0)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if there are any moves we can make, if not terminate the game
        if len(game.get_legal_moves()) == 0:
            return (-1, -1)

        #Get the current player
        player = game.active_player

        #Pick the best move for the current player and return it
        best_move = self.minmax_decision(game, depth, player)
        return best_move

    def minmax_decision(self, game, depth, player):
        """
        For the current player, this function returns their best move using minimax algorithm

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        player : isolation.Board.player
            The current active player

        Returns
        -------
        (int,int)
            A tuple representing the best move the player can make
        """

        #Set the layer count
        layer = 0

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Recurse one more layer deeper
        legal_moves = game.get_legal_moves()
        branch_scores = [] #Where we save the scores for each of the branches
        for move in legal_moves:
            new_game = game.forecast_move(move)
            move_score = self.min_move(new_game, layer, depth, player) #Remember min player is the next to move
            branch_scores.append(move_score)
        return legal_moves[branch_scores.index(max(branch_scores))] #Return the move with the best score

    def max_move(self, game, layer, depth, player):
        """
        Checks all the possiable moves max can make and returns the move with the highest evaluation score.
        If no more move can be made, the win/loss state is returned

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        layer : int
            Layer is an integer representing our current depth

        player : isolation.Board.player
            The current active player

        Returns
        -------
        float
            -inf if the player has lost, +inf if the player has won, or a score based on a heuristic evaluation function
        """

        #Increment the layer count
        layer = layer + 1

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if we have reached the depth limit
        if layer == depth:
            return self.score(game, player) # Return the value of this leaf

        #Check if we have reached an End-Game state
        if len(game.get_legal_moves()) == 0: #No legal moves left, so the game is over
            if player == game.is_winner(player):
                return float("inf") #Player won
            else:
                return float("-inf") #Player lost

        #Recurse one more layer deeper
        legal_moves = game.get_legal_moves()
        branch_scores = [] #Where we save the scores for each of the branches
        for move in legal_moves: #Begin recursion over each branch
            new_game = game.forecast_move(move) #Create a new game board with each branch
            move_score = self.min_move(new_game, layer, depth, player) #Get the score for that branch
            branch_scores.append(move_score) #Append the score to our list
        return max(branch_scores) #Return the MAX branch score, the move we would make

    def min_move(self, game, layer, depth, player):
        """
        Checks all the possiable moves min can make and returns the move with the lowest evaluation score.
        If no more move can be made, the win/loss state is returned

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        layer : int
            Layer is an integer representing our current depth

        player : isolation.Board.player
            The current active player

        Returns
        -------
        float
            -inf if the player has lost, +inf if the player has won, or a score based on a heuristic evaluation function
        """

        #Increment the layer count
        layer = layer + 1

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if we have reached the depth limit
        if layer == depth:
            return self.score(game, player) # Return the value of this leaf

        #Check if we have reached an End-Game state
        if len(game.get_legal_moves()) == 0: #No legal moves left, so the game is over
            if player == game.is_winner(player):
                return float("inf") #Player won
            else:
                return float("-inf") #Player lost

        #Recurse one more layer deeper
        legal_moves = game.get_legal_moves()
        branch_scores = [] #Where we save the scores for each of the branches
        for move in legal_moves: #Begin recursion over each branch
            new_game = game.forecast_move(move) #Create a new game board with each branch
            move_score = self.max_move(new_game, layer, depth, player) #Get the score for that branch
            branch_scores.append(move_score) #Append the score to our list
        return min(branch_scores) #Return the MIN branch score, the move we think the other player would make

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            depth = 1 # Assume we have enough time to run at least a depth of one
            #Keep searching for a better best move
            #By increasing the depth of the search until the time runs out
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth + 1

        except SearchTimeout:
            #Return the last best_move we calculated
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if there are any moves we can make, if not terminate the game
        if len(game.get_legal_moves()) == 0:
            return (-1, -1)

        #Get the current player
        player = game.active_player

        #Pick the best move for the current player and return it
        best_move = self.minmax_decision(game, depth, player, alpha, beta)
        return best_move

    def minmax_decision(self, game, depth, player, alpha, beta):
        """
        For the current player, this function returns their best move using minimax algorithm with alpha beta pruning

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        player : isolation.Board.player
            The current active player

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int,int)
            A tuple representing the best move the player can make
        """

        #Set the layer count
        layer = 0

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        branch_scores = [] #Where we save the scores for each of the branches

        move_score = float("-inf")

        #Recurse one more layer deeper
        for move in legal_moves: #Begin recursion over each branch
            new_game = game.forecast_move(move) #Create a new game board with each branch
            #move_score = self.min_move(new_game, layer, depth, player, alpha, beta) #Get the score for that branch
            move_score = max(move_score, self.min_move(new_game, layer, depth, player, alpha, beta))
            branch_scores.append(move_score)
            alpha = max(alpha, move_score)
        return legal_moves[branch_scores.index(max(branch_scores))] #Return the move with the best score

    def max_move(self, game, layer, depth, player, alpha, beta):
        """
        Checks all the possiable moves max can make and returns the move with the highest evaluation score.
        If no more move can be made, the win/loss state is returned

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        layer : int
            Layer is an integer representing our current depth

        player : isolation.Board.player
            The current active player

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        float
            -inf if the player has lost, +inf if the player has won, or a score based on a heuristic evaluation function
        """

        #Increment the layer count
        layer = layer + 1

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if we have reached the depth limit
        if layer == depth:
            return self.score(game, player) # Return the value of this leaf

        #Check if we have reached an End-Game state
        if len(game.get_legal_moves()) == 0: #No legal moves left, so the game is over
            if player == game.is_winner(player):
                return float("inf") #Player won
            else:
                return float("-inf") #Player lost


        legal_moves = game.get_legal_moves()
        move_score = float("-inf")

        #Recurse one more layer deeper
        for move in legal_moves: #Begin recursion over each branch
            new_game = game.forecast_move(move) #Create a new game board with each branch
            move_score = max(move_score, self.min_move(new_game, layer, depth, player, alpha, beta))
            if move_score >= beta:
                return move_score
            alpha = max(alpha, move_score)
        return move_score

    def min_move(self, game, layer, depth, player, alpha, beta):
        """
        Checks all the possiable moves min can make and returns the move with the lowest evaluation score.
        If no more move can be made, the win/loss state is returned

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        layer : int
            Layer is an integer representing our current depth

        player : isolation.Board.player
            The current active player

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        float
            -inf if the player has lost, +inf if the player has won, or a score based on a heuristic evaluation function
        """

        #Increment the layer count
        layer = layer + 1

        #Check we still have time to make a move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Check if we have reached the depth limit
        if layer == depth:
            return self.score(game, player) # Return the value of this leaf

        #Check if we have reached an End-Game state
        if len(game.get_legal_moves()) == 0: #No legal moves left, so the game is over
            if player == game.is_winner(player):
                return float("inf") #Player won
            else:
                return float("-inf") #Player lost

        #Recurse one more layer deeper
        legal_moves = game.get_legal_moves()
        branch_scores = [] #Where we save the scores for each of the branches
        move_score = float("inf")

        #Recurse one more layer deeper
        for move in legal_moves: #Begin recursion over each branch
            new_game = game.forecast_move(move) #Create a new game board with each branch
            move_score = min(move_score, self.max_move(new_game, layer, depth, player, alpha, beta))
            if move_score <= alpha:
                return move_score
            beta = min(beta, move_score)
        return move_score
