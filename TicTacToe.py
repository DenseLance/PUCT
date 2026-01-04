from Game import Game

class TicTacToe(Game):
    def __init__(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.players = {"X": 0, "O": 1}
        self.turn_order = ["X", "O"]
        
    def render(self):
        board = ""
        board += "|".join(self.board[0]) + "\n"
        board += "-----\n"
        board += "|".join(self.board[1]) + "\n"
        board += "-----\n"
        board += "|".join(self.board[2]) + "\n"
        print(board)

    def get_state(self):
        # State is viewed from the perspective of current player
        return tuple(tuple(1 if self.board[row][col] == self.turn_order[0] else -1 if self.board[row][col] == self.turn_order[1] else 0 for col in range(len(self.board[row]))) for row in range(len(self.board)))

    def num_players(self):
        return len(self.players)
    
    def current_player(self):
        return self.players[self.turn_order[0]]

    def max_num_actions(self):
        return 9
    
    def possible_actions(self):
        return [row * 3 + col for row in range(len(self.board)) for col in range(len(self.board[row])) if self.board[row][col] == " "]
    
    def take_action(self, action):
        self.board[action // 3][action % 3] = self.turn_order[0]
        self.turn_order.append(self.turn_order.pop(0))
    
    def has_outcome(self):
        terminated = False
        # check horizontal
        terminated |= self.board[0][0] == self.board[0][1] and self.board[0][0] == self.board[0][2] and self.board[0][0] != " "
        terminated |= self.board[1][0] == self.board[1][1] and self.board[1][0] == self.board[1][2] and self.board[1][0] != " "
        terminated |= self.board[2][0] == self.board[2][1] and self.board[2][0] == self.board[2][2] and self.board[2][0] != " "
        # check vertical
        terminated |= self.board[0][0] == self.board[1][0] and self.board[0][0] == self.board[2][0] and self.board[0][0] != " "
        terminated |= self.board[0][1] == self.board[1][1] and self.board[0][1] == self.board[2][1] and self.board[0][1] != " "
        terminated |= self.board[0][2] == self.board[1][2] and self.board[0][2] == self.board[2][2] and self.board[0][2] != " "
        # check diagonal
        terminated |= self.board[0][0] == self.board[1][1] and self.board[0][0] == self.board[2][2] and self.board[0][0] != " "
        terminated |= self.board[0][2] == self.board[1][1] and self.board[0][2] == self.board[2][0] and self.board[0][2] != " "
        return terminated or not any(" " in row for row in self.board)

    def final_scores(self):
        winners = []
        # check horizontal
        winners += [self.board[0][0]] if self.board[0][0] == self.board[0][1] and self.board[0][0] == self.board[0][2] and self.board[0][0] != " " else []
        winners += [self.board[1][0]] if self.board[1][0] == self.board[1][1] and self.board[1][0] == self.board[1][2] and self.board[1][0] != " " else []
        winners += [self.board[2][0]] if self.board[2][0] == self.board[2][1] and self.board[2][0] == self.board[2][2] and self.board[2][0] != " " else []
        # check vertical
        winners += [self.board[0][0]] if self.board[0][0] == self.board[1][0] and self.board[0][0] == self.board[2][0] and self.board[0][0] != " " else []
        winners += [self.board[0][1]] if self.board[0][1] == self.board[1][1] and self.board[0][1] == self.board[2][1] and self.board[0][1] != " " else []
        winners += [self.board[0][2]] if self.board[0][2] == self.board[1][2] and self.board[0][2] == self.board[2][2] and self.board[0][2] != " " else []
        # check diagonal
        winners += [self.board[0][0]] if self.board[0][0] == self.board[1][1] and self.board[0][0] == self.board[2][2] and self.board[0][0] != " " else []
        winners += [self.board[0][2]] if self.board[0][2] == self.board[1][1] and self.board[0][2] == self.board[2][0] and self.board[0][2] != " " else []
        # check draw
        if len(winners) == 0 and not any(" " in row for row in self.board):
            return [0, 0]
        assert len(winners) > 0, "TicTacToe has no outcome yet!"
        return [1, -1] if winners[0] == "X" else [-1, 1] # TicTacToe is a zero-sum game, note that X is index 0 and O is index 1
