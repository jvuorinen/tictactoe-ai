import numpy as np
from random import sample


class TicTacToe:
    def __init__(self):
        # self.boards contains a boolean board for each player
        self.board_p1 = np.zeros((3, 3)).astype(np.bool)
        self.board_p2 = np.zeros((3, 3)).astype(np.bool)
        self.turn = 1
        self.winner = None
        self.finished = False

    def _change_turn(self):
        self.turn = 2 if self.turn == 1 else 1
        # print("turn: {}".format(self.turn))

    def _get_active_player_board(self):
        b = self.board_p1 if (self.turn == 1) else self.board_p2
        return b

    def _get_possible_moves(self):
        free = (~self.board_p1) & (~self.board_p2)
        return [move for move in zip(*np.where(free))]

    def _check_move_validity(self, move):
        # Make sure "move" has correct format and neither player already has a mark there
        assert (type(move) == tuple) & (len(move) == 2) & (0 <= sum(move) <= 4)
        return ~(self.board_p1[move] | self.board_p2[move])

    def _mark_move(self, move):
        # print("Marking {} for player number {}".format(move, self.turn))
        b = self._get_active_player_board()
        b[move] = True

    def _find_winner(self):
        for i, board in ((1, self.board_p1), (2, self.board_p2)):
            if np.any(np.sum(board, axis=0) == 3): return i
            elif np.any(np.sum(board, axis=1) == 3): return i
            elif np.all(np.diag(board)): return i
            elif np.all(np.diag(np.fliplr(board))): return i
        return False

    def _check_for_victory(self):
        w = self._find_winner()
        if w:
            # print("winner found: {}".format(w))
            self.winner = w
            self.finished = True
            self.turn = 0

    def _check_for_stalemate(self):
        moves = self._get_possible_moves()
        if len(moves) == 0:
            self.finished = True
            self.turn = 0
            self.winner = 0

    def play_move(self, move):
        if self._check_move_validity(move):
            self._mark_move(move)
            self._change_turn()
            self._check_for_victory()
            self._check_for_stalemate()
        else:
            print("Move {} not valid".format(move))


    def play_random_move(self):
        if self.finished:
            print("Game is already finished!")
            return

        moves = self._get_possible_moves()

        if len(moves) >= 1:
            # print("Possible moves: ", moves)
            move = sample(moves, 1)[0]
            self.play_move(move)
        else:
            print("No moves possible!")


    def play_random_game(self):
        i = 0
        while self.finished == False:
            self.play_random_move()
            i += 1
            if i > 100:
                return


    def print_state(self):
        b0 = np.zeros((3, 3)).astype(np.int)
        b1 = self.board_p1.astype(int)
        b2 = 2*self.board_p2.astype(int)
        b = b0 + b1 + b2
        print("Finished: {}".format(self.finished))
        print("Current turn: {}".format(self.turn))
        print("Winner: {}".format(self.winner))
        print(b)

    def reset(self):
        self.board_p1[:] = False
        self.board_p2[:] = False
        self.turn = 1
        self.winner = None
        self.finished = False

def play_random_games(n):
    ttt = TicTacToe()

    wins = [0, 0, 0]

    for i in range(n):
        ttt.reset()
        ttt.play_random_game()
        # ttt.print_state()
        w = ttt.winner
        wins[w] += 1

    print(wins)


if __name__ == '__main__':
    play_random_games(1000)
