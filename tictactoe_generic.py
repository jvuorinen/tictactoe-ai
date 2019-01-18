import numpy as np
from random import sample
from itertools import groupby


def longest_consecutive_run(a):
    longest = 0
    for i, j in groupby(a):
        longest = max(longest, sum(j))
    return longest


def check_if_move_leads_to_win(board, move, win_length):
    max_idx = board.shape[0] - 1
    i, j = move
    row = board[i,:]
    column = board[:, j]

    offset_1 = j - i
    offset_2 = (max_idx - i) - j
    diag_1 = board.diagonal(offset=offset_1)
    diag_2 = np.fliplr(board).diagonal(offset=offset_2)

    for check in [row, column, diag_1, diag_2]:
        if longest_consecutive_run(check) >= win_length:
            return True
    return False


class TicTacToe:
    def __init__(self, size=3, win_length=3):
        # self.boards contains a boolean board for each player
        self.size = size
        self.n_cells = size ** 2
        self.win_length = win_length
        self.board_p1 = np.zeros((self.size, self.size)).astype(np.bool)
        self.board_p2 = np.zeros((self.size, self.size)).astype(np.bool)
        self.turn = sample((1,2), 1)[0]
        self.i = 1
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
        assert (type(move) == tuple) & (len(move) == 2) & (0 <= sum(move) <= 2 * (self.size - 1))
        return ~(self.board_p1[move] | self.board_p2[move])

    def _mark_move(self, move):
        # print("Marking {} for player number {}".format(move, self.turn))
        b = self._get_active_player_board()
        b[move] = True
        # Mark possible victory
        if check_if_move_leads_to_win(b, move, self.win_length):
            self.winner = self.turn
            self.finished = True
            self.turn = 0
            return
        # Mark possible stalemate
        if self.i == self.n_cells:
            self.finished = True
            self.turn = 0
            self.winner = 0
            return
        self.i += 1

    def play_move(self, move):
        if self._check_move_validity(move):
            self._mark_move(move)
            if self.finished:
                return True
            self._change_turn()

        else:
            # print("Move {} not valid".format(move))
            return False


    def play_random_move(self):
        if self.finished:
            # print("Game is already finished!")
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
            # print(self)
            self.play_random_move()
            i += 1
            if i > 100:
                return

    def __repr__(self):
        b0 = np.zeros((self.size, self.size)).astype(np.int)
        b1 = self.board_p1.astype(int)
        b2 = 2 * self.board_p2.astype(int)
        t0 = str(b0 + b1 + b2) + '\n'
        t1 = "Finished: {}\n".format(self.finished)
        t2 = "Current turn: {}\n".format(self.turn)
        t3 = "Winner: {}\n".format(self.winner)
        t4 = "Turns passed: {}\n".format(self.i)
        return t0 + t1 + t2 + t3 + t4

    def reset(self):
        self.board_p1[:] = False
        self.board_p2[:] = False
        self.i = 1
        self.turn = sample((1,2), 1)[0]
        self.winner = None
        self.finished = False

    def play_random_games(self, n):
        wins = [0, 0, 0]
        for i in range(n):
            self.reset()
            self.play_random_game()
            # print(self)
            w = self.winner
            # print(w)
            wins[w] += 1
        print(wins)


if __name__ == '__main__':
    ttt = TicTacToe(size=5, win_length=4)
    ttt.play_random_games(1000)
    # print(ttt)
