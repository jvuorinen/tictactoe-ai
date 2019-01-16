import numpy as np


class TicTacToe:
    def __init__(self):
        # self.boards contains a boolean board for each player
        self.board_p1 = np.zeros((3, 3)).astype(np.bool)
        self.board_p2 = np.zeros((3, 3)).astype(np.bool)
        self.turn = 1

    def _change_turn(self):
        self.turn = 2 if self.turn == 1 else 1
        # print("turn: {}".format(self.turn))

    def _get_active_player_board(self):
        if self.turn == 1:
            return self.board_p1
        else:
            return self.board_p2

    def _check_move_validity(self, move):
        # Make sure "move" has correct format and neither player already has a mark there
        assert (type(move) == tuple) & (len(move) == 2) & (0 <= sum(move) <= 4)
        return ~(self.board_p1[move] | self.board_p2[move])

    def _mark_move(self, move):
        b = self._get_active_player_board()
        b[move] == True

    def play(self, move):
        if self._check_move_validity(move):
            self._mark_move(move)
            self._change_turn()

        else:
            print("Move {} not valid".format(move))

    def print_state(self):
        b0 = np.zeros((3, 3)).astype(np.int)
        b1 = self.board_p1.astype(int)
        b2 = self.board_p2.astype(int)
        b = b0 + b1 + b2
        print("Current turn: {}".format(self.turn))
        print(b)

if __name__ == '__main__':
    ttt = TicTacToe()

    move = (3,3)


    ttt.print_state()
    ttt._change_turn()

