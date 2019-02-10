import numpy as np
from copy import deepcopy
from math import sqrt, log
from collections import Counter
from datetime import datetime

from tictactoe import TicTacToe

N_PLAYERS = 2

def ucb(w, n, t, c=1.414):
    if t == 0:
        return 0
    return (w / n) + (c * sqrt(log(t) / n))

def get_best_child(node):
    return max(node.visited_children, key=(lambda x: x.ucb))


class MillisecondClock:
    def __init__(self):
        self.t0 = datetime.now()

    def get_ms(self):
        dt = datetime.now() - self.t0
        seconds = dt.seconds
        ms = dt.microseconds/1000
        return seconds*1000 + ms



class MockGame:
    def __init__(self, size=25):
        self.size = size
        self.finished = False
        self.turn = np.random.randint(2)
        self.winner = None
        self.depth = 10

    def change_turn(self):
        self.turn = 1 if self.turn == 0 else 0

    def get_previous_turn(self):
        return 1 if self.turn == 0 else 0

    def save(self):
        self.saved_state = (self.turn, self.winner, self.finished, self.depth)

    def load(self):
        (self.turn, self.winner, self.finished, self.depth) = self.saved_state

    def get_possible_actions(self):
        return np.random.choice(self.size, self.depth, replace=False)

    def play_action(self, action):
        self.change_turn()
        self.depth -= 1

    def play_random_game(self):
        return np.random.randint(2)


class MCTSnode:
    def __init__(self, parent, previous_action, previous_turn, depth, node_id, is_root=False):
        self.n_won = np.array([0 for _ in range(N_PLAYERS)])
        self.n_sims = 0
        self.win_rate = 0
        self.parent = parent
        self.previous_turn = previous_turn
        self.depth = depth
        self.id = node_id
        self.unvisited_children = set()
        self.visited_children = []
        self.never_visited = True
        self.is_leaf = True
        self.ucb = 0
        self.best_child = None
        self.previous_action = previous_action
        self.is_root = is_root

    def __repr__(self):
        return "Node {}".format(self.id)



class MonteCarloTreeSearch:
    def __init__(self, simulator):
        self.n_rollouts = 0
        self.n_nodes = 0
        self.root = MCTSnode(
            parent=None,
            previous_action=None,
            previous_turn=simulator.get_previous_turn(),
            depth=0,
            node_id=0,
            is_root=True)
        self.sim = deepcopy(simulator)
        self.sim.save()
        self.depth_counter = Counter()

    def __repr__(self):
        return "MCTS object with {} nodes explored\nDepths reached\n{}".format(
            self.n_nodes, "\n".join("{}: {}".format(a, b) for a, b in mcts.depth_counter.items()))

    def select(self):
        """Starting at root node R, recursively select optimal child
        nodes (explained below) until a leaf node is reached. Returns
        this leaf node."""
        current = self.root
        actions = []
        while not current.is_leaf:
            # print("Selection... considering between:")
            # print(["{} ({:.2f})".format(n, n.ucb) for n in current.visited_children])
            # print("Node's stats say that best child is {} ({})".format(current.best_child, current.best_child.ucb))
            current = current.best_child
            actions.append(current.previous_action)
            # print("Selected", current)
            # print()

        for a in actions:
            self.sim.play_action(a)
        # print("--------------------------------------")
        return current

    def expand(self, leaf):
        """If L is a not a terminal node (i.e. it does not end the game)
        then create child node and return it. NOTE: this function
        will not work correctly if self.sim's state is not synced
        properly."""
        if not self.sim.finished:
            if leaf.never_visited:
                children = self.sim.get_possible_actions()
                leaf.unvisited_children.update(children)
                leaf.never_visited = False

            if leaf.unvisited_children:
                self.n_nodes += 1
                move = leaf.unvisited_children.pop()
                c = MCTSnode(parent=leaf,
                             previous_action=move,
                             previous_turn=self.sim.turn,
                             depth=leaf.depth + 1,
                             node_id=self.n_nodes)
                self.depth_counter[leaf.depth + 1] += 1
                leaf.visited_children.append(c)
                self.sim.play_action(move)
                if not leaf.unvisited_children:
                    leaf.is_leaf = False
            else:
                # print("Not sure if it's cool to be here")
                c = leaf
            return c

    def simulate(self):
        """Run a simulated playout from C until a result is achieved."""
        score = [0 for _ in range(N_PLAYERS)]
        self.sim.play_random_game()
        w = self.sim.winner
        if w in (0,1):
            score[w] += 1
        return np.array(score)

    def backpropagate(self, child, score):
        """Update the current move sequence with the simulation result."""
        current = child
        while current:
            # 1) update current nodes win counter and sim counter
            # print("Node before:")
            # print(current)
            # print(current.n_won)
            # print(current.ucb)
            current.n_won += score
            current.n_sims += 1
            # 2) calculate ucb value for current node (note turn order)

            # print(current.n_won[current.previous_turn])

            if not current.is_root:
                # print("Before updating, current node is {} with ucb {}".format(current, current.ucb))
                # print("Its parent's best child is {}".format(current.parent.best_child))
                current.ucb = ucb(
                    w=current.n_won[current.previous_turn],
                    n=current.n_sims,
                    t=current.n_sims + 1)  # parents n has not yet been updated

                current.parent.best_child = get_best_child(current.parent)
                # print("After updating, current node is {} with ucb {}".format(current, current.ucb))
                # print("Its parent's best child is {} with ucb {}".format(current.parent.best_child, current.parent.best_child.ucb))
                # print()
            current = current.parent

    def search(self, limit_milliseconds=1000):
        clock = MillisecondClock()
        # print('saved sim')
        while clock.get_ms() < limit_milliseconds:
            for i in range(100):
                # logging.debug("Sim {}...".format(i))
                self.sim.load()
                l = self.select()
                # logging.debug("Selected leaf node {}".format(l))
                c = self.expand(l)
                # logging.debug("Selected child node{}".format(c))
                s = self.simulate()
                self.backpropagate(c, s)
                self.n_rollouts += 1
                # logging.debug("...")
                # print([n.best_child for n in mcts.root.best_child.visited_children])
        self.sim.load()

    def get_best_action(self):
        return max(self.root.visited_children, key=(lambda x: x.n_sims)).previous_action


if __name__ == '__main__':
    np.random.seed(42)
    # logging.basicConfig(level=logging.DEBUG)
    # game = MockGame()
    # mcts = MonteCarloTreeSearch(game)

    # mcts.search(1000)
    # print(mcts)


    ttt = TicTacToe(size=5, win_length=4)
    mcts = MonteCarloTreeSearch(ttt)
    mcts.search(1000)
    mcts.get_best_action()
    # print([n.previous_turn for n in r.visited_children[0].visited_children[1].visited_children])
