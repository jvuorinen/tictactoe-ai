import numpy as np
from copy import deepcopy
import logging

N_PLAYERS = 2


def ucb(w, n, t, c=1.414):
    if t == 0:
        return 0
    return (w / n) + (c * np.sqrt(np.log(t) / n))


class MockGame:
    def __init__(self, size=25):
        self.size = size
        self.finished = False
        self.turn = np.random.randint(2)
        self.winner = None
        self.depth = 25

    def change_turn(self):
        self.turn = 1 if self.turn == 0 else 0

    def save(self):
        self.saved_state = (self.turn, self.winner, self.finished, self.depth)

    def load(self):
        (self.turn, self.winner, self.finished, self.depth) = self.saved_state

    def get_possible_actions(self):
        return np.random.choice(self.size, self.depth, replace=False)

    def play(self, action):
        self.change_turn()
        self.depth -= 1
        pass

    def simulate_game(self):
        return np.random.randint(2)


class MCTSnode:
    def __init__(self, parent, previous_action, previous_turn, depth, id):
        self.n_won = np.array([0 for _ in range(N_PLAYERS)])
        self.n_sims = 0
        self.parent = parent
        self.previous_turn = previous_turn
        self.depth = depth
        self.id = id
        self.unvisited_children = set()
        self.visited_children = []
        self.never_visited = True
        self.is_leaf = True
        self.ucb = 0
        self.best_child_ucb = 0
        self.best_child = None
        self.previous_action = previous_action

    def __repr__(self):
        return "Node {} at depth {}".format(self.id, self.depth)



class MonteCarloTreeSearch:
    def __init__(self, simulator):
        self.n_rollouts = 0
        self.n_nodes = 1
        self.root = MCTSnode(
            parent=None,
            previous_action=None,
            previous_turn=None,
            depth=0,
            id=0)
        self.sim = deepcopy(simulator)
        self.sim.save()

    def __repr__(self):
        return "nodes {}, rollouts {}".format(self.n_nodes, self.n_rollouts)

    def select(self):
        """Starting at root node R, recursively select optimal child
        nodes (explained below) until a leaf node is reached. Returns
        this leaf node."""
        current = self.root
        actions = []
        while not current.is_leaf:
            # logging.debug("Selection... passed {}".format(current))
            current = current.best_child
            actions.append(current.previous_action)
        for a in actions:
            self.sim.play(a)
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
                             id=self.n_nodes)
                leaf.visited_children.append(c)
                self.sim.play(move)
                if not leaf.unvisited_children:
                    leaf.is_leaf = False
            else:
                # print("Not sure if it's cool to be here")
                c = leaf
            return c

    def simulate(self):
        """Run a simulated playout from C until a result is achieved."""
        score = [i for i in range(N_PLAYERS)]
        self.sim.simulate_game()
        w = self.sim.winner
        if w:
            score[w] += 1
        return np.array(score)

    def backpropagate(self, child, score):
        """Update the current move sequence with the simulation result."""
        current = child
        while current:
            # 1) update current nodes win counter and sim counter
            current.n_won += score
            current.n_sims += 1
            # 2) calculate ucb value for current node (note turn order)
            print("calculating ucb for", current)
            print(current.n_won[current.previous_turn])
            current.ucb = ucb(
                w=current.n_won[current.previous_turn],
                n=current.n_sims,
                t=current.n_sims + 1)  # parents n has not yet been updated
            # 3) update parent's best node
            if current.parent:
                if current.ucb > current.parent.best_child_ucb:
                    current.parent.best_child_ucb = current.ucb
                    current.parent.best_child = current

            current = current.parent

    def search(self, limit=10):
        self.sim.save()
        for i in range(limit):
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
        return self.root.best_child.previous_action


if __name__ == '__main__':
    np.random.seed(42)
    # logging.basicConfig(level=logging.DEBUG)
    game = MockGame()
    mcts = MonteCarloTreeSearch(game)

    mcts.search(10)
    print(mcts)
