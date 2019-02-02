import numpy as np
from copy import deepcopy


def ucb(w, n, t, c = 1.414):
    if t == 0:
        return 0
    return (w/n) + (c * np.sqrt(np.log(t) / n))


class MockGame:
    def __init__(self, size = 10):
        self.size = size
        self.finished = False
        self.state = np.zeros(size).astype(int)
        self.turn = np.random.randint(2)

    def save(self):
        self.saved_state = self.state.copy()
        self.saved_turn = self.turn

    def load(self):
        self.state = self.saved_state.copy()
        self.turn = self.saved_turn

    def get_possible_actions(self):
        n = np.random.randint(1, 6)
        np.random.choice(self.size, n, replace=False)

    def play(self, action):
        self.state[action] += 1

    def simulate_game(self):
        np.random.randint(2)



class MCTSnode:
    def __init__(self, parent, previous_action, depth):
        self.n_won = [0, 0]
        self.n_sims = 0
        self.parent = parent
        self.depth = depth
        self.children = []
        self.unvisited_children = set()
        self.never_visited = True
        self.is_leaf = True
        self.ucb = 0
        self.best_child_ucb = 0
        self.best_child = None
        self.previous_action = previous_action

    def __repr__(self):
        return "MCTS node, state:\n" \
            "wins {}\n" \
               "visited {}\n" \
               "ucb-values {}\n" \
               "parent {}\n" \
               "children {}\n" \
               "best_children {}\n" \
               "is_leaf {}".format(
                    self.n_won,
                    self.n_sims,
                    self.ucbs,
                    self.parent,
                    self.children,
                    self.best_child,
                    self.is_leaf)


class MonteCarloTreeSearch:
    def __init__(self, simulator):
        self.n_rollouts = 0
        self.n_nodes = 1
        self.root = MCTSnode(None, None, 0)
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
        while not current.isleaf:
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
                leaf.unvisited_children.update(self.sim.get_possible_moves())
                leaf.never_visited = False

            if leaf.unvisited_children:
                move = leaf.unvisited_children.pop()
                c = MCTSnode(parent = leaf,
                             previous_action = move,
                             depth = leaf.depth + 1)
                self.sim.play(move)
                if leaf.unvisited_children:
                    leaf.is_leaf = False
            return c

    def simulate(self):
        """Run a simulated playout from C until a result is achieved."""
        result = self.sim.simulate_game()
        return result

    def backpropagate(self, child, result):
        """Update the current move sequence with the simulation result."""
        current = child
        while current:
            # 1) update current nodes win counter and sim counter
            current.n_sims += 1
            # 2) calculate ucb value for current node (note turn order)
            current.ucb = ucb(
                w= "???",
                n=current.n_sims,
                t=current.n_sims+1) # parents n has not yet been updated
            # 3) update parent's best node
            if current.ucb > current.parent.best_child_ucb:
                current.parent.best_child_ucb = current.ucb
                current.parent.best_child = current

            current = current.parent


    def mcts_search(self, limit=10):
        for i in range(limit):
            l = self.select()
            c = self.expand(l)
            r = self.simulate()
            self.backpropagate(c, r)
            self.n_rollouts += 1
        return self.root.best_child.previous_action


if __name__ == '__main__':
    np.random.seed(123)
    mock = MockGame()
    mcts = MonteCarloTreeSearch(mock)



