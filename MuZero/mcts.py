"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py

Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random as rd
import collections
import numpy as np
from copy import deepcopy
from util.alphazero_util import HelperTrainingExample

# Exploration constant
c_PUCT = 1.25
# Dirichlet noise alpha parameter.
D_NOISE_ALPHA = 0.25
# Number of steps into the episode after which we always select the
# action with highest action probability rather than selecting randomly
TEMP_THRESHOLD = 10
C_2 = 19652
DISCOUNT= 1.05


class DummyNode:
    """
    Special node that is used as the node above the initial root node to
    prevent having to deal with special cases when traversing the tree.
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)
        self.child_R = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None): pass

    def add_virtual_loss(self, up_to=None): pass

    def revert_visits(self, up_to=None): pass

    def backup_value(self, value, up_to=None): pass


class MCTSNode:
    """
    Represents a node in the Monte-Carlo search tree. Each node holds a single
    environment state.
    """

    def __init__(self, state, n_actions, TreeEnv, h=None, action=None, parent=None):
        """
        :param state: State that the node should hold.
        :param n_actions: Number of actions that can be performed in each
        state. Equal to the number of outgoing edges of the node.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param h: heuristic value
        :param action: Index of the action that led from the parent node to
        this node.
        :param parent: Parent node.
        """
        self.TreeEnv = TreeEnv
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth+1
        self.h=h
        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0  # Number of virtual losses on this node
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        self.child_R = np.zeros([n_actions], dtype=np.float32)
        # Save copy of original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value
    
    @property
    def R(self):
        
        #Returns the current total value of the node.
        return self.parent.child_R[self.action]
    @R.setter
    def R(self, value):
        self.parent.child_R[self.action] = value
    
    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    #@property
    def child_Q(self, min_max_stats):
        return min_max_stats.normalize(-self.child_R + DISCOUNT*(self.child_W / (1 + self.child_N)))

    @property
    def child_U(self):
        return (math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N)) *(c_PUCT + math.log((self.N + C_2 + 1)/C_2))

    #@property
    def child_action_score(self, min_max_stats):
        """
        Action_Score(s, a) = Q(s, a) + U(s, a) as in paper. A high value
        means the node should be traversed.
        """
        return self.child_Q(min_max_stats) + self.child_U

    def select_leaf(self, min_max_stats):
        """
        Traverses the MCT rooted in the current node until it finds a leaf
        (i.e. a node that only exists in its parent node in terms of its
        child_N and child_W values but not as a dedicated node in the parent's
        children-mapping). Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode. Note that the
        estimated value and prior probabilities still have to be set with
        `incorporate_estimates` afterwards.
        :param min_max_stats: normalization function to scale value
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            current.N += 1
            # Encountered leaf node (i.e. node that is not yet expanded).
            if not current.is_expanded:
                break
            # Choose action with highest score.
            index, best_move = current.get_best_action(min_max_stats)
            current = current.maybe_add_child(index, best_move)
        return current

    def get_best_action(self, min_max_stats):
        p = self.child_action_score(min_max_stats)
        possible_action, possible_action_keys = self.TreeEnv.get_possible_actions(self.state.head)
        possible_action_ohe = self.TreeEnv.ae.transform(possible_action_keys).sum(axis=0)
        p *= possible_action_ohe
        p[p==0]=-float("inf")
        index = np.argmax(p)
        action_key = self.TreeEnv.ae.inverse_transform([index])[0]
        action=None
        for oper in possible_action:
            if oper.name==action_key:
                action = oper
        if action is None:
            print(p)
        return index, action

    def maybe_add_child(self, index, action):
        """
        Adds a child node for the given action if it does not yet exists, and
        returns it.
        :param action: Action to take in current state which leads to desired
        child node.
        :return: Child MCTSNode.
        """
        if index not in self.children:
            # Obtain state following given action.
            new_state = self.TreeEnv.next_state(self.state, action)
            self.children[index] = MCTSNode(new_state, self.n_actions,
                                             self.TreeEnv,
                                             action=index, parent=self)
            self.children[index].h = self.TreeEnv.get_h(new_state.head, self.children[index].depth)
            #self.children[index].cost = self.children[index].h
            self.child_R[index] = min(1, max(0, self.h - self.children[index].h))
        return self.children[index]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 200
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 200
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, min_max_stats, up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param min_max_stats: Normalization function to scale value
        :param up_to: The node to propagate until.
        """
        # A done node (i.e. episode end) should not go through this code path.
        # Rather it should directly call `backup_value` on the final node.
        # TODO: Add assert here
        # Another thread already expanded this node in the meantime.
        # Ignore wasted computation but correct visit counts.
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs
        # This is a deviation from the paper that led to better results in
        # practice (following the MiniGo implementation).
        #self.R = reward
        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, min_max_stats, up_to=up_to)

    def backup_value(self, value, min_max_stats, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param min_max_stats: Normalization function to scale value
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        min_max_stats.update(-self.R + DISCOUNT*self.Q)
        value = -self.R + DISCOUNT * value
        self.parent.backup_value(value, min_max_stats, up_to)

    def is_done(self):
        return self.TreeEnv.is_done_state(self.state.head, self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly large than 1 to encourage diversity in early steps.
        :return: Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----"*level
        node_string += "Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• N={}".format(self.N)
        node_string += "\n• score:\n{}".format(self.child_action_score[self.action])
        node_string += "\n• Q:\n{}".format(self.child_Q[self.action])
        node_string += "\n• P:\n{}".format(self.child_prior[self.action])
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level+1)

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTS:
    """
    Represents a Monte-Carlo search tree and provides methods for performing
    the tree search.
    """

    def __init__(self, agent_netw, TreeEnv, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=3):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param TreeEnv: Static class that defines the environment dynamics,
        e.g. which state follows from another state when performing an action.
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree
        before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating
        them in conjunction.
        """
        self.agent_netw = agent_netw
        self.TreeEnv = TreeEnv
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None        # Overwritten in initialize_search

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.TreeEnv.initial_state()
        n_actions = self.TreeEnv.n_actions
        self.root = MCTSNode(init_state, n_actions, self.TreeEnv, h = self.TreeEnv.get_h(init_state.head, 0))
        # Number of steps into the episode after which we always select the
        # action with highest action probability rather than selecting randomly
        if int(self.TreeEnv.ep_length)<15:
            self.temp_threshold= int(self.TreeEnv.ep_length/3)
        else:
            self.temp_threshold = int(self.TreeEnv.ep_length/2)
        self.qs = []
        self.rewards.append(self.TreeEnv.get_h(init_state.head, 0))
        self.searches_pi = []
        self.obs = []


    def tree_search(self, min_max_stats, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param min_max_stats: Normalization function to scale values in the tree
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []
        # Failsafe for when we encounter almost only done-states which would
        # prevent the loop from ever ending.
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel:
            failsafe += 1
            #self.root.print_tree()
            #print("_"*50)
            leaf = self.root.select_leaf(min_max_stats)
            # If we encounter done-state, we do not need the agent network to
            # bootstrap. We can backup the value right away.

            if leaf.is_done():
                value = self.TreeEnv.get_h(leaf.state.head, leaf.depth)
                leaf.backup_value(-value, min_max_stats, up_to=self.root)
                continue

            # Otherwise, discourage other threads to take the same trajectory
            # via virtual loss and enqueue the leaf for evaluation by agent
            # network.
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            data = []
            for leaf in leaves:
                data.append(leaf.state)
            action_probs, values = self.agent_netw.step(self.TreeEnv.get_obs_for_states(data))
            for leaf, action_prob, value in zip(leaves, action_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_estimates(action_prob, -value[0]*self.TreeEnv.maxi[0], min_max_stats, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = rd.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action
        

    def take_action(self, index, action):
        """
        Takes the specified action for the root state. The subsequent child
        state becomes the new root state of the tree.
        :param action: Action to take for the root state.
        """
        # Store data to be used as experience tuples.
        
        action_proba=self.root.visits_as_probs() # TODO: Use self.root.position.n < self.temp_threshold as argument
        #print(action_proba, action_proba[index])
        self.qs.append(self.root.Q)
        h = self.TreeEnv.get_h(self.root.children[index].state.head, self.root.children[index].depth)
        reward = min(1, max(0, self.rewards[-1] - h))
        #print(reward)
        self.rewards.append(h)
        self.obs.append(HelperTrainingExample(deepcopy(self.root.state), self.TreeEnv.goal,
                                               action_proba, self.root.Q, reward))

        # Resulting state becomes new root of the tree.
        self.root = self.root.maybe_add_child(index, action)
        del self.root.parent.children




def execute_episode(agent_netw, num_simulations, TreeEnv, test=False):
    """
    Executes a single episode of the task using Monte-Carlo tree search with
    the given agent network. It returns the experience tuples collected during
    the search.
    :param agent_netw: Network for predicting action probabilities and state
    value estimate.
    :param num_simulations: Number of simulations (traverses from root to leaf)
    per action.
    :param TreeEnv: Static environment that describes the environment dynamics.
    :param test: boolean to indicate test/training phase
    :return: The observations for each step of the episode, the policy outputs
    as output by the MCTS (not the pure neural network outputs), the individual
    rewards in each step, total return for this episode and the final state of
    this episode.
    """
    mcts = MCTS(agent_netw, TreeEnv)

    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    min_max_stats = MinMaxStats()
    first_node = mcts.root.select_leaf(min_max_stats)
    probs, val = agent_netw.step(TreeEnv.get_obs_for_state(first_node.state))
    le = val[0][0]*TreeEnv.maxi[0]
    print(le)
    """
    if test:
        mcts.TreeEnv.ep_length = val[0][0]*mcts.TreeEnv.maxi
    """

    first_node.incorporate_estimates(probs[0], -le,  min_max_stats, first_node)
    

    while True:
        possible_action, _ = mcts.TreeEnv.get_possible_actions(mcts.root.state.head)
        mcts.root.inject_noise()
        current_simulations = mcts.root.N
        num_parallel = min(len(possible_action), 8)

        # We want `num_simulations` simulations per action not counting
        # simulations from previous actions.
        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search(min_max_stats, num_parallel = num_parallel)

        # mcts.root.print_tree()
        # print("_"*100)

        index = mcts.pick_action()
        action_key = mcts.TreeEnv.ae.inverse_transform([index])[0]
        print(action_key, mcts.root.Q)
        for oper in possible_action:
            if oper.name==action_key:
                action = oper
        mcts.take_action(index, action)

        if mcts.root.is_done():
            break
    # Computes the returns at each step from the list of rewards obtained at
    # each step. The return is the sum of rewards obtained *after* the step.
    ret = TreeEnv.get_return(mcts.root.state.head, mcts.root.depth)
    val = TreeEnv.get_h(mcts.root.state.head, mcts.root.depth)
    #ret = np.array(ret) * (20/(19 + len(ret)))

    value = val
    print(value, mcts.obs[-1].reward)
    for obs in reversed(mcts.obs):
        value = value + obs.reward
        obs.value = value
        #print(obs.value, obs.reward)


    total_rew = np.sum(mcts.rewards)

    
    return (mcts.obs, mcts.searches_pi, val, total_rew, mcts.root)

