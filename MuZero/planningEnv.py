import numpy as np
from copy import deepcopy

from util.stacked_state import StackedState
from static_env import StaticEnv
from util.alphazero_util import parse_global_list_training, HelperTrainingExample
from config import AlphaZeroConfig as config
from pyperplan import heuristics
from pyperplan.heuristics.relaxation import hFFHeuristic as hff
from pyperplan.heuristics.lm_cut import LmCutHeuristic as lm_cut
from pyperplan.search import searchspace

class PlanningAgent(StaticEnv):
    """
    AlphaZero agent. The agent uses AlphaZero algorithm which uses Monte Carlo Tree Search
    """
    def __init__(self, n_actions, task, ae, dict_fact, maxi=[1,1,0],
                 max_simulation= config.MAX_SIMULATION):
        """
        Contructor of AlphaZero Agent
        :param n_action: Maximum number of actions
        :param task: task containing the inital_state and goal state
        :param ae: an action encoder
        :param dict_fact: a dictionary for all facts(max. limited) in the domain
        :param maxi: contains maximum and min values
        :param max_simulation: MCTS max simulation

        """
        self.n_actions = n_actions
        self.dict_facts = dict_fact
        self.ae = ae
        self.task = task
        self.maxi = maxi
        self.ep_length = max_simulation
        self.goal = task.goals
        self.stacked_state = StackedState(task.initial_state)
        self.step_idx = 0

    def reset(self):
        """
        Resets the agent to the intial_state
        :return initial_state, goal state, rewards, if goal_reached 
        """
        state = StackedState(self.task.initial_state)
        goal = self.task.goals
        return state, goal, 0, False, None

   
    def step(self, action):
        """
        Takes an action
        :param action: an action to take
        :return: state after action is taken, if goal is reached, rewards
        """
        self.step_idx += 1
        ns = action.apply(self.stacked_state.head)
        #ns = self.stacked_state.head.union(ns)
        #reward = self.get_return(ns, self.step_idx)
        self.stacked_state.append(ns)
        done = self.task.goal_reached(ns) or step_idx>=self.ep_length
        return self.stacked_state, done, None, None

    def get_possible_actions(self, state):
        """
        Returns all the possible actions available from a state
        :param state: state 
        :return: possible actions and their names
        """
        possible_action = [op for op in self.task.operators if op.applicable(state)]
        possible_action_keys = [oper.name for oper in possible_action]
        return possible_action, possible_action_keys

    def is_done_state(self, state, step_idx):
        """
        Checks terminal conditions
        :param state: the state the agent is in
        :param step_idx: number of actions taken so far
        :return: true if terminal conditions are satisfied
        """
        return self.task.goal_reached(state) or step_idx>=self.ep_length

    def get_h(self, state, step_idx):
        """
        Returns the heuristic value
        :param state: state the agent is in
        :param step_idx: number of actions taken so far
        :return: the heuristic value of the current state
        """

        task = deepcopy(self.task)
        task.initial_state = state
        node = searchspace.make_root_node(task.initial_state)
        heuristics = hff(task)
        h = heuristics(node)
        return h

    def get_return(self, state, step_idx):

        if self.task.goal_reached(state):
            return 1
        else:
            return 0


    def initial_state(self):
        return StackedState(self.task.initial_state)

    def get_obs_for_state(self, state):
        """
        converts a state to the format of the neural network input
        """
        X = state.get_deep_representation_stack1(self.dict_facts, self.goal)
        X = np.array(X)
        X = np.reshape(X, (1, X.shape[0], X.shape[1]))

        return X

    def get_obs_for_states(self, states):
        """
        Changes the observed states into an input representation of the neural network
        """
        X=[]

        for state in states:
            F = state.get_deep_representation_stack1(self.dict_facts, self.goal)
            X.append(F)

        X = np.array(X)

        return X
           
    @staticmethod
    def next_state(state, action):
        """
        Gives the next state given an action
        :param: state: the current state
        :param: action: an action to take
        :return: the state after an action is taken
        """
        n_state = action.apply(state.head)
        next_stacked_state = deepcopy(state)
        next_stacked_state.append(n_state)        
        return next_stacked_state

    

