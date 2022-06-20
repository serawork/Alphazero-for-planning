from copy import deepcopy
from collections import deque
import numpy as np
from config import StackedStateConfig
from config import AlphaZeroConfig as config

from feature import *

class StackedState:
    """
    Represents the past observed state into a stack for neural network input
    """

    def __init__(self, state,
                 max_len=StackedStateConfig.MAX_TIME_STEPS):
        """
        :param state: current state to be added to stack
        :param max_len: number of past states 
        """
        self.state=state
        self.deque_collection = deque(maxlen=max_len)
        self.deque_collection.append(state)
        self.head = state
        self.max_len = max_len
    
    def get_deep_representation_stack1(self, dict_facts, goal):
        """
        Converts stack of observed states into neural network reperesentation
        :param dict_facts: dictionary of predicates 
        :param goal: goal state
        :return: stack of observed states followed by goal state
        """
        stacked_features = []
        stack =[]
        gf1 = get_bow(goal, dict_facts)
        for state in self.deque_collection:
            feature = get_bow(state, dict_facts)
            feature.extend(gf1)
            stacked_features.append(feature)
        sta = [-1]*(2*len(dict_facts))
        if len(stacked_features) < StackedStateConfig.MAX_TIME_STEPS:
            for i in range(StackedStateConfig.MAX_TIME_STEPS - len(stacked_features)):
                stack.append(sta)

        stack.extend(stacked_features)
        stack = np.array(stack)        
        return stack

    def append(self, state):
        self.deque_collection.append(deepcopy(state))
        self.head = deepcopy(state)

    def delete(self, index):
        del self.deque_collection[index]

    def __repr__(self):
        returned_string = ""
        for state in self.deque_collection:
            li = sorted(list(state))
            returned_string += str(li) + " then "
        return returned_string

    def __eq__(self, other):
        return (
            self.head == other.head
        )

    def __hash__(self):
        return hash(self.stacked_state.__repr__())