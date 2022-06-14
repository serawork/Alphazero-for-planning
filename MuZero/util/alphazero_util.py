from copy import deepcopy
import numpy as np

from util.stacked_state import StackedState



class HelperTrainingExample:
    """
    Class that is used to collect the data from the MCTS simulation
    """
    def __init__(self, stacked_state, goal, action_proba, value, reward=None):
        """
        :param stacked_state: observed states
        :param goal: the goal state
        :param action_proba: probability of actions
        :param value: value os observed states
        :param: reward: reward of taking an action
        """

        self.stacked_state = stacked_state
        self.action_proba = action_proba
        self.goal = goal
        self.value = value
        self.reward=reward


def parse_global_list_training(global_list_training, dict_facts):
    """
    Parse the list of the object HelperTrainingExample that is ready
    to be input of the neural network ( as the targetted output )
    :param global_list_training: list object of HelperTrainingExample
    :param dict_facts: dictionary of all predicates
    :return: list of state representation, list of action probability, value, reward
    """
    deep_repr_state = []
    action_proba = []
    value = []
    X=[]
    reward = []


    for i in global_list_training:
        F = i.stacked_state.get_deep_representation_stack1(dict_facts, i.goal)
        X.append(F)
        action_proba.append(i.action_proba)
        value.append(i.value)
        reward.append(i.reward)

    return X, action_proba, value, reward

def action_spaces_new(task):
    """
    Generator all possible actions in the game
    :return:
    """
    operators = task.operators
    all_list_action =  [oper.name for oper in operators]
    return all_list_action



