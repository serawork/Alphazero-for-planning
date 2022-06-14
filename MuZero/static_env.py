

class StaticEnv:
    """
    Abstract class for a static environment. A static environment follows the
    same dynamics as a normal, stateful environment but without saving any
    state inside. As a consequence, all prior information (e.g. the current
    state) has to be provided as a parameter.
    The MCTS algorithm uses static environments because during the tree search,
    it jumps from one state to another (not following the dynamics), such that
    an environment which stores a single state does not make sense.
    """

    @staticmethod
    def next_state(state, action):
        """
        Given the current state of the environment and the action that is
        performed in that state, returns the resulting state.
        :param state: Current state of the environment.
        :param action: Action that is performed in that state.
        :return: Resulting state.
        """
        raise NotImplementedError


    