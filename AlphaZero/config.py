class AlphaZeroConfig:
    """
    A static class that contains most of the configuration for AlphaZero Algorithm.
    """

    # Maximum of length for DL model planner
    MAX_LENGTH_PLANNER = 100

    MAX_SIMULATION_AGENT = 20

    #Path to folder containing test problems
    "test_prob/"

    # Path of the best model. Can contain absolute path.
    BEST_MODEL_PATH = "best_model.hdf5"

    # Path of the current model or the checkpoint. Can contain absolute path.
    CURRENT_MODEL_PATH = "checkpoint.hdf5"


    # Number of simulations of the MCTS
    MCTS_SIMULATION = 60
    MAX_SIMULATION = 20


class StackedStateConfig:
    """
    A static class that contains most of the configuration for the Stacked State.
    """

    # Maximum of time steps of the stacked state
    MAX_TIME_STEPS = 2
