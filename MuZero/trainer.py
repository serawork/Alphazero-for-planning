
import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import clone_model

from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split




class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, Policy, learning_rate=0.0001):

        self.step_model = Policy()

        def train(X, action_proba, v):

           
            epochs = 50       # Number of training epochs
            batch_size = 64          # Batch size
            es_patience = 10       # Patience fot early stopping
            #A = np.clip(A + A.T, 0, 1)

            x_train, action_proba_train, v_train = shuffle(X, action_proba, v)

            #To balance the values
            class_weights = class_weight.compute_class_weight('balanced',
                                             np.unique(v),
                                             v)


            class_weights = dict(enumerate(class_weights))
            
            
            print("Fitting the model!")
            print("------------")

            
            hist = self.step_model.model.fit(x_train,
                  [action_proba_train, v_train],
                  verbose=2,
                  batch_size=batch_size,
                  validation_split=0.1,
                  class_weight=class_weights,
                  epochs=epochs,
                  callbacks=[
                      EarlyStopping(patience=es_patience, restore_best_weights=True) ]
                   )

            self.step_model.model.save("checkpoint.hdf5", overwrite=True)
            return hist

        self.train = train
