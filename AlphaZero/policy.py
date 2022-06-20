import numpy as np

from config import StackedStateConfig
from keras.layers import Input, Dense, Flatten, LSTM, Dropout, Masking, ReLU
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

from keras.callbacks import EarlyStopping

from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split


class Rlstm():

    def __init__(self, action_size, facts, hidden_size = StackedStateConfig.MAX_TIME_STEPS):

        X_in = Input(shape = (hidden_size, 2*facts))
        l1 = Masking(mask_value=-1.)(X_in)
        l1 = LSTM(units=128, return_sequences = True)(l1)
        l1 = Dropout(0.2)(l1)
        l1 = LSTM(units =256, return_sequences = True)(l1)
        l1 = Dropout(0.2)(l1)
        l1 = LSTM(units = 128, return_sequences = True)(l1)
        l1 = Dropout(0.2)(l1)
        l1 = LSTM(units = 156)(l1)
        l1 = Dropout(0.2)(l1)

        policy = Dense(action_size, activation='softmax', name='pi')(l1)
        #v = Dense(1)(l1)
        value = Dense(1, activation='relu', name='v')(l1)
        #value = ReLU(max_value=1., name='v')(v)

        # Build model
        self.model = Model(inputs=X_in, outputs=[policy, value])
        optimizer = Adam(lr=0.001)

        self.model.compile(optimizer=optimizer, loss= ['categorical_crossentropy','mean_squared_error'],
         metrics={'pi':'accuracy', 'v':'mean_squared_error'})
        self.model.summary()

    def step(self, X):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        pi, v = self.model.predict(X)

        return pi, v

