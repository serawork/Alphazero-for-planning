from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np 
import pandas as pd


class ActionEncoder():
    """
    Class that will be used to encode all the key into
    One Hot Encoder and Label Encoder (map integer into action key)
    """

    def __init__(self):
        self.dict_act_key_to_mirror_key = {}
        self.dict_index_act_to_mirror_index = {}

    def fit(self, list_all_action):
        """
        Fit the encoder of Label Encoder. So it can map an integer to an action key.
        Also fit the One Hot Encoder.
        :param list_all_action: list of all possible action keys in the game
        :return:
        """
        all_action_df= pd.DataFrame(list_all_action, columns=['Action_types'])
        self.le = preprocessing.LabelEncoder()
        all_action_df['Action_types_cat']=self.le.fit_transform(all_action_df['Action_types'])
        self.ct = ColumnTransformer([('onehot_encoder', OneHotEncoder(categories='auto', sparse=False), [0])], remainder='passthrough')
        #list_all_action = list_all_action.reshape(len(list_all_action), 1)
        self.ct.fit(all_action_df[['Action_types_cat']])
            

    def transform(self, data):
        """
        Transform the action key into one hot encoder
        :param data: action key
        :return:
        
        poss_action_df= pd.DataFrame(data, columns=['Action_types'])
        print(poss_action_df)
        poss_action_df['Action_types_cat'] = self.le.transform(poss_action_df['Action_types'])
        data = self.onehot_encoder.transform(poss_action_df[['Action_types_cat']])
        return data
        """
        data = self.le.transform(data)
        data = data.reshape(len(data), 1)
        data = self.ct.transform(data)
        return data

    def inverse_transform(self, data):
        """
        Transform the integer into the action key based on the dictionary
        fitted in the Label Encoder.
        :param data: int, an encoded label integer
        :return: action key
        """
        data = self.le.inverse_transform(data)
        return data
