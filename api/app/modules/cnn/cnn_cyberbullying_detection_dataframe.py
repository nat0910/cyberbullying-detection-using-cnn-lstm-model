import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
import pandas as pd
import numpy as np


import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import os  

tokenize_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"tokenize_data.json")

saved_model_file_path = os.path.join(os.path.join(os.path.join(os.getcwd(),'app'),'model'),'cyber_bullying_model_3_13-4-23(90.7029%).h5')
loaded_saved_model = load_model(saved_model_file_path)



class detect_cyberbullying_cnn_dataframe:
    def __init__(self,input):
        self.__loaded_model = loaded_saved_model
        self.__predicted_output = " "
        self.__input = input

    def __tokenizing(self):
        with open(tokenize_data_file,'r') as file :
            tokenized_data = json.load(file)
        tokenizer = tokenizer_from_json(tokenized_data)
        tokenized_data = tokenizer.texts_to_sequences(self.__input)
        return pad_sequences(tokenized_data,maxlen=177,padding='post')

    def __output(self,predicted_output):
        y_pred_type=[]

        for var in predicted_output:
            if(var == 0):
                y_pred_type.append('age')
            if(var == 1):
                y_pred_type.append('ethnicity')
            if(var == 2):
                y_pred_type.append('gender')
            if(var == 3):
                y_pred_type.append('not cyberbullying')
            if(var == 4):
                y_pred_type.append('religion')

        return y_pred_type

    def predict(self):
        self.__df = self.__tokenizing()
        self.__y_pred = self.__loaded_model.predict(self.__df)
        self.__predicted_output =  np.argmax(self.__y_pred,axis = 1)
        self.__label_output = self.__output(self.__predicted_output)
        return  self.__label_output