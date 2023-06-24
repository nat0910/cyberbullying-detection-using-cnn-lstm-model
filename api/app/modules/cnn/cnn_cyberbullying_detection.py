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

saved_model_file_path = os.path.join(os.path.join(os.path.join(os.getcwd(),'app'),'model'),'cyber_bullying_model_3_13-4-23(90.7029%).h5')
loaded_saved_model = load_model(saved_model_file_path)

tokenize_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"tokenize_data.json")

class detect_cyberbullying_cnn_text:
    def fit(self,text):
        self.__text = text
        self.__loaded_model = loaded_saved_model
        self.__predicted_output = " "

    def __tokenize(self):

        with open(tokenize_data_file,'r') as file : 
            tokenized_data = json.load(file)
        tokenizer = tokenizer_from_json(tokenized_data)
        self.__text = tokenizer.texts_to_sequences(self.__text)
        # print(self.__text)
        self.__text = pad_sequences(self.__text,maxlen=177,padding='post')

    def __output(self):
        if(self.__predicted_output == 0):
            return 'age'
        if(self.__predicted_output == 1):
            return 'ethnicity'
        if(self.__predicted_output == 2):
            return 'gender'
        if(self.__predicted_output == 3):
            return 'not cyberbullying'
        if(self.__predicted_output == 4):
            return 'religion'

    def model_metrics(self):
        metrics = '  precision    recall  f1-score   support\n\n              age       0.98      0.94      0.96      2399\n        ethnicity       0.97      0.97      0.97      2366\n           gender       0.88      0.84      0.86      2371\nnot_cyberbullying       0.77      0.85      0.81      2375\n         religion       0.94      0.93      0.94      2439\n\n         accuracy                           0.91     11950\n        macro avg       0.91      0.91      0.91     11950\n     weighted avg       0.91      0.91      0.91     11950\n'
        print(metrics)

    def predict(self):
        self.__tokenize()
        y_pred =  self.__loaded_model.predict(self.__text)
        self.__predicted_output =  np.argmax(y_pred)
        
        return self.__output()


