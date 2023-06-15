seed_value = 42
import random
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler


random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

import os  
for filename in os.listdir('model'):
        model_file = os.path.join('model', filename)

class BiLSTM_Sentiment_Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, lstm_layers, bidirectional,batch_size, dropout,DEVICE):
        super(BiLSTM_Sentiment_Classifier,self).__init__()
        
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.DEVICE = DEVICE

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=lstm_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden):
        self.batch_size = x.size(0)
        ##EMBEDDING LAYER
        embedded = self.embedding(x)
        #LSTM LAYERS
        out, hidden = self.lstm(embedded, hidden)
        #Extract only the hidden state from the last LSTM cell
        out = out[:,-1,:]
        #FULLY CONNECTED LAYERS
        out = self.fc(out)
        out = self.softmax(out)

        return out, hidden

    def init_hidden(self, batch_size):
        #Initialization of the LSTM hidden and cell states
        h0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(self.DEVICE)
        c0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(self.DEVICE)
        hidden = (h0, c0)
        return hidden


class detect_cyberbullying_ltsm_text:
    def __intialize_model(self):
        self.model = BiLSTM_Sentiment_Classifier(self.VOCAB_SIZE,
                                                 self.EMBEDDING_DIM,
                                                 self.HIDDEN_DIM,
                                                 self.NUM_CLASSES,
                                                 self.LSTM_LAYERS,
                                                 self.BIDIRECTIONAL,
                                                 self.BATCH_SIZE,
                                                 self.DROPOUT,
                                                 self.DEVICE)
        import os  
        for filename in os.listdir('model'):
                if filename == 'cyber_2_state_dict.pt':
                    model_file = os.path.join('model', filename)

        if  self.DEVICE  == 'cuda':
            self.model.load_state_dict(torch.load(model_file ))
            self.model.to(self.DEVICE)
            self.model.eval()
        else:
            self.model.load_state_dict(torch.load(model_file,map_location=self.DEVICE))
            self.model.eval()

    def __init__(self) :
        self.NUM_CLASSES = 5 #We are dealing with a multiclass classification of 5 classes
        self.HIDDEN_DIM = 100 #number of neurons of the internal state (internal neural network in the LSTM)
        self.LSTM_LAYERS = 1 #Number of stacked LSTM layers
        self.LR = 3e-4 #Learning rate
        self.DROPOUT = 0.5 #LSTM Dropout
        self.BIDIRECTIONAL = True #Boolean value to choose if to use a bidirectional LSTM or not
        self.EPOCHS = 10 #Number of training epoch
        self.EMBEDDING_DIM = 200
        self.BATCH_SIZE = 32
        self.max_len = 188
        self.VOCAB_SIZE = 41888
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.__intialize_model()
        self.text = ''

    def fit(self,input:str):
        self.text = input

    def Tokenize_input(self):
        seq_len =self.max_len
        vocab_to_init = []
        text_int = []
        r = []

        for filename in os.listdir('modules'):
            if filename == "ltsm_tokenize_2.json":
                tokenize_data_file = os.path.join('modules',filename)

        with open(tokenize_data_file,'r') as file:
            vocab_to_init = json.load(file)
            
           
        for word in self.text.split():
            if not word in vocab_to_init:
                r.append(0)
                continue

            r.append(vocab_to_init[word])

        text_int.append(r)

        features = np.zeros((len(text_int), seq_len), dtype = int)
        for i, review in enumerate(text_int):
            if len(review) <= seq_len:
                zeros = list(np.zeros(seq_len - len(review)))
                new = zeros + review
            else:
                new = review[: seq_len]
            features[i, :] = np.array(new)
        
        return  np.array(features)
    
    def label_output(self,predicted_output):
        if(predicted_output == 0):
            return 'religion'
        if(predicted_output == 1):
            return 'age'
        if(predicted_output == 2):
            return 'ethnicity'
        if(predicted_output == 3):
            return 'gender'
        if(predicted_output == 4):
            return 'not cyberbullying'
    

    def predict(self):
        tokenized_text = self.Tokenize_input()
        inputs_txt = torch.tensor(tokenized_text) 
        inputs_txt = inputs_txt.to(self.DEVICE)

        with torch.no_grad():
            h = self.model.init_hidden(1)
            pred_output , h = self.model(inputs_txt,h)
            pred_output = torch.argmax(pred_output,dim=1)
            pred_output = pred_output.squeeze().tolist()
            return self.label_output(pred_output)