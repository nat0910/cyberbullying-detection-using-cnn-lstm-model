import re
import string
import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))

ps=PorterStemmer()
lem = WordNetLemmatizer()




class preprocessing_input_text_cnn:

    def __init__(self):
        self.__text = ''

    def fit(self,text:str):
        self.__text = text
        
    def __clean_hashtags(self):
        self.__text = re.sub(r'\s*#\w+','',self.__text)

    def __remove_emoji(self):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        self.__text = emoji_pattern.sub(r'', self.__text)
    
    def __remove_url(self):
        banned_list = string.punctuation
        self.__text = re.sub(r"(?:\@|https?\://)\S+", " ", self.__text)
        self.__text = re.sub(r'[^\x00-\x7f]',r'', self.__text)
        self.__text = self.__text.translate(str.maketrans(' ',' ',banned_list))
        return self.__text
    
    def __remove_stopwords(self):
        clean_text = []
        for el in word_tokenize(self.__text):
            if not el in stop_words:
                clean_text.append(el)
        self.__text = clean_text

    def __decontract(self):
        self.__text = self.__text.replace('\r',' ').replace('\n',' ').lower()
        self.__text = re.sub(r"won\'t", " will not", self.__text)
        self.__text = re.sub(r"won\'t've", " will not have", self.__text)
        self.__text = re.sub(r"can\'t", " can not", self.__text)
        self.__text = re.sub(r"don\'t", " do not", self.__text)
        
        self.__text = re.sub(r"can\'t've", " can not have", self.__text)
        self.__text = re.sub(r"ma\'am", " madam", self.__text)
        self.__text = re.sub(r"let\'s", " let us", self.__text)
        self.__text = re.sub(r"ain\'t", " am not", self.__text)
        self.__text = re.sub(r"shan\'t", " shall not", self.__text)
        self.__text = re.sub(r"sha\n't", " shall not", self.__text)
        self.__text = re.sub(r"o\'clock", " of the clock", self.__text)
        self.__text = re.sub(r"y\'all", " you all", self.__text)

        self.__text = re.sub(r"n\'t", " not", self.__text)
        self.__text = re.sub(r"n\'t've", " not have", self.__text)
        self.__text = re.sub(r"\'re", " are", self.__text)
        self.__text = re.sub(r"\'s", " is", self.__text)
        self.__text = re.sub(r"\'d", " would", self.__text)
        self.__text = re.sub(r"\'d've", " would have", self.__text)
        self.__text = re.sub(r"\'ll", " will", self.__text)
        self.__text = re.sub(r"\'ll've", " will have", self.__text)
        self.__text = re.sub(r"\'t", " not", self.__text)
        self.__text = re.sub(r"\'ve", " have", self.__text)
        self.__text = re.sub(r"\'m", " am", self.__text)
        self.__text = re.sub(r"\'re", " are", self.__text)

    def __filter_chars(self):
        sent = []
        for word in self.__text.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        self.__text =  ' '.join(sent)
    
    def __remove_mult_spaces(self):
        self.__text = re.sub("\s\s+" , " ", self.__text)
    
    def __lemmatize(self):
        lemmatize_words = []
        for words in self.__text:
            lemmatize_words.append(lem.lemmatize(words))
        self.__text = lemmatize_words

    def __output_data(self):
        self.__text = pd.DataFrame({"text":[self.__text]})
       
    def clean_data(self):
        self.__decontract()
        self.__remove_emoji()
        self.__remove_url()
        self.__clean_hashtags()
        self.__filter_chars()
        self.__remove_mult_spaces()
        self.__remove_stopwords()
        self.__lemmatize()
        self.__output_data()
        return self.__text['text']


class preprocessing_input_text_ltsm:
    def fit(self,text:str):
        self.text = text
        
    def __clean_hashtags(self):
        self.text = re.sub(r'\s*#\w+','',self.text)

    def __remove_emoji(self):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        self.text = emoji_pattern.sub(r'', self.text)
    
    def __remove_url(self):
        banned_list = string.punctuation
        self.text = re.sub(r"(?:\@|https?\://)\S+", " ", self.text)
        self.text = re.sub(r'[^\x00-\x7f]',r'', self.text)
        self.text = self.text.translate(str.maketrans(' ',' ',banned_list))
        return self.text
    
    def __remove_stopwords(self):
        clean_text = []
        for el in word_tokenize(self.text):
            if not el in stop_words:
                clean_text.append(el)
        self.text = clean_text

    def __decontract(self):
        self.text = self.text.replace('\r',' ').replace('\n',' ').lower()
        self.text = re.sub(r"won\'t", " will not", self.text)
        self.text = re.sub(r"won\'t've", " will not have", self.text)
        self.text = re.sub(r"can\'t", " can not", self.text)
        self.text = re.sub(r"don\'t", " do not", self.text)
        
        self.text = re.sub(r"can\'t've", " can not have", self.text)
        self.text = re.sub(r"ma\'am", " madam", self.text)
        self.text = re.sub(r"let\'s", " let us", self.text)
        self.text = re.sub(r"ain\'t", " am not", self.text)
        self.text = re.sub(r"shan\'t", " shall not", self.text)
        self.text = re.sub(r"sha\n't", " shall not", self.text)
        self.text = re.sub(r"o\'clock", " of the clock", self.text)
        self.text = re.sub(r"y\'all", " you all", self.text)

        self.text = re.sub(r"n\'t", " not", self.text)
        self.text = re.sub(r"n\'t've", " not have", self.text)
        self.text = re.sub(r"\'re", " are", self.text)
        self.text = re.sub(r"\'s", " is", self.text)
        self.text = re.sub(r"\'d", " would", self.text)
        self.text = re.sub(r"\'d've", " would have", self.text)
        self.text = re.sub(r"\'ll", " will", self.text)
        self.text = re.sub(r"\'ll've", " will have", self.text)
        self.text = re.sub(r"\'t", " not", self.text)
        self.text = re.sub(r"\'ve", " have", self.text)
        self.text = re.sub(r"\'m", " am", self.text)
        self.text = re.sub(r"\'re", " are", self.text)

    def __filter_chars(self):
        sent = []
        for word in self.text.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        self.text =  ' '.join(sent)
    
    def __remove_mult_spaces(self):
        self.text = re.sub("\s\s+" , " ", self.text)
    
    def __lemmatize(self):
        lemmatize_words = []
        for words in self.text:
            lemmatize_words.append(lem.lemmatize(words))
        self.text =  ' '.join([word for word in lemmatize_words])

    def clean_data(self):
        self.__remove_emoji()
        self.__decontract()
        self.__remove_url()
        self.__clean_hashtags()
        self.__filter_chars()
        self.__remove_mult_spaces()
        self.__remove_stopwords()
        self.__lemmatize()
        return self.text
