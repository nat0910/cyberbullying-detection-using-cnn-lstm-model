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


class df_prepocessing_cnn:
    def __init__(self,text):
        text = text.to_frame()
        self.text = text.iloc[:,0]
    
    def remove_emoji(self,text):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_url(self,text):
        text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
        text = re.sub(r'[^\x00-\x7f]',r'', text)
        banned_list = string.punctuation
        text = text.translate(str.maketrans(' ',' ',banned_list))
        return text

    def remove_mult_spaces(self,text):
        return re.sub("\s\s+" , " ", text)
    
    def remove_stopwords(self,text):
        clean_text = []
        for el in word_tokenize(text):
            if not el in stop_words:
                clean_text.append(el)
        return clean_text

    def decontract(self,text):
        text = text.replace('\r',' ').replace('\n',' ').lower()
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text

    def filter_chars(self,text):
        sent = []
        for word in text.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)
    
    def clean_hashtags(self,text):
        text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
        text = " ".join(word.strip() for word in re.split('#|_', text)) #remove hashtags symbol from words in the middle of the sentence
        return text

    def lemmatize(self,text):
        lemmatize_words = []
        for words in text:
            lemmatize_words.append(lem.lemmatize(words))
            # lemmatize_words.append(ps.stem(words))
        return lemmatize_words
    
    def clean_data(self):
       self.text = self.text.apply(self.decontract)
       self.text = self.text.apply(self.remove_emoji)
       self.text = self.text.apply(self.remove_url)
       self.text = self.text.apply(self.clean_hashtags)
       self.text = self.text.apply(self.filter_chars)
       self.text = self.text.apply(self.remove_mult_spaces)
       self.text = self.text.apply(self.remove_stopwords)
       self.text = self.text.apply(self.lemmatize)
       return self.text
    

class df_prepocessing_lstm:
    def __init__(self,text):
        text = text.to_frame()
        self.text = text.iloc[:,0]
    
    def remove_emoji(self,text):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def remove_url(self,text):
        text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
        text = re.sub(r'[^\x00-\x7f]',r'', text)
        banned_list = string.punctuation
        text = text.translate(str.maketrans(' ',' ',banned_list))
        return text

    def remove_mult_spaces(self,text):
        return re.sub("\s\s+" , " ", text)
    
    def remove_stopwords(self,text):
        clean_text = []
        for el in word_tokenize(text):
            if not el in stop_words:
                clean_text.append(el)
        return clean_text

    def decontract(self,text):
        text = text.replace('\r',' ').replace('\n',' ').lower()
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)
        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return text

    def filter_chars(self,text):
        sent = []
        for word in text.split(' '):
            if ('$' in word) | ('&' in word):
                sent.append('')
            else:
                sent.append(word)
        return ' '.join(sent)
    
    def clean_hashtags(self,text):
        text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text)) #remove last hashtags
        text = " ".join(word.strip() for word in re.split('#|_', text)) #remove hashtags symbol from words in the middle of the sentence
        return text

    def lemmatize(self,text):
        lemmatize_words = []
        for words in text:
            lemmatize_words.append(lem.lemmatize(words))
            # lemmatize_words.append(ps.stem(words))
        return ' '.join([word for word in lemmatize_words])
    
    def clean_data(self):
       self.text = self.text.apply(self.decontract)
       self.text = self.text.apply(self.remove_emoji)
       self.text = self.text.apply(self.remove_url)
       self.text = self.text.apply(self.clean_hashtags)
       self.text = self.text.apply(self.filter_chars)
       self.text = self.text.apply(self.remove_mult_spaces)
       self.text = self.text.apply(self.remove_stopwords)
       self.text = self.text.apply(self.lemmatize)
       return self.text
    