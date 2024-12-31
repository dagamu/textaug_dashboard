from neattext.functions import clean_text
import numpy as np
import re
import ast

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing import MultiLabelBinarizer

class NeatTextCleanAction:

    name = "NeatText Clean"
    def apply_to(self, dataset):
        if dataset.X_format == "TEXT":
            dataset.X_features = clean_text(dataset.X_features)
            
class CustomFnAction:

    name = "Custom Text Processing"
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def data_processing(self, text):
        text= text.lower()
        text = re.sub('<br />', '', text)
        text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        
        filtered_text = [ self.stemmer.stem(w) for w in text_tokens if not w in self.stop_words]
        return " ".join(filtered_text)

    def apply_to(self, dataset):
        if dataset.X_format == "TEXT":
            for text in dataset.X_features:
                text = self.data_processing(text)
                
class FormatLabels:
    
    name = "Format Labels"
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=False)
    
    def apply_to(self, dataset):
        
        if dataset.y_format == "SEP":
            dataset.y_train = np.char.split(dataset.y_train, sep=dataset.format_sep)
            dataset.y_test = np.char.split(dataset.y_test, sep=dataset.format_sep)

        elif dataset.y_format == "LITERAL":
            dataset.y_train = [ast.literal_eval(v[0]) for v in dataset.y_train] 
            dataset.y_test = [ast.literal_eval(v[0]) for v in dataset.y_test] 
            
        if dataset.y_format != "BINARY":
            dataset.y_train = self.mlb.fit_transform(dataset.y_train)
            dataset.y_test = self.mlb.fit_transform(dataset.y_test)
            dataset.labels = self.mlb.classes_
        

class PreprocessingManager():
    def __init__(self):
        self.actions = [] #FormatLabels()
        
    def apply_to(self, dataset):
        for action in self.actions:
            action.apply_to(dataset)
        dataset.preprocessed = True