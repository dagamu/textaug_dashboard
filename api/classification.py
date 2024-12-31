from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import BinaryRelevance #, LabelPowerset, ClassifierChains  

import numpy as np

class TextRepresentationMethod:
    kind = "text_representation"
    def __init__(self, base, key):
        self.base = base
        self.name = type(base).__name__
        self.key = key
        
class ClassificationModel:
    kind = "classification_model"
    def __init__(self, base, key):
        self.base = base
        self.name = type(base).__name__
        self.key = key
        
class ProblemTransformation:
    kind = "problem_transformation"
    def __init__(self, base, key):
        self.base = base
        self.name = type(base).__name__
        self.key = key

class ClassificationManager:
    
    available_methods_labels = {
            "text_representation": {
                "TF": CountVectorizer,
                "TF-IDF": TfidfTransformer
                # TODO: TF-muRFL
            }, 
            "classification_model": {
                "MNB": MultinomialNB,
                "MLP NN": MLPClassifier
                # TODO: SVC
            }, 
            "problem_transformation": {
                "binary": BinaryRelevance
                # TODO: Label Powerset, CC
            }
    }
    available_methods = { "text_representation": {}, "classification_model": {}, "problem_transformation": {} }
    
    def __init__(self):
        self.selected_methods = { "text_representation": [], "classification_model": [], "problem_transformation": [] }
        self.get_available_methods()
        self.default_methods()
        
    def get_available_methods(self):
        for label, base in self.available_methods_labels["text_representation"].items():
            self.available_methods["text_representation"][label] = TextRepresentationMethod(base, label)
        for label, base in self.available_methods_labels["classification_model"].items():
            self.available_methods["classification_model"][label] = ClassificationModel(base, label)
        for label, base in self.available_methods_labels["problem_transformation"].items():
            self.available_methods["problem_transformation"][label] = ProblemTransformation(base, label)
        
    def default_methods(self):
        for kind, methods in self.available_methods.items():
            self.add_method(kind, list(methods.keys())[0] )
        
    def add_method(self, kind, key):
        if kind in self.available_methods.keys():
            if key in self.available_methods[kind].keys():
                new_method = self.available_methods[kind][key]
                self.selected_methods[kind].append(new_method)
                
    def del_method(self, target_method):
        for i, method in enumerate(self.selected_methods[target_method.kind]):
            if method.name == target_method.name:
                del self.selected_methods[target_method.kind][i]
                
    def train_model(self, vectorizer, multi_model, base_model, X, y ):
        
        base_vec = CountVectorizer()
        X = np.array(X)
        y_features = np.array(y) # preprocessing
            
        base_clf = base_model()
        vec = vectorizer()
        if not isinstance(vec, CountVectorizer):
            base_clf = Pipeline([('vectorizer', vec ),
                                 ('clf', base_clf) ])
        
        
        clf = Pipeline([("base_vec", base_vec ),
                        ("multi_clf", multi_model(base_clf) ) ])
        clf.fit(X, y_features)
            
        return clf