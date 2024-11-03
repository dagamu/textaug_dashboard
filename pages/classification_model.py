import streamlit as st
from menu import menu

import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, hamming_loss, f1_score # TODO: Add more performance metrics
from skmultilearn.problem_transform import BinaryRelevance #, LabelPowerset, ClassifierChains  

VECTORIZER_METHODS = {
    "Term Frequency (Count Vectorizer)": CountVectorizer,
    "TF-IDF": TfidfTransformer,
    #"TF-muRLF": TfmurlfVectorizer
}

PROBLEM_TRANSFORM_METHODS = {
    "Binary Encoding": { "model": BinaryRelevance, "preprocessing": MultiLabelBinarizer},
    #"Label Powerset": LabelPowerset,
    #"ClassifierChains": ClassifierChains,
}

AVAIBLE_MODELS = {
    "Multinomial Naive Bayes": (MultinomialNB, {}),
    "MLP NN": (MLPClassifier, { "max_iter": 500, "verbose": False }),
    #"SVM Classifier": SVC,
}

def lwise_accuracy(y, y_pred):
    acc = y == y_pred
    acc = np.mean(acc, axis=1)
    return np.mean(acc)

def get_performance( clf, preprocessing, X, y ):
    y_features = preprocessing.transform(y)
    return {
        "acc": lwise_accuracy(y_features , clf.predict(X) ),
        "hl": hamming_loss(y_features , clf.predict(X) ),
        "exact_acc": accuracy_score(y_features, clf.predict(X))
    }

def train_model(vectorizer, multi_model, preprocessing, base_model, X, y ):
        
    base_vec = CountVectorizer()
    y_features = preprocessing.transform(y)
        
    base_clf = base_model
    if not isinstance(vectorizer, CountVectorizer):
        base_clf = Pipeline([   ('vectorizer', vectorizer ),
                                ('clf', base_model) ])
    
    
    clf = Pipeline([    ("base_vec", base_vec ),
                        ("multi_clf", multi_model(base_clf) ) ])
    clf.fit(X, y_features)
        
    return clf

def ClasificationModelPage():
    if not "df" in st.session_state:
        st.warning('There is no dataset :(') 
        return 
            
    df = st.session_state["df"]
    labels_column = st.session_state["labels_column"]
    
    st.title("Classification Model")
    
    samples_column = st.selectbox("Samples Column", df.columns )
    pt_method = st.selectbox("Problem Transformation Method", PROBLEM_TRANSFORM_METHODS.keys() )
    vec_method = st.selectbox("Vectorizer Method", VECTORIZER_METHODS.keys() )
    selected_model = st.selectbox("Select Model to Train", AVAIBLE_MODELS.keys() )
   
    if pt_method and vec_method and selected_model: 
        if st.button("Train", type="primary"):
            
            X_features = df[samples_column].values
            y_features =  df[labels_column].values
            
            multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[ str(pt_method) ].values()
            preprocessing = preprocessing()
            preprocessing.fit(y_features)
            
            X_train, X_test, y_train, y_test = train_test_split( X_features, y_features, test_size=0.3, random_state=42 )
            
            base_model, model_params = AVAIBLE_MODELS[ str(selected_model) ]
            base_model = base_model(**model_params)
            
            vectorizer = VECTORIZER_METHODS[ str(vec_method) ]()
            
            clf = train_model(vectorizer, multi_model, preprocessing, base_model, X_train, y_train )
            
            train_per = get_performance( clf, preprocessing, X_train, y_train )
            test_per = get_performance( clf, preprocessing, X_test, y_test )
            
            st.info(f"Train Split Performance: Accuracy {train_per['acc']*100:.2f}%, Hamming Loss {train_per['hl']:.3f}", icon="ℹ")
            st.info(f"Test Split Performance: Accuracy {test_per['acc']*100:.2f}%, Hamming Loss {train_per['hl']:.3f}", icon="ℹ")
    
if __name__ == "__main__":         
    ClasificationModelPage()
    menu()
