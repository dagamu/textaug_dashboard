import streamlit as st
from menu import menu

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from skmultilearn.problem_transform import BinaryRelevance#, LabelPowerset, ClassifierChains  

VECTORIZER_METHODS = {
    "Term Frequency (Count Vectorizer)": CountVectorizer,
    "TF-IDF": TfidfVectorizer,
}

PROBLEM_TRANSFORM_METHODS = {
    "Binary Encoding": { "model": BinaryRelevance, "preprocessing": MultiLabelBinarizer},
    #"Label Powerset": LabelPowerset,
    #"ClassifierChains": ClassifierChains,
}

AVAIBLE_MODELS = {
    "Multinomial Naive Bayes": MultinomialNB,
    #"SVM Classifier": SVC,
    "MLP NN": MLPClassifier,
}

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
    model_select = st.selectbox("Select Model to Train", AVAIBLE_MODELS.keys() )
   
    if pt_method and vec_method and model_select: 
        if st.button("Train", type="primary"):
            
            multi_model, preprocessing = PROBLEM_TRANSFORM_METHODS[ str(pt_method) ].values()
            
            vectorizer = VECTORIZER_METHODS[ str(vec_method) ]()
            X_features = vectorizer.fit_transform( df[samples_column] )
            
            preprocessing = preprocessing()
            y_features = preprocessing.fit_transform( df[labels_column] )
            
            X_train, X_test, y_train, y_test = train_test_split( X_features, y_features, test_size=0.3, random_state=42 )
            
            base_model = AVAIBLE_MODELS[ str(model_select) ]()
            clf = multi_model(base_model)
            clf.fit(X_train, y_train)
        
            train_acc   = accuracy_score(   y_train, clf.predict(X_train) ) * 100
            test_acc    = accuracy_score(   y_test , clf.predict(X_test ) ) * 100
            train_hl    = hamming_loss(     y_train, clf.predict(X_train) )
            test_hl     = hamming_loss(     y_test , clf.predict(X_test ) )
           
            st.info(f"Train Split Performance: Accuracy {train_acc:.2f}%, Hamming Loss {train_hl:.3f}", icon="ℹ")
            st.info(f"Test Split Performance: Accuracy {test_acc:.2f}%, Hamming Loss {test_hl:.3f}", icon="ℹ")
    
if __name__ == "__main__":         
    ClasificationModelPage()
    menu()
