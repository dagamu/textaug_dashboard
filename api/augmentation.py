import pandas as pd
import numpy as np

from src.eda_nlpaug import EDAug
from src.char_aug import CharAugmenter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from src.tfmurlf import TfmurlfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

class AugSelectionMethod:
    def get_aug_input(self, X_features, y_features, size):
        data_size = len(X_features)
        mask = np.zeros(data_size)
        idx = np.random.choice( len(mask), size, replace= size > data_size)
        mask[idx] = 1
        return mask.astype(bool)
    
class RandomSelection(AugSelectionMethod):
    name = "Random Selection"
    key = "random"

class AugmentationManager:
    
    available_methods = {
        "eda_aug": EDAug,
        "char_aug": CharAugmenter,
    }
    
    def __init__(self):
        self.aug_selection_methods = [ RandomSelection ]
        self.items = []
        self.steps = []
        self.step_kind = None
        
    def add_method(self, key, params):
        new_method = self.available_methods[key](**params)
        self.items.append(new_method)
        
    def remove(self, item):
        self.items.remove(item)
        
    def set_steps(self, steps):
        self.steps = steps
        
    def AugmentDataset(self, dataset, n_samples, aug_method, aug_choice):

        if n_samples == 0:
            return 
        
        X_features = dataset.X_train
        y_features = dataset.y_train
        
        input_mask = aug_choice.get_aug_input(X_features, y_features, size=n_samples)
        
        text_input = np.array(X_features)[input_mask].tolist()
        labels_input = np.array(y_features)[input_mask].tolist()
        
        dataset.X_aug += aug_method.augment(text_input)
        dataset.y_aug += labels_input
            
    
def calcSelectionWeights( text_col, labels_col, selection_method ):
    
    weights = np.ones( len(text_col) )
    if selection_method == "Random":
        weights = np.ones( len(text_col) )
    
    base_vec = CountVectorizer().fit_transform(text_col).toarray()
    
    if selection_method == "TF-IDF":
        vectorized = TfidfTransformer().fit_transform(base_vec)
        weights = np.sum( vectorized.toarray(), axis=1)

    if selection_method == "TF-muRFL":
        
        y_features = MultiLabelBinarizer().fit_transform(labels_col)
        labels_freq = np.sum(y_features, axis=0)
        irl_bl = np.max(labels_freq) / labels_freq 
        
        weights = np.zeros( len(text_col) )
        TF_muRLF_transformer = TfmurlfTransformer()
        TF_muRLF_transformer.setup(base_vec, y_features)
        
        weights = np.zeros( len(text_col) )
        for label_index, label_imbalance_ratio in enumerate(irl_bl):
            vectorized = TF_muRLF_transformer.fit_transform(base_vec, y_features[:, label_index] )
            weights += np.sum(vectorized, axis=1) * label_imbalance_ratio
    
    return np.squeeze( np.squeeze(weights) / np.sum(weights) )