from math import log2
from numpy import average as mean
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class TfmurlfTransformer(BaseEstimator, TransformerMixin):
  
  target = []

  def __init__(self):
    self.labels = []
    self.vocabulary = set()

  def set_label_term_matrix(self, X, y):
    self.samples_ = X
    self.target = y
    self.label_term_matrix = y.T @ X
    return self

  def transform(self, X):
    X_ = X.copy()
    label = self.current_label

    for idx in np.ndindex(X_.shape):
      label_index, term_index = idx
      a_tl = self.label_term_matrix[label_index][term_index]
      X_[idx] *= log2( 2 + a_tl / max(1, a_tl) ) 

    return X_

  def fit(self, X, y):
    self.current_label = np.where(( self.target == y[:, None]).all(axis=0))[0]
    return self