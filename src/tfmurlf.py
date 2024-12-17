import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TfmurlfTransformer(BaseEstimator, TransformerMixin):

  target = []

  def __init__(self, mu=np.average):
    self.mu = mu

  def calc_inv_vec(self, X, y, label_index):

    label_term = self.label_term_m
    anti_matrix = np.delete(label_term, label_index, axis=0)
    result = np.average(anti_matrix, axis=0)
    result = np.maximum(result, np.ones(result.shape) )

    return result

  def setup(self, X, y):
    self.samples_ = X
    self.target = y
    self.n_samples = X.shape[0]

    self.label_term_m = y.T @ X

    return self

  def transform(self, X):
    X_ = np.array(X, dtype=np.float32)
    label_index = self.current_label
    inv_vec = self.calc_inv_vec(self.samples_, self.target, label_index)

    for i, sample in enumerate(X_):
      factor = self.label_term_m[label_index, :] / inv_vec
      factor += np.ones( sample.shape[0] ) * 2
      factor = np.log2(factor)
      factor = np.squeeze(factor)

      sample = np.multiply(sample, factor, dtype=np.float32)
      X_[i, :] = sample

    return X_

  def fit(self, X, y):
    col_match = np.where(( np.squeeze(y)[:,None] == self.target ).all(axis=0))[0]
    if len(col_match) > 0:
      col_match = col_match[0]

    self.current_label = col_match
    return self