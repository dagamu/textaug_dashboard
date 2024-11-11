import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score # TODO: Add more performance metrics

def lwise_accuracy(y, y_pred):
    acc = y == y_pred
    acc = np.mean(acc, axis=1)
    return np.mean(acc)

def get_performance( clf, preprocessing, X, y, prefix="", round_=-1, percentage=False ):
    y_features = preprocessing.transform(y)
    y_pred = clf.predict(X)
    
    result = {
        f"{prefix+'_'}exact_acc":    accuracy_score(y_features, y_pred),
        f"{prefix+'_'}acc":          lwise_accuracy(y_features , y_pred ),
        f"{prefix+'_'}hl":           hamming_loss(y_features , y_pred ),
        f"{prefix+'_'}precision":    precision_score(y_features, y_pred, zero_division=0 ,average="samples"),
        f"{prefix+'_'}recall":       recall_score(y_features, y_pred, average="samples"),
        f"{prefix+'_'}f1":           f1_score(y_features, y_pred, average="samples")
    }
    
    if percentage:
        result = { key: value * 100 for key, value in result.items() }
        
    if round_ != -1:
        result = { key: round(value, round_) for key, value in result.items() }
        
    return result