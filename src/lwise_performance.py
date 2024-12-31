import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score

def lwise_accuracy(y, y_pred):
    acc = y == y_pred
    acc = np.mean(acc, axis=1)
    return np.mean(acc)

def get_performance( clf, X, y, prefix="", round_=-1, percentage=False ):
    y_features = y
    y_pred = clf.predict(X)
    
    prefix = prefix + '_' if prefix != "" else ""
    
    result = {
        f"{prefix}exact_acc":    accuracy_score(y_features, y_pred),
        f"{prefix}acc":          lwise_accuracy(y_features , y_pred ),
        f"{prefix}hl":           hamming_loss(y_features , y_pred),
        f"{prefix}precision":    precision_score(y_features, y_pred, zero_division=0 ,average="samples"),
        f"{prefix}recall":       recall_score(y_features, y_pred, zero_division=0, average="samples"),
        f"{prefix}f1":           f1_score(y_features, y_pred, zero_division=0, average="samples")
    }
    
    if percentage:
        result = { key: value * 100 for key, value in result.items() }
        
    if round_ != -1:
        result = { key: round(value, round_) for key, value in result.items() }
        
    return result