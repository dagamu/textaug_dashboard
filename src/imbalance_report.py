import numpy as np

def create_imbalance_report(y_features):

    q = y_features.shape[1]
    label_count = np.sum(y_features, axis=1)
    card = np.mean(label_count)
    density = card / y_features.shape[1]

    freq = np.sum(y_features, axis=0)
    max_count = np.argmax(freq)
    min_count = np.argmin(freq)
    
    IRLbl = np.max(freq) / freq
    mean_ir = np.mean(IRLbl)
    max_ir = np.max(IRLbl)
    
    IRLbl_sigma = np.sqrt( np.sum( (IRLbl- mean_ir)**2 / (q-1) ) )
    cv_ir = IRLbl_sigma / mean_ir

    IRs = y_features * IRLbl
    IRmeans = np.sum(IRs, axis=1) / np.where(label_count == 0, 1, label_count)
    IRprod = np.prod(np.where(IRs == 0, 1, IRs), axis=1)
    scumble_ins = np.where(IRmeans == 0, 0, 1 - (IRprod ** (1 / label_count)) / IRmeans)
    scumble = np.mean(scumble_ins)
    
    return {
                "instances": y_features.shape[0],
                "n_labels": q,
                "card": card,
                "density": density,
                "labels_freq": freq,
                "max_count": max_count,
                "min_count": min_count,
                "irl_bl": IRLbl,
                "mean_ir": mean_ir,
                "max_ir": max_ir,
                "cv_ir": cv_ir,
                "scumble_ins": scumble_ins,
                "scumble": scumble
            }