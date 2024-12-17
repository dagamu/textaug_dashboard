import configparser
import json
import os

from api.session import Session


def read_default_session():
    
    session = Session()
    
    session_config = configparser.ConfigParser()
    session_config.read('presets/default.config')  

    datasets_raw = session_config.items("DATASETS")[0][1]
    datasets_list = json.loads(datasets_raw)
    for item in datasets_list:
        session.datasets.load_dataset(item["source"], item["params"])
        
    aug_methods_raw = session_config.items("DATA_AUGMENTATION")[0][1]
    aug_methods_list = json.loads(aug_methods_raw)
    for kind, params in aug_methods_list:
        session.aug_manager.add_method(kind, params)
        
    aug_steps = json.loads(session_config["DATA_AUGMENTATION"]["steps"])
    session.aug_manager.set_steps(aug_steps)

    return session