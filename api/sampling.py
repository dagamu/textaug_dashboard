import numpy as np

class FullDatasetSampling:
    key = "full_dataset"
    name = "Full Dataset"
    unique = True
    n_iterations = 1
    
    def get_samples(self, dataset):
        return np.ones( (1, len(dataset.X_train) ) )

class RandomSampling:
    key = "random_sampling"
    name = "Random Sampling"
    unique = False
    
    def __init__(self, sampling_list):
        self.sampling_list = sampling_list
        self.n_iterations = len(sampling_list)
        
    def get_samples(self, dataset):
        
        shape = (len(self.sampling_list), len(dataset.X_train) )
        samples = np.zeros(shape=shape)
        kind = "%" if self.sampling_list[0] < 1 else "N"
        
        for i, value in enumerate(self.sampling_list):
            n_actives = round(shape[1] * value) if kind == "%" else value
            actives_ind = np.random.choice(shape[1], n_actives, replace=False)
            samples[i, actives_ind] = 1
            
        return samples

class SamplingManager:
    
    items = [ RandomSampling([0.1]) ]
    available_methods = {
        "Full Dataset": FullDatasetSampling,
        "Random Sample": RandomSampling,
    }
    
    def add_method(self, method, params):
        if method.unique:
            for item in self.items:
                if item.key == method.key:
                    return
        self.items.append(method(**params))
        
    def remove(self, item):
        self.items.remove(item)