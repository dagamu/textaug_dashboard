import numpy as np
from src.genetic_sampler import GeneticSampler

class FullDatasetSampling:
    key = "full_dataset"
    name = "Full Dataset"
    unique = True
    n_iterations = 1
    
    def get_samples(self, dataset):
        return ["100%"], np.ones( (1, len(dataset.X_train) ) )

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
        target_sizes = []
        kind = "%" if self.sampling_list[0] < 1 else "N"
        
        for i, value in enumerate(self.sampling_list):
            n_actives = round(shape[1] * value) if kind == "%" else value
            actives_ind = np.random.choice(shape[1], n_actives, replace=False)
            samples[i, actives_ind] = 1
            target_sizes.append(f"{value*100:.2f}%" if kind == "%" else f"{value}")
            
        return samples
    
class GeneticSamplerAPI:
    key = "genetic_sampler"
    name = "Genetic Sampler"
    unique = True
    n_iterations = 1
    
    def __init__(self, keep_labels=True, pob_size=20, max_iterations=20, n_samples_list=[0.1], MeanIrRange=[-1, 0, 1], MaxIrRange=[-1, 0, 1]):
        self.keep_labels = keep_labels
        self.pob_size = pob_size
        self.max_iterations = max_iterations
        self.n_samples_list = n_samples_list
        self.MeanIrRange = sorted(MeanIrRange, reverse=True)
        self.MaxIrRange = sorted(MaxIrRange, reverse=True)
        self.n_iterations = len(self.MeanIrRange) * len(self.MaxIrRange) * len(self.n_samples_list)
    
    def calc_irlbl(self, pob):
      freq = pob @ self.y_
      zero_mask = freq == 0
      freq = np.ma.masked_array( freq, zero_mask, fill_value=0.01 )
      
      multi_args = {} if pob.ndim == 1 else { "axis": 1, "keepdims": True }
      max_freq = np.max(freq, **multi_args )
      irl_bl = max_freq / freq
      return irl_bl
  
    def loss(self, pob):
      irl_bl = self.calc_irlbl(pob)
      metrics = { "n_samples": np.sum(pob, axis=1), "mean_ir": np.mean(irl_bl, axis=1), "max_ir":  np.max(irl_bl, axis=1) }
      
      loss_val = 0
      for target, value in self.target_values.items():
        if target in metrics.keys():
          loss_val += ( value - metrics[target] ) ** 2
      
      return loss_val
  
    def set_loss(self, target_values):      
      self.target_values = target_values
        
    def generate_sample(self, y_ , sampler, n_samples):
      return sampler.sample(y_, self.loss, self.max_iterations, target_actives=n_samples, keep_labels=self.keep_labels).astype(bool)
              
    def get_samples(self, dataset):
        
        masks = []
        target_sizes = []
        y_ = dataset.y_train
        self.y_ = y_
        freq = np.sum(y_, axis=0)
        sampler = GeneticSampler(pob_size=self.pob_size)

        mean_scope_list = [ freq.max() / freq.min() / 100 ]
        max_scope_list = [ y_.shape[0] / 100 ]
        
        n_samples_list = [ int(x * y_.shape[0]) for x in self.n_samples_list] 
        for i, n_samples in enumerate(n_samples_list):
          
          if n_samples < 0:
            n_samples = -1
            
          for mean_ir_target in self.MeanIrRange:
            for max_ir_target in self.MaxIrRange:
              
              target_values = { }
              if n_samples != -1:
                n_samples = min(n_samples, y_.shape[0])
                target_values["n_samples"] = n_samples
                
              if mean_ir_target != -1:
                target_values["mean_ir"] = np.mean(mean_scope_list) * mean_ir_target
                
              if max_ir_target != -1:
                target_values["max_ir"] = np.mean(max_scope_list) * max_ir_target
              
              self.set_loss(target_values)
              sample = self.generate_sample(y_, sampler, n_samples)
              irlbl = self.calc_irlbl(sample)
              masks.append(sample)
              
              if mean_ir_target != "Free":
                mean_scope_list.append( np.mean(irlbl) / max(mean_ir_target, 0.1) )
                
              if max_ir_target != "Free":
                max_scope_list.append( np.max(irlbl) / max(max_ir_target, 0.1) )
                
              target_sizes.append(f"{self.n_samples_list[i] * 100}%")
                
        return target_sizes, masks
    

class SamplingManager:
    
    items = [ GeneticSamplerAPI(n_samples_list=[0.1, 0.2, 0.5]) ]
    available_methods = {
        "Full Dataset": FullDatasetSampling,
        "Random Sample": RandomSampling,
        "Genetic Algorithm": GeneticSamplerAPI
    }
    
    def add_method(self, method, params):
        if method.unique:
            for item in self.items:
                if item.key == method.key:
                    return
        self.items.append(method(**params))
        
    def remove(self, item):
        self.items.remove(item)