import numpy as np
import streamlit as st

from utils import custom_multiselect

def uniform_crossover( parentA, parentB ):
    mask = np.random.randint(0, 2, parentA.shape[0] )
    new_genA =  parentA * mask + parentB * (1-mask)
    new_genB =  parentB * mask + parentA * (1-mask)
    return [new_genA, new_genB]

class GeneticSampler:
  
  def __init__(self, pob_size, crossover):
    self.pob = []
    self.pob_size = pob_size
    self.crossover = crossover
    self.assert_time = 0

  def initialize_population(self, n_samples, target_actives=-1 ):

    n_actives = [target_actives] * self.pob_size
    if target_actives == -1:
        n_actives = np.random.binomial( n=n_samples, p=0.1, size=self.pob_size )
        
    self.pob = np.zeros( (self.pob_size, n_samples) )
    for i in range(self.pob_size):
        replace = n_samples < n_actives[i]
        ones_indices = np.random.choice( n_samples, n_actives[i], replace=replace )
        self.pob[i, ones_indices] = 1

  def assert_masks(self, y_):
    zero_cases = np.where( (self.pob @ y_) == 0 )
    zero_cases = ( zero_cases[0], self.pivot_samples[zero_cases[1]] )
    self.pob[zero_cases] = 1

  def calc_pivot_samples(self, y_):
    sorted_indices = np.argsort( np.sum(y_, axis=1) )
    self.pivot_samples = np.zeros(y_.shape[1], dtype=np.uint64)
    for i in range(y_.shape[1]):
        self.pivot_samples[i] = sorted_indices[ np.argmax( y_[sorted_indices, i] == 1 ) ]

  def mutate(self, new_genes, mutation_prob):
      for gen in new_genes:
          if np.random.rand() < mutation_prob:
              mutation_index = np.random.randint(0, gen.shape[0])
              gen[mutation_index] = np.random.randint(0, 2)
      return new_genes

  def update_pob(self, mutation_prob):
      newGen = self.pob_size // 2 + 1
      while newGen < self.pob_size - 1:
            parentA, parentB = np.random.choice( range(self.pob_size // 2), 2 )
            new_genes = self.crossover(self.pob[parentA], self.pob[parentB] )
            new_genes = self.mutate(new_genes, mutation_prob)
            self.pob[ newGen:newGen + len(new_genes) ] = new_genes
            newGen += len(new_genes)
    
  def sample(self, y_, loss, max_iterations=50, target_actives=-1, keep_labels=True, mutation_prob=0.01, verbose=0):
    # Y: MultiLabelBinarizer Product(Samples x Labels)
    y_ = np.array(y_, dtype=np.uint8 )
    n_samples = y_.shape[0]
      
    self.initialize_population(n_samples, target_actives)

    if keep_labels:
        self.calc_pivot_samples(y_)
      
    for i in range(max_iterations):

      if keep_labels:
          self.assert_masks(y_)
      pob_loss = loss(self.pob)

      sorted_indices = np.argsort(pob_loss)
      self.pob = self.pob[sorted_indices]

      if np.any(pob_loss == 0):
        break

      self.update_pob(mutation_prob)

      if verbose:
        if i % verbose == 0:
          print(f"{i} - Min Loss: {np.min(pob_loss):.4f}, Mean Loss: {np.mean(pob_loss):.4f}")   

    return self.pob[0]

class GeneticSamplerSetup:
  
    def range_cases(self, label, key):
      with st.container():
              col1, col2 = st.columns([3,1])
              
              min_value, max_value = col1.slider(f"{label} ({key})", min_value=0, max_value=100, value=(0,100))
              n_cases = col2.number_input("N° Cases", min_value=1, key=key)
              
              range_ = [ "Free" ]
              if n_cases > 1 and min_value != max_value:
                step = (max_value-min_value)/(n_cases-1)
                range_ += np.arange( min_value, max_value+1, step=step ).tolist()
              else:
                mean_value = (max_value+min_value)/2
                range_ +=  [mean_value]
                
              formated_list =  ["Free"] + [ f"{ round(i, 2)}%" for i in range_[1:] ]
              col1.text( 'Cases: ' + ', '.join(formated_list) )
              return range_
            
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
      
      metrics = {
        "n_samples": np.sum(pob, axis=1),
        "mean_ir": np.mean(irl_bl, axis=1),
        "max_ir":  np.max(irl_bl, axis=1),
      }
      
      loss_val = 0
      for target, value in self.target_values.items():
        if target in metrics.keys():
          loss_val += ( value - metrics[target] ) ** 2
      
      return loss_val
  
    def set_loss(self, target_values):      
      self.target_values = target_values
  
    def render(self):
        self.active = st.checkbox("Generate Dataset Sampling", value=False)
        if not self.active:
          return
        
        with st.container(border=True):
            st.markdown("**Genetic Sampler**")
            
            self.keep_labels = st.checkbox("Keep N° of Labels")
            self.pob_size = st.number_input("Population Size", min_value=5, value=20 )
            self.max_iterations = st.number_input("Max Iterations", min_value=5, value=10 )
            self.n_samples_list = custom_multiselect("Number of Samples", ["Free"], ["Free"], 100, "genetic_nsamples")
            
            self.MeanIrRange = self.range_cases("Mean Imbalance Ratio ", "MeanIR")
            self.MaxIrRange = self.range_cases("Max Imbalance Ratio", "MaxIR")
        
    def get_sample(self, y_ ,sampler, n_samples):
      return sampler.sample(y_, self.loss, self.max_iterations, target_actives=n_samples, keep_labels=self.keep_labels).astype(bool)
              
    def generate_masks(self, y_):
        
        masks = []
        self.y_ = y_
        freq = np.sum(y_, axis=0)
        sampler = GeneticSampler(pob_size=self.pob_size, crossover=uniform_crossover)
        
        for n_samples in self.n_samples_list:
          
          if n_samples == "Free":
            n_samples = -1
            
          self.set_loss( target_values={ "max_ir": np.max(freq) } )
          mask = self.get_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          max_ir_found = np.max(irlbl)
          
          self.set_loss( target_values={ "max_ir": 0 } )
          mask = self.get_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          min_ir_found = np.max(irlbl)
          
          self.set_loss( target_values={ "mean_ir": np.max(freq) } )
          mask = self.get_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          max_mean_ir_found = np.mean(irlbl)
          
          self.set_loss( target_values={ "mean_ir": 0 } )
          mask = self.get_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          min_mean_ir_found = np.mean(irlbl)
          
          for mean_ir_target in self.MeanIrRange:
            for max_ir_target in self.MaxIrRange:
              
              target_values = { }
              if n_samples != "Free":
                n_samples = n_samples if n_samples < y_.shape[0] else y_.shape[0]
                target_values["n_samples"] = n_samples
                
              if mean_ir_target != "Free":
                target_values["mean_ir"] = (max_mean_ir_found - min_mean_ir_found) *  mean_ir_target / 100 + min_mean_ir_found
                
              if max_ir_target != "Free":
                target_values["max_ir"] = (max_ir_found - min_ir_found) *  max_ir_target / 100 + min_ir_found
              
              self.set_loss(target_values)
              masks.append( self.get_sample(y_, sampler, n_samples) )
              
        return masks
              