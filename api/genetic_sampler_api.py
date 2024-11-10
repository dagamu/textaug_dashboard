import streamlit as st
import numpy as np

from src.genetic_sampler import GeneticSampler, uniform_crossover

from utils import custom_multiselect

class GeneticSamplerAPI:
  
    name = "Genetic Sampling"
  
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
      st.markdown("**Genetic Sampler**")
      
      self.keep_labels = st.checkbox("Keep N° of Labels")
      self.pob_size = st.number_input("Population Size", min_value=5, value=20 )
      self.max_iterations = st.number_input("Max Iterations", min_value=5, value=10 )
      self.n_samples_list = custom_multiselect("Number of Samples", ["Free"], ["Free"], 100, "genetic_nsamples")
      
      self.MeanIrRange = self.range_cases("Mean Imbalance Ratio ", "MeanIR")
      self.MaxIrRange = self.range_cases("Max Imbalance Ratio", "MaxIR")
      
      self.n_iterations = len(self.MeanIrRange) * len(self.MaxIrRange) * len(self.n_samples_list)
        
    def generate_sample(self, y_ , sampler, n_samples):
      return sampler.sample(y_, self.loss, self.max_iterations, target_actives=n_samples, keep_labels=self.keep_labels).astype(bool)
              
    def get_sample(self, y_):
        
        masks = []
        self.y_ = y_
        freq = np.sum(y_, axis=0)
        sampler = GeneticSampler(pob_size=self.pob_size, crossover=uniform_crossover)
        
        for n_samples in self.n_samples_list:
          
          if n_samples == "Free":
            n_samples = -1
            
          self.set_loss( target_values={ "max_ir": np.max(freq) } )
          mask = self.generate_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          max_ir_found = np.max(irlbl)
          
          self.set_loss( target_values={ "max_ir": 0 } )
          mask = self.generate_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          min_ir_found = np.max(irlbl)
          
          self.set_loss( target_values={ "mean_ir": np.max(freq) } )
          mask = self.generate_sample(y_, sampler, n_samples)
          irlbl = self.calc_irlbl(mask)
          max_mean_ir_found = np.mean(irlbl)
          
          self.set_loss( target_values={ "mean_ir": 0 } )
          mask = self.generate_sample(y_, sampler, n_samples)
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
              masks.append( self.generate_sample(y_, sampler, n_samples) )
              
        return masks