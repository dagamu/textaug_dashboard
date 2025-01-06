import numpy as np

def uniform_crossover( parentA, parentB ):
    mask = np.random.randint(0, 2, parentA.shape[0] )
    new_genA =  parentA * mask + parentB * (1-mask)
    new_genB =  parentB * mask + parentA * (1-mask)
    return [new_genA, new_genB]

class GeneticSampler:
  
  def __init__(self, pob_size, crossover=uniform_crossover):
    self.pob = []
    self.pob_size = pob_size
    self.crossover = crossover

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

    return self.pob[0].astype(bool)              