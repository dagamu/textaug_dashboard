import nlpaug.augmenter.word as naw
from numpy.random import choice
from nltk.corpus import wordnet
from itertools import chain
from random import randint

INITIAL_ADDITIONAL_PARAMS = {
    "SR": { "aug_min": 1, "aug_max": 10, "aug_p": 0.3 },    # Synonym Replacement
    "RI": {},                                               # Random Insertion
    "RS": { "aug_min": 1, "aug_max": 10, "aug_p": 0.3 },    # Random Swap
    "RD": { "aug_min": 1, "aug_max": 10, "aug_p": 0.3 },    # Random Deletion
}

class RandomSynonymInsertion:    
    def augment(self, text):
        words = text.split(" ")
        
        lemmas = []
        i = 0 
        while not lemmas and i < 50:
            selected_word = choice(words, 1)[0]
            synonyms = wordnet.synsets(selected_word)
            lemmas = list(set(chain.from_iterable([w.lemma_names() for w in synonyms])))
            i += 1
     
        if len(lemmas):
            selected_synonym = choice(lemmas, 1)[0]
        else: 
            selected_synonym = selected_word
            
        words.insert( randint(0,len(words)), selected_synonym )
        return [" ".join(words)]
    
class EDAug:
    
    # TODO: Return or save methods of augmentation to show in table
    
    def __init__(self, p_weights = [1,1,1,1], lexical_src='wordnet', lang="eng", stopwords=None, additional_params={}, name="Easy Data Augmentation" ):
        
        additional_params = {**INITIAL_ADDITIONAL_PARAMS, **additional_params}
        sr_add_params = additional_params["SR"]
        ri_add_params = additional_params["RI"]
        rs_add_params = additional_params["RS"]
        rd_add_params = additional_params["RD"]
        
        self.name = name
        
        self.p_weights = p_weights
        
        self.sr_aug = naw.SynonymAug(aug_src=lexical_src, lang=lang, stopwords=stopwords, **sr_add_params )
        self.ri_aug = RandomSynonymInsertion(**ri_add_params)
        self.rs_aug = naw.RandomWordAug(action="swap", stopwords=stopwords, **rs_add_params )
        self.rd_aug = naw.RandomWordAug(action="delete", stopwords=stopwords, **rd_add_params )
        
        self.methods = {"SR": self.sr_aug, "RI": self.ri_aug, "RS": self.rs_aug, "RD": self.rd_aug }
    
    def augment_text(self, text):
        p_dist = [ w/sum(self.p_weights) for w in self.p_weights ]
        method = choice( list(self.methods.keys()), 1, p=p_dist)[0]
        augmenter = self.methods[method]
        return augmenter.augment(text)[0]
    
    def augment(self, data, n = 1):
        if type(data) == list:
            n = 1
            return [ self.augment_text(text) for text in data ]
        if type(data) == str:
            return [ self.augment_text(data) for _ in range(n) ]
        
        else:
            raise TypeError("Only str and list<str> types are supported for augmentation")