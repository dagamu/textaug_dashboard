import nlpaug.augmenter.char as nac

class CharAugmenter:
    
    name = "Character Augmenter"
    key = "char_aug"
    
    available_methods = {
        "OCR": nac.OcrAug,
        "keyboard": nac.KeyboardAug,
        "random": nac.RandomCharAug,
    }
    
    def __init__(self, kind, params):
        self.name += f" ({kind})"
        self.key += f"({kind})"
        self.augmenter = self.available_methods[kind](**params)
         
    def augment(self, *args):
        return self.augmenter.augment(*args)