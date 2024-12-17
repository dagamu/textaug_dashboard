from api.dataset import DatasetManager
from api.preprocessing import PreprocessingManager
from api.classification import ClassificationManager
from api.sampling import SamplingManager
from api.augmentation import AugmentationManager
from api.pipeline.runner import PipelineRunner

class Session():
    def __init__(self):
        self.datasets = DatasetManager()
        self.preprocessing = PreprocessingManager()
        self.sampling = SamplingManager()
        self.classification = ClassificationManager()
        self.aug_manager = AugmentationManager()
        self.runner = PipelineRunner(self)
        
    def apply_preprocessing(self, dataset):
        for df in self.datasets.items:
            if dataset.key_train == df.key_train:
                self.preprocessing.apply_to(dataset)
                
    def pipeline_run(self, update_fn):
        self.runner.run(update_fn)
        