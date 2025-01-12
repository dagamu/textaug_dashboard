from api.dataset import DatasetManager
from api.preprocessing import PreprocessingManager
from api.classification import ClassificationManager
from api.sampling import SamplingManager
from api.augmentation import AugmentationManager
from api.runner import PipelineRunner
from api.report import ReportGenerator

class Session():
    def __init__(self):
        self.datasets = DatasetManager()
        self.preprocessing = PreprocessingManager()
        self.sampling = SamplingManager()
        self.classification = ClassificationManager()
        self.aug_manager = AugmentationManager()
        self.runner = PipelineRunner(self)
        self.report = ReportGenerator(self)
        
    def apply_preprocessing(self, dataset):
        for df in self.datasets.items:
            if dataset.key_train == df.key_train:
                self.preprocessing.apply_to(dataset)
                
    def pipeline_run(self, update_fn):
        return self.runner.run(update_fn)
        