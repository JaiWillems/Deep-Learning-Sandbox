import abc

from src.model_utils.SequentialModelInterface import SequentialModelInterface
from src.performance_utils.ModelPerformance import ModelPerformance


class SequentialModelRunnable(metaclass=abc.ABCMeta, SequentialModelInterface):
    """Runnable to execute a model's setup, training, and testing.
    """

    def runnable(self) -> ModelPerformance:
        """Run the models learning and verification workflow.
        
        Returns
        -------
        TrainingPerformance, TestingPerformance
        """
        data = self.load_and_prepare_data()
        model = self.build_model()
        training_performance = self.train_model(model, data)
        testing_performance = self.test_model(model, data)

        return ModelPerformance(training_performance, testing_performance)
