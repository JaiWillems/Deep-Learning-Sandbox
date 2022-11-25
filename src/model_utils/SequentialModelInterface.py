import abc

from keras.models.cloning import Sequential

from src.data_utils.ModelData import ModelData
from src.performance_utils.TestingPerformance import TestingPerformance
from src.performance_utils.TrainingPerformance import TrainingPerformance


class SequentialModelInterface(metaclass=abc.ABCMeta):
    """Framework for sequential machine learning networks.
    """

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (hasattr(subclass, 'load_and_prepare_data') and
                callable(subclass.load_and_prepare_data) and
                hasattr(subclass, 'build_model') and
                callable(subclass.build_model) and
                hasattr(subclass, 'train_model') and
                callable(subclass.train_model) and
                hasattr(subclass, 'test_model') and
                callable(subclass.test_model) or
                NotImplemented)

    @abc.abstractmethod
    def load_and_prepare_data(self) -> ModelData:
        """Load and preprocess data for the model.

        This method loads the training and testing data; the data shall
        pre-processed via vectorizing, regularizing, and feature extraction.

        Returns
        -------
        ModelData
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_model(self) -> Sequential:
        """Generate a Keras Sequential model.

        This method defines the network architecture and hyperparameters to use.

        Returns
        -------
        Sequential
            A Keras sequential model architecture.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_model(self, model: Sequential,
                    data: ModelData) -> TrainingPerformance:
        """Train model.

        This method trains the model on the input training data and tracks the
        performance using a validation set.

        Parameters
        ----------
        model : Sequential
            Model to be trained.
        data : ModelData
            Training, validation, and testing data.

        Returns
        -------
        TrainingPerformance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def test_model(self, model: Sequential,
                   data: ModelData) -> TestingPerformance:
        """Test model.

        This method tests the trained model on the input testing data and
        provides performance data.

        Parameters
        ----------
        model : Sequential
            Model to be tested.
        data : ModelData
            Training, validation, and testing data.

        Returns
        -------
        TestingPerformance
        """
        raise NotImplementedError
