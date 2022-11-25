from abc import ABC

from keras.datasets import mnist
from keras.utils import to_categorical

from src.data_utils.DataSet import DataSet
from src.data_utils.ModelData import ModelData
from src.model_utils.SequentialModelRunnable import SequentialModelRunnable


class MNISTModelRunnable(ABC, SequentialModelRunnable):
    """Sequential model runnable with MNIST data processing.
    """

    def load_and_prepare_data(self) -> ModelData:
        train_images, train_labels, test_images, test_labels = mnist.load_data()

        training_set = DataSet(
            train_images.reshape((60000, 28 * 28)).astype("float32") / 255,
            to_categorical(train_labels)
        )
        testing_set = DataSet(
            test_images.reshape((60000, 28 * 28)).astype("float32") / 255,
            to_categorical(test_labels)
        )

        return ModelData(training_set, testing_set)
