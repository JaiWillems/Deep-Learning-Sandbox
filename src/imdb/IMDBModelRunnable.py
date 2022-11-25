from abc import ABC

import numpy as np
from keras.datasets import imdb

from src.data_utils.DataSet import DataSet
from src.data_utils.ModelData import ModelData
from src.model_utils.SequentialModelRunnable import SequentialModelRunnable


class IMDBModelRunnable(ABC, SequentialModelRunnable):
    """Sequential model runnable with IMDB data processing.
    """

    NUM_WORDS = 10000

    def load_and_prepare_data(self) -> ModelData:
        train_data, train_labels, test_data, test_labels = imdb.load_data(
            num_words=self.NUM_WORDS
        )

        training_set = DataSet(
            self._encode_sequences(train_data),
            np.asarray(train_labels).astype("float32")
        )
        testing_set = DataSet(
            self._encode_sequences(test_data),
            np.asarray(test_labels).astype("float32")
        )

        return ModelData(training_set, testing_set)

    def _encode_sequences(self, sequences: list) -> np.ndarray:
        """One-hot encode sequences.
        """
        results = np.zeros((len(sequences), self.NUM_WORDS))
        for index, sequence in enumerate(sequences):
            results[index, sequence] = 1
        return results
