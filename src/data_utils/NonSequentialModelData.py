import numpy as np

from src.data_utils.DataSet import DataSet
from src.data_utils.ModelData import ModelData


class NonSequentialModelData(ModelData):
    """Container for non-sequential training and testing data.

    This class is specific to non-sequential data as it randomly shuffles the
    training data when preparing the training sets.

    Parameters
    ----------
    training_set : DataSet
        The training samples and labels.
    testing_set : DataSet
        The testing samples and labels.
    """

    def __init__(self, training_set: DataSet, testing_set: DataSet) -> None:
        super(NonSequentialModelData, self).__init__(training_set, testing_set)

    def __getattribute__(self, attr):
        method = object.__getattribute__(self, attr)
        if not method:
            raise Exception("Method %s is not implemented." % attr)
        if callable(method):
            self._shuffle_training_data()
        return method

    def _shuffle_training_data(self) -> None:
        shuffler = np.random.permutation(self._training_set.size)
        self._training_set = DataSet(
            self._training_set.samples[shuffler],
            self._training_set.labels[shuffler]
        )
