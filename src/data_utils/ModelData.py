from src.data_utils.DataSet import DataSet
from src.data_utils.Folds import Fold, KFold, IterativeKFold


class ModelData:
    """Container for training and testing data.

    Parameters
    ----------
    training_set : DataSet
        The training samples and labels.
    testing_set : DataSet
        The testing samples and labels.
    """

    def __init__(self, training_set: DataSet, testing_set: DataSet):
        self._training_set = training_set
        self._testing_set = testing_set

    def training_set(self) -> DataSet:
        """Getter for the training set.

        Returns
        -------
        DataSet
        """
        return self._training_set

    def hold_out_training_set(self, validation_size: int) -> Fold:
        """Getter for the training set using hold-out validation.

        Parameters
        ----------
        validation_size : int
            The number of samples in the validation set.

        Returns
        -------
        Fold
            Fold containing the training and validation sets.
        """
        if validation_size > self._training_set.size:
            raise ValueError("The validation set cannot contain more samples "
                             "than in the training set.")

        return Fold(self._training_set[validation_size:],
                    self._training_set[:validation_size])

    def k_fold_training_set(self, k: int) -> KFold:
        """Getter for the training set using k-fold validation.

        Parameters
        ----------
        k : int
            The number of folds.

        Returns
        -------
        KFold
            K-fold containing the training and validation sets.
        """
        validation_size = self._training_set.size // k
        return KFold([
            Fold(self._training_set[:validation_size * fold_number],
                 self._training_set[validation_size * fold_number:])
            for fold_number in range(k)
        ])

    def iterative_k_fold_training_set(self, p: int, k: int) -> IterativeKFold:
        """Getter for the training set using iterative k-fold validation.

        Parameters
        ----------
        p : int
            The number of k-fold iterations.
        k : int
            The number of folds.

        Returns
        -------
        IterativeKFold
            Iterative k-fold containing the training and validation sets.
        """
        return IterativeKFold([self.k_fold_training_set(k) for _ in range(p)])

    def testing_set(self) -> DataSet:
        """Getter for the testing set.

        Returns
        -------
        DataSet
        """
        return self._testing_set
