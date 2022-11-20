from typing import List

from src.data_utils.DataSet import DataSet


class Fold:
    """Container combining training and validation sets.

    Parameters
    ----------
    training_set : DataSet
    validation_set : DataSet
    """

    def __init__(self, training_set: DataSet, validation_set: DataSet) -> None:
        self._training_set = training_set
        self._validation_set = validation_set

    @property
    def training_set(self) -> DataSet:
        """Get training set.

        Returns
        -------
        DataSet
        """
        return self._training_set

    @property
    def validation_set(self) -> DataSet:
        """Get validation set.

        Returns
        -------
        DataSet
        """
        return self._validation_set


class KFold:
    """Container combining multiple folds.

    Parameters
    ----------
    folds : List[Fold]
    """

    def __init__(self, folds: List[Fold]) -> None:
        self._folds = folds

    @property
    def get_folds(self) -> List[Fold]:
        """Get folds.

        Returns
        -------
        List[Fold]
        """
        return self._folds


class IterativeKFold:
    """Container combining multiple K-folds.

    Parameters
    ----------
    k_folds : List[KFold]
    """

    def __init__(self, k_folds: List[KFold]) -> None:
        self._k_folds = k_folds

    @property
    def get_k_folds(self) -> List[KFold]:
        """Get k-folds.

        Returns
        -------
        List[KFold]
        """
        return self._k_folds
