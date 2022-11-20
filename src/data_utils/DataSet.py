import numpy as np


class DataSet:
    """Container for data samples and labels.

    Parameters
    ----------
    samples : np.ndarray
        Multidimensional array containing sample data.
    labels : np.ndarray
        Multidimensional array containing sample labels.

    Notes
    -----
    Sample arrays follow a standard construction:
        
        Vector data are 2D arrays of shape (samples, features).
        
        Time series data are 3D arrays of shape (samples, timesteps, features).
        
        Images are 4D arrays of shape (samples, height, width, channels).
        
        Video are 5D arrays of shape (samples, frames, height, width, channels).

    The label arrays follow the samples first conventions (i.e. the 0-axis of
    the labels array is the samples dimension).
    """

    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        self._samples = samples
        self._labels = labels

    def __getitem__(self, item) -> 'DataSet':
        return DataSet(self._samples[item], self._labels[item])

    @property
    def samples(self) -> np.ndarray:
        """Get sample array.
        
        Returns
        -------
        np.ndarray
        """
        return self._samples

    @property
    def labels(self) -> np.ndarray:
        """Get label array.
        
        Returns
        -------
        np.ndarray
        """
        return self._labels

    @property
    def size(self) -> int:
        """Get the number of samples in the set.

        Returns
        -------
        int
        """
        return self._samples.shape[0]
