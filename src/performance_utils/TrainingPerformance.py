import numpy as np


class TrainingPerformance:
    """Container for training performance data.

    Parameters
    ----------
    training_loss : np.ndarray
        Array containing the training loss for each training epoch.
    validation_loss : np.ndarray
        Array containing the validation loss for each training epoch.
    training_metrics : dict, optional
        Dictionary with string keys representing the metric names and array
        values containing the training metrics' data for each training epoch.
    validation_metrics : dict, optional
        Dictionary with string keys representing the metric names and array
        values containing the validation metrics' data for each training epoch.
    """

    def __init__(self, training_loss: np.ndarray, validation_loss: np.ndarray,
                 training_metrics: dict = None,
                 validation_metrics: dict = None) -> None:
        if validation_metrics is None:
            validation_metrics = {}
        if training_metrics is None:
            training_metrics = {}
        self._training_loss = training_loss
        self._validation_loss = validation_loss
        self._training_metrics = training_metrics
        self._validation_metrics = validation_metrics

    @property
    def training_loss(self) -> np.ndarray:
        return self._training_loss

    @property
    def validation_loss(self) -> np.ndarray:
        return self._validation_loss

    @property
    def training_metrics(self) -> dict:
        return self._training_metrics

    @property
    def validation_metrics(self) -> dict:
        return self._validation_metrics
