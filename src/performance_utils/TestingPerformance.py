class TestingPerformance:
    """Container for training performance data.

    Parameters
    ----------
    testing_loss : float
        The model's loss on the testing data.
    testing_metrics : dict, optional
        Dictionary with string keys representing the metric names and float
        values representing the testing metrics' data.
    """

    def __init__(self, testing_loss: float,
                 testing_metrics: dict = None) -> None:
        if testing_metrics is None:
            testing_metrics = {}
        self._testing_loss = testing_loss
        self._testing_metrics = testing_metrics

    @property
    def testing_loss(self) -> float:
        return self._testing_loss

    @property
    def testing_metrics(self) -> dict:
        return self._testing_metrics
