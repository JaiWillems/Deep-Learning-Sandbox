from src.performance_utils.TestingPerformance import TestingPerformance
from src.performance_utils.TrainingPerformance import TrainingPerformance


class ModelPerformance(TrainingPerformance, TestingPerformance):
    """Container for model performance data.

    Parameters
    ----------
    training_performance : TrainingPerformance
    testing_performance : TestingPerformance
    """

    def __init__(self, training_performance: TrainingPerformance,
                 testing_performance: TestingPerformance):
        TrainingPerformance.__init__(
            self,
            training_performance.training_loss,
            training_performance.validation_loss,
            training_performance.training_metrics,
            training_performance.validation_metrics
        )
        TestingPerformance.__init__(
            self,
            testing_performance.testing_loss,
            testing_performance.testing_metrics
        )
