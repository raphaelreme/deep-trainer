"""Metrics used for training and evaluation procedure

Provide some examples of useful metrics for classification
"""

from typing import Any, Callable, Dict, Iterator, List

import torch


class Metric:
    """Base class for metric to be tracked during training.

    Could probably be more general but for now let's assume that any metric can be computed
    from the predictions of the models and some targets.

    A metric is updated at each batch, and can be aggregated whenever it is necessary.
    A last value can be accessed. But for some metrics it will not make sense and NaN can be returned

    Attr:
        last_value (float): Value of the metric on the last update (can be NaN)
        display_name (str): Name to be used in the loggers
        train (bool): Compute this metric during train loop
        evaluate (bool): Compute this metric during evaluate loop
    """

    def __init__(self, display_name: str = None, train: bool = True, evaluate: bool = True):
        self.last_value = float("nan")
        self.display_name = display_name if display_name else self.__class__.__name__
        self.train = train
        self.evaluate = evaluate

    def update(self, predictions: Any, targets: Any):
        """Update the metric with the current batch, and set the metric last_value for this
        particular batch if this makes sense.

        Args:
            predictions (Any): Output of the model
            targets (Any): Target values
        """
        raise NotImplementedError

    def aggregate(self) -> float:
        """Aggregate the metric for all the batches seen since last reset.

        Returns:
            float: The aggregated value
        """
        raise NotImplementedError

    def reset(self):
        """Reset the metric."""
        self.last_value = float("nan")


class MetricsHandler:
    """Handle a given list of metrics"""

    def __init__(self, metrics: List[Metric]):
        self.training = True
        self.metrics = metrics

    def current_metrics(self) -> Iterator[Metric]:
        """Return an iterator on the current metrics

        Returns:
            Iterator[Metric]: Metric to use in the current mode
        """

        def func(metric: Metric) -> bool:
            if self.training and not metric.train:
                return False
            if (not self.training) and (not metric.evaluate):
                return False
            return True

        return filter(func, self.metrics)

    def train(self, training: bool = True):
        """Switch to train metrics

        Args:
            training (bool): Switch to train if True or to eval if False
        """
        self.training = training

    def eval(self):
        """Switch to evaluate metrics"""
        self.train(False)

    def update(self, predictions: Any, targets: Any):
        """Update all the metrics for the current mode

        For a more finegrained control, each metric can be updated on its own.

        You can for instance go through the current metrics with:
        ```
        for metric in metric_handler:
            metric.update(...)
        ```

        Args:
            predictions (Any): Output of the model
            targets (Any): Target values
        """
        for metric in self.current_metrics():
            metric.update(predictions, targets)

    @property
    def last_values(self) -> Dict[str, float]:
        """Dict of last computed metrics"""
        values = {}

        for metric in self.current_metrics():
            values[metric.display_name] = metric.last_value

        return values

    @property
    def aggregated_values(self) -> Dict[str, float]:
        """Dict of aggregated metrics"""
        values = {}

        for metric in self.current_metrics():
            values[metric.display_name] = metric.aggregate()

        return values

    def reset(self):
        """Reset all the metrics"""
        for metric in self.metrics:
            metric.reset()

    def __iter__(self):
        return iter(self.current_metrics())


class PytorchMetric(Metric):
    """Average a pytorch loss function"""

    def __init__(self, loss_function: Callable, display_name: str = None, train: bool = True, evaluate: bool = True):
        if display_name is None:
            display_name = getattr(loss_function, "__name__", getattr(loss_function, "__class__").__name__)

        super().__init__(display_name=display_name, train=train, evaluate=evaluate)
        self.loss_function = loss_function
        self._sum = 0
        self._n_samples = 0

    def update(self, predictions: Any, targets: Any):
        loss = self.loss_function(predictions, targets)
        batch_size = targets.shape[0]
        self._sum += loss.item() * batch_size
        self._n_samples += batch_size
        self.last_value = loss.item()

    def aggregate(self) -> float:
        return self._sum / self._n_samples

    def reset(self):
        super().reset()
        self._sum = 0
        self._n_samples = 0


class Accuracy(PytorchMetric):
    """Accuracy metric.

    Compute the proportion of corrected predicted targets. (Multiclass classification)
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, "Accuracy", train, evaluate)

    @staticmethod
    def _compute(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted_targets = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_targets == targets) / torch.numel(targets)


class Error(PytorchMetric):
    """Error metric.

    Compute the proportion of corrected predicted targets. (Multiclass classification)
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, "Error", train, evaluate)

    @staticmethod
    def _compute(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted_targets = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_targets != targets) / torch.numel(targets)


class TopK(PytorchMetric):
    """TopK metric.

    A prediction is valid if the true target is in the top k. (Multiclass classification)
    """

    def __init__(self, k: int, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, f"Top-{k}", train, evaluate)
        self.k = k

    def _compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        top_k = torch.topk(predictions, self.k, -1).indices
        return (top_k == targets.unsqueeze(-1)).any(-1).sum() / torch.numel(targets)


# TODO: Have a MultiClass metric that records all the confusion matrix
# And subclasses that average it as needed
class BalancedAccuracy(Metric):
    """Balanced Accuracy Metric

    Compute the Balanced Accuracy of the predictions. <=> Average the recall of each target.
    This criterion has meaning only at aggregation time.
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__("BalancedAccuracy", train, evaluate)
        self.target_occurences: Dict[int, int] = {}
        self.true_positives: Dict[int, int] = {}

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        predicted_targets = torch.argmax(predictions, dim=1)

        assert predicted_targets.shape == targets.shape
        assert len(predicted_targets.shape) == 1

        for i in range(predicted_targets.shape[0]):
            target: int = targets[i].item()  # type: ignore
            predicted: int = predicted_targets[i].item()  # type: ignore

            self.target_occurences[target] = self.target_occurences.get(target, 0) + 1
            self.true_positives[target] = self.true_positives.get(target, 0) + (target == predicted)

    def aggregate(self) -> float:
        recall = [self.true_positives[target] / self.target_occurences[target] for target in self.target_occurences]
        return sum(recall) / len(recall)

    def reset(self):
        super().reset()
        self.target_occurences = {}
        self.true_positives = {}
