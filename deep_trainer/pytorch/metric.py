"""Metrics used for training and evaluation procedure

Provide some examples of useful metrics for classification
"""

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import torch


class Prerequisite:
    """A common code to run before some metrics that can be shared between metrics

    The idea is that some metrics can define prerequisites (shared between metrics)
    Each prerequisite can also have its own prerequisites. The metrics handler will first
    update in order the prerequisites, then the metrics.

    It's up to the metric/prerequisite to fetch the work of its own prerequisites.

    An example is given with the ConfusionMatrix class.
    """

    def __init__(self) -> None:
        self.prerequisites: Set[Prerequisite] = set()

    def update(self, batch: Any, outputs: Any) -> None:
        """Update the prerequisite with the current batch

        Args:
            batch (Any): Batch used for outputs computation. Probably contains labels
            outputs (Any): Output of the model
        """
        raise NotImplementedError()

    def aggregate(self) -> None:
        """Aggregate the prerequisite"""
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the prerequisite"""
        raise NotImplementedError()


class Metric:
    """Base class for metric to be tracked during training.

    A metric is updated at each batch, and can be aggregated whenever it is necessary.
    A last value can be accessed. But for some metrics it will not make sense and NaN can be returned

    Attr:
        prerequisites (Set[Prerequisite]): Prerequisites to update and aggregate before this metric
            Prerequisites are updated by the MetricsHandler class. If you use prerequisites without a metrics handler
            you should update them manually.
        last_value (float): Value of the metric on the last update (can be NaN)
        display_name (str): Name to be used in the loggers
        train (bool): Compute this metric during train loop
        evaluate (bool): Compute this metric during evaluate loop
        minimize (bool): Should the metric be minimized or maximized
    """

    def __init__(self, display_name: Optional[str] = None, train: bool = True, evaluate: bool = True, minimize=True):
        self.prerequisites: Set[Prerequisite] = set()
        self.last_value = float("nan")
        self.display_name = display_name if display_name else self.__class__.__name__
        self.train = train
        self.evaluate = evaluate
        self.minimize = minimize

    def update(self, batch: Any, outputs: Any) -> None:
        """Update the metric with the current batch, and set the metric last_value for this
        particular batch if this makes sense.

        Args:
            batch (Any): Batch used for outputs computation. Probably contains labels
            outputs (Any): Output of the model
        """
        raise NotImplementedError

    def aggregate(self) -> float:
        """Aggregate the metric for all the batches seen since last reset.

        Returns:
            float: The aggregated value
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the metric."""
        self.last_value = float("nan")


class MetricsHandler:
    """Handle a given list of metrics"""

    def __init__(self, metrics: List[Metric]):
        self.training = True
        self.metrics = metrics
        self._validation_metric: Optional[Metric] = None

    def current_metrics(self) -> Iterator[Metric]:
        """Return an iterator on the current metrics

        Returns:
            Iterator[Metric]: Metric to use in the current mode
        """

        def func(metric: Metric) -> bool:
            return self.training and metric.train or not self.training and metric.evaluate

        return filter(func, self.metrics)

    @staticmethod
    def build_prerequisites(metrics: Iterable[Metric]) -> List[Prerequisite]:
        """Build all prerequisites from a list of metrics in a sorted order

        It will build a topological sort from leaf prerequisite to metrics.
        """
        # The sorting implementation is probably sub-optimal, but we rarely needs to sort more than
        # 10 prerequisites, so this will do fine
        # Will not detect circular dependencies
        seen = set()
        sorted_prerequisites: List[Prerequisite] = []

        def _rec_update(prerequisite: Prerequisite, parent: Optional[Prerequisite] = None):
            if prerequisite in seen:
                return

            seen.add(prerequisite)
            if parent is not None:
                # We add just before its first parent
                # Potential other parents can be added later and therefore
                # will always be after in the list
                sorted_prerequisites.insert(sorted_prerequisites.index(parent), prerequisite)
            else:
                # If no parent, we add at the end
                sorted_prerequisites.append(prerequisite)

            for other in prerequisite.prerequisites:
                _rec_update(other, prerequisite)

        for metric in metrics:
            for prerequisite in metric.prerequisites:
                _rec_update(prerequisite)

        return sorted_prerequisites

    def train(self, training: bool = True) -> None:
        """Switch to train metrics

        Args:
            training (bool): Switch to train if True or to eval if False
        """
        self.training = training

    def eval(self) -> None:
        """Switch to evaluate metrics"""
        self.train(False)

    def get_validation_metric(self) -> Optional[Metric]:
        """Get the validation metric

        Returns None if not set.

        Returns:
            Optional[Metric]
        """
        return self._validation_metric

    def set_validation_metric(self, index: int) -> None:
        """Select a validation metric

        Without validation metric, the train criterion is used by default to validate training.

        Args:
            index (int): Index of the metric
        """
        if not self.metrics[index].evaluate:
            raise ValueError(f"Metric {self.metrics[index]} at {index} is not in evaluate mode")
        self._validation_metric = self.metrics[index]

    def update(self, batch: Any, outputs: Any) -> None:
        """Update all the metrics for the current mode

        For a more finegrained control, each metric can be updated on its own.

        You can for instance go through the current metrics with:
        ```
        for metric in metric_handler:
            metric.update(...)
        ```

        Args:
            batch (Any): Batch used for outputs computation. Probably contains labels
            outputs (Any): Output of the model
        """
        for prerequisite in self.build_prerequisites(self.current_metrics()):
            prerequisite.update(batch, outputs)

        for metric in self.current_metrics():
            metric.update(batch, outputs)

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

        for prerequisite in self.build_prerequisites(self.current_metrics()):
            prerequisite.aggregate()

        values = {}

        for metric in self.current_metrics():
            values[metric.display_name] = metric.aggregate()

        return values

    def reset(self) -> None:
        """Reset all the metrics and prerequisites"""

        for prerequisites in self.build_prerequisites(self.metrics):
            prerequisites.reset()

        for metric in self.metrics:
            metric.reset()

    def __iter__(self):
        return self.current_metrics()


# Some metrics examples
class PytorchMetric(Metric):
    """Average a pytorch loss function"""

    def __init__(
        self,
        loss_function: Callable,
        display_name: Optional[str] = None,
        train: bool = True,
        evaluate: bool = True,
        minimize: bool = True,
    ):
        if display_name is None:
            display_name = getattr(loss_function, "__name__", getattr(loss_function, "__class__").__name__)

        super().__init__(display_name=display_name, train=train, evaluate=evaluate, minimize=minimize)
        self.loss_function = loss_function
        self._sum = 0
        self._n_samples = 0

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> None:
        _, targets = batch

        loss = self.loss_function(outputs, targets)
        batch_size = targets.shape[0]
        self._sum += loss.item() * batch_size
        self._n_samples += batch_size
        self.last_value = loss.item()

    def aggregate(self) -> float:
        return self._sum / self._n_samples

    def reset(self) -> None:
        super().reset()
        self._sum = 0
        self._n_samples = 0


class Accuracy(PytorchMetric):
    """Accuracy metric.

    Compute the proportion of corrected predicted targets. (Multiclass classification)
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, "Accuracy", train, evaluate, False)

    @staticmethod
    def _compute(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted_targets = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_targets == targets) / torch.numel(targets)


class Error(PytorchMetric):
    """Error metric.

    Compute the proportion of corrected predicted targets. (Multiclass classification)
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, "Error", train, evaluate, False)

    @staticmethod
    def _compute(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted_targets = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_targets != targets) / torch.numel(targets)


class TopK(PytorchMetric):
    """TopK metric.

    A prediction is valid if the true target is in the top k. (Multiclass classification)
    """

    def __init__(self, k: int, train: bool = True, evaluate: bool = True):
        super().__init__(self._compute, f"Top-{k}", train, evaluate, False)
        self.k = k

    def _compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        top_k = torch.topk(predictions, self.k, -1).indices
        return (top_k == targets.unsqueeze(-1)).any(-1).sum() / torch.numel(targets)


class BalancedAccuracy(Metric):
    """Balanced Accuracy Metric

    Compute the Balanced Accuracy of the predictions. <=> Average the recall of each target.
    This criterion has meaning only at aggregation time.

    Equivalent to Recall-macro
    """

    def __init__(self, train: bool = True, evaluate: bool = True):
        super().__init__("BalancedAccuracy", train, evaluate, False)
        self.target_occurences: Dict[int, int] = {}
        self.true_positives: Dict[int, int] = {}

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> None:
        _, targets = batch

        predicted_targets = torch.argmax(outputs, dim=1)

        assert predicted_targets.shape == targets.shape
        assert len(predicted_targets.shape) == 1

        for i in range(predicted_targets.shape[0]):
            target: int = targets[i].item()  # type: ignore
            predicted: int = predicted_targets[i].item()  # type: ignore

            self.target_occurences[target] = self.target_occurences.get(target, 0) + 1
            self.true_positives[target] = self.true_positives.get(target, 0) + (target == predicted)

    def aggregate(self) -> float:
        recall = [self.true_positives[target] / occurence for target, occurence in self.target_occurences.items()]
        return sum(recall) / len(recall)

    def reset(self) -> None:
        super().reset()
        self.target_occurences = {}
        self.true_positives = {}


# Examples of metrics with prerequisite
class ConfusionMatrix(Prerequisite):
    """Compute a confusion matrix as a prerequisite (Multiclass classification)"""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.confusion_matrix = torch.zeros((k, k))

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> None:
        _, targets = batch

        predicted_targets = torch.argmax(outputs, dim=1)

        assert predicted_targets.shape == targets.shape

        for i in range(predicted_targets.shape[0]):
            target: int = targets[i].item()  # type: ignore
            predicted: int = predicted_targets[i].item()  # type: ignore

            self.confusion_matrix[target, predicted] += 1

    def aggregate(self) -> None:
        pass  # Nothing to do

    def reset(self):
        self.confusion_matrix = torch.zeros((self.k, self.k))


class Recall(Metric):
    """Compute Recall from ConfusionMatrix prerequisite (Multiclass classification)

    Equivalent to BalancedAccuracy in the 'macro' setting.
    """

    def __init__(
        self, confusion_prerequisite: ConfusionMatrix, average="macro", train: bool = True, evaluate: bool = True
    ):
        """Constructor

        Args:
            confusion_prerequisite (ConfusionMatrix): Prerequisite needed to compute this metric
            average ("macro" | "weigthed"): Aggregation method of the score (Similar to sklearn)
                'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
                'weighted': Calculate metrics for each label, and find their weighted mean.
                Default: 'macro'
            train (bool): Use it for training
            evaluate (bool): Use it for evaluation
        """
        super().__init__(f"Recall-{average}", train, evaluate, False)
        self.prerequisites.add(confusion_prerequisite)
        self.confusion_matrix = confusion_prerequisite

        assert average in {"macro", "weighted"}  # Micro from sklearn is equivalent to accuracy. Let's not use it
        self.average = average

    def update(self, batch: Any, outputs: Any) -> None:
        pass  # All is done in the confusion prerequisite

    def aggregate(self) -> float:
        confusion_matrix = self.confusion_matrix.confusion_matrix

        target_occurences = confusion_matrix.sum(dim=1)
        true_positives = confusion_matrix.diag()

        # Only use targets with at least one sample
        valid_targets = target_occurences != 0
        target_occurences = target_occurences[valid_targets]
        true_positives = true_positives[valid_targets]

        if self.average == "macro":
            return torch.mean(true_positives / target_occurences).item()

        # weighted
        # recalls = tps / occurences
        # weighted_avg_recall = sum(recalls * occurences / total)
        #                     = sum(tps / total) = sum(tps) / total
        return (true_positives.sum() / target_occurences.sum()).item()


class Precision(Metric):
    """Compute Precision from ConfusionMatrix prerequisite (Multiclass classification)"""

    def __init__(
        self, confusion_prerequisite: ConfusionMatrix, average="macro", train: bool = True, evaluate: bool = True
    ):
        """Constructor

        Args:
            confusion_prerequisite (ConfusionMatrix): Prerequisite needed to compute this metric
            average ("macro" | "weigthed"): Aggregation method of the score (Similar to sklearn)
                'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
                'weighted': Calculate metrics for each label, and find their weighted mean.
                Default: 'macro'
            train (bool): Use it for training
            evaluate (bool): Use it for evaluation
        """
        super().__init__(f"Precision-{average}", train, evaluate, False)
        self.prerequisites.add(confusion_prerequisite)
        self.confusion_matrix = confusion_prerequisite

        assert average in {"macro", "weighted"}  # Micro from sklearn is equivalent to accuracy. Let's not use it
        self.average = average

    def update(self, batch: Any, outputs: Any) -> None:
        pass  # All is done in the confusion prerequisite

    def aggregate(self) -> float:
        confusion_matrix = self.confusion_matrix.confusion_matrix

        target_occurences = confusion_matrix.sum(dim=1)
        prediction_occurences = confusion_matrix.sum(dim=0)
        true_positives = confusion_matrix.diag()

        # Only keep targets with at least a prediction or a true sample
        valid_targets = (target_occurences + prediction_occurences) != 0
        target_occurences = target_occurences[valid_targets]
        prediction_occurences = prediction_occurences[valid_targets]
        true_positives = true_positives[valid_targets]

        precisions = true_positives / prediction_occurences
        precisions[prediction_occurences == 0] = 0  # If no prediction, then the precision is 0.

        if self.average == "macro":
            return precisions.mean().item()

        # weighted
        return (torch.sum(precisions * target_occurences) / target_occurences.sum()).item()


class F1(Metric):
    """Compute F1 score from ConfusionMatrix prerequisite (Multiclass classification)"""

    def __init__(
        self, confusion_prerequisite: ConfusionMatrix, average="macro", train: bool = True, evaluate: bool = True
    ):
        """Constructor

        Args:
            confusion_prerequisite (ConfusionMatrix): Prerequisite needed to compute this metric
            average ("macro" | "weigthed"): Aggregation method of the score (Similar to sklearn)
                'macro': Calculate metrics for each label, and find their unweighted mean.
                    This does not take label imbalance into account.
                'weighted': Calculate metrics for each label, and find their weighted mean.
                Default: 'macro'
            train (bool): Use it for training
            evaluate (bool): Use it for evaluation
        """
        super().__init__(f"F1-{average}", train, evaluate, False)
        self.prerequisites.add(confusion_prerequisite)
        self.confusion_matrix = confusion_prerequisite

        assert average in {"macro", "weighted"}
        self.average = average

    def update(self, batch: Any, outputs: Any) -> None:
        pass  # All is done in the confusion prerequisite

    def aggregate(self) -> float:
        confusion_matrix = self.confusion_matrix.confusion_matrix

        target_occurences = confusion_matrix.sum(dim=1)
        prediction_occurences = confusion_matrix.sum(dim=0)
        true_positives = confusion_matrix.diag()

        # Only keep targets with at least a true sample
        valid_targets = target_occurences != 0
        if prediction_occurences[~valid_targets].sum() != 0:
            raise ValueError(
                "Cannot compute recall for targets without any sample."
                "And unable to ignore some of these targets as they are predicted"
            )
        target_occurences = target_occurences[valid_targets]
        prediction_occurences = prediction_occurences[valid_targets]
        true_positives = true_positives[valid_targets]

        precisions = true_positives / prediction_occurences
        precisions[prediction_occurences == 0] = 0  # If no prediction, then the precision is 0.

        recalls = true_positives / target_occurences

        f1_score = 2 / (1 / precisions + 1 / recalls)  # Do not use prec * rec / (prec + rec) as it can yields nan

        if self.average == "macro":
            return f1_score.mean().item()

        # weighted
        return (torch.sum(f1_score * target_occurences) / target_occurences.sum()).item()
