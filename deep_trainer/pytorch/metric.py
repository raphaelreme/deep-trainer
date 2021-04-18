"""Criteria/Metrics used for training and evaluation procedure"""

from typing import Callable

import torch


class Criterion:
    """Base class for criteria"""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update the criterion

        See `Criterion.update`
        """
        return self.update(predictions, labels)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Update the criterion with the current batch, and returns the loss of this batch
        if this make sense.

        Args:
            predictions (torch.Tensor): Output of the model
            labels (torch.Tensor): Target values

        Returns:
            torch.Tensor: The loss of the current batch if it exists. ("nan" otherwise)
                For some criteria the loss has meaning only when all the data is used.
        """
        raise NotImplementedError

    def aggregate(self) -> float:
        """Aggregate the criterion for all the batches seen in a single float.

        Returns:
            float: The criterion for all the batches seen.
        """
        raise NotImplementedError

    def reset(self) -> "Criterion":
        """Reset the criterion.

        Returns:
            Criterion: self
        """
        return self


class AveragingCriterion(Criterion):
    """Averaging Criterion

    Apply the loss_function to each batch and aggregate the result with an average.
    """

    def __init__(self, loss_function: Callable, name: str = None):
        """Constructor

        Args:
            loss_function (callable): Expect a pytorch function that should compute a loss given
                predictions and labels
        """
        if name is None:
            name = getattr(loss_function, "__name__", getattr(loss_function, "__class__").__name__)
        super().__init__(name)
        self.loss_function = loss_function
        self._sum = 0
        self._n_samples = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.loss_function(predictions, labels)
        self._sum += loss.item()
        self._n_samples += labels.shape[0]
        return loss

    def aggregate(self) -> float:
        return self._sum / self._n_samples

    def reset(self):
        self._sum = 0
        self._n_samples = 0
        return self


class Accuracy(AveragingCriterion):
    """Accuracy criterion.

    Compute the proportion of corrected predicted labels. (Multiclass classification)
    """

    def __init__(self):
        super().__init__(self._compute)

    @staticmethod
    def _compute(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predicted_labels = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_labels == labels) / torch.numel(labels)


class Error(AveragingCriterion):
    """Error criterion.

    Compute the proportion of wrongly predicted labels. (Multiclass classification)
    """

    def __init__(self):
        super().__init__(self._compute)

    @staticmethod
    def _compute(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predicted_labels = torch.argmax(predictions, dim=1)
        return torch.sum(predicted_labels != labels) / torch.numel(labels)


class BalancedAccuracy(Criterion):
    """Balanced Accuracy Criterion

    Compute the Balanced Accuracy of the predictions. <=> Average the recall of each labels.
    This criterion has meaning only at aggregation time.
    """

    def __init__(self):
        super().__init__("BalancedAccuracy")
        self.label_occurences = {}
        self.true_positives = {}

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predicted_labels = torch.argmax(predictions, dim=1)

        assert predicted_labels.shape == labels.shape
        assert len(predicted_labels.shape) == 1

        for i in range(predicted_labels.shape[0]):
            label = labels[i].item()
            predicted = predicted_labels[i].item()

            self.label_occurences[label] = self.label_occurences.get(label, 0) + 1
            self.true_positives[label] = self.true_positives.get(label, 0) + (label == predicted)

        return torch.tensor(float("nan"))  # pylint: disable=not-callable

    def aggregate(self) -> float:
        recall = [self.true_positives[key] / self.label_occurences[key] for key in self.label_occurences]
        return sum(recall) / len(recall)  # Averaged

    def reset(self):
        self.label_occurences = {}
        self.true_positives = {}
        return self
