"""Deprecated module.

Will be removed in a future version.
Please use `deep_trainer.metric` instead.
"""

import warnings

from deep_trainer.metric import (  # noqa: F401
    F1,
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    Metric,
    MetricsHandler,
    Precision,
    Prerequisite,
    PytorchMetric,
    Recall,
    TopK,
)

warnings.warn(
    "deep_trainer.pytorch.metric is deprecated and will be removed "
    "in a future version. Please use deep_trainer.metric instead.",
    DeprecationWarning,
    stacklevel=2,
)
