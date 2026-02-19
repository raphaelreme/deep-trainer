"""Deprecated module.

Will be removed in a future version.
Please use `deep_trainer.logging` instead.
"""

import warnings

from deep_trainer.logging import DictLogger, MultiLogger, TensorBoardLogger, TrainLogger  # noqa: F401

warnings.warn(
    "deep_trainer.pytorch.logging is deprecated and will be removed "
    "in a future version. Please use deep_trainer.logging instead.",
    DeprecationWarning,
    stacklevel=2,
)
