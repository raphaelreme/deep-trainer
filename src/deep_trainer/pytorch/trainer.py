"""Deprecated module.

Will be removed in a future version.
Please use `deep_trainer.trainer` instead.
"""

import warnings

from deep_trainer.trainer import PytorchTrainer, build_description, cyclic_iterator, round_to_n  # noqa: F401

warnings.warn(
    "deep_trainer.pytorch.trainer is deprecated and will be removed "
    "in a future version. Please use deep_trainer.trainer instead.",
    DeprecationWarning,
    stacklevel=2,
)
