"""Deep Trainer

Helps you train your deep learning models
"""

from .pytorch.metric import Metric
from .pytorch.logging import TrainLogger
from .pytorch.trainer import PytorchTrainer

__version__ = "0.1.0"
