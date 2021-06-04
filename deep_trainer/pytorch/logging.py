"""Handle log in deep_trainer

You can create your own logger class following these examples.

By default the TensorBoard logger is used.
"""

from typing import Dict, List, Tuple

import torch.utils.tensorboard


class TrainLogger:
    """Base logger class. By default, do not log.

    Each logger should define its own `log` method.
    """

    def log(self, name: str, value: float, step: int):
        """Log a value for a given step of training"""
        # pass


class TensorBoardLogger(TrainLogger):
    """Log training values to tensorboard"""

    def __init__(self, output_dir: str):
        super().__init__()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)

    def log(self, name: str, value: float, step: int):
        self.tensorboard_writer.add_scalar(name, value, step)


class DictLogger(TrainLogger):
    """Log training values in the given dict

    A reference to the dict is kept in `logs` attribute. You can access the logs through it.
    """

    def __init__(self, logs: Dict[str, Tuple[List[int], List[float]]] = None):
        super().__init__()
        if logs is None:
            self.logs = {}
        else:
            self.logs = logs

    def log(self, name: str, value: float, step: int):
        if name not in self.logs:
            self.logs[name] = ([], [])
        steps, values = self.logs[name]
        steps.append(step)
        values.append(value)


class MultiLogger(TrainLogger):
    """Redirects logs to several loggers"""

    def __init__(self, loggers: List[TrainLogger]):
        super().__init__()
        self.loggers = loggers

    def log(self, name: str, value: float, step: int):
        for logger in self.loggers:
            logger.log(name, value, step)
