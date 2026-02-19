"""Logger used in the Trainer.

You can create your own logger class following these examples.

By default the TensorBoard logger is used.
"""

import sys

import torch.utils.tensorboard

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class TrainLogger:
    """Base logger class. By default, do not log.

    Each logger should define its own `log` method.
    """

    def log(self, name: str, value: float, step: int) -> None:
        """Log a value for a given step of training."""


class TensorBoardLogger(TrainLogger):
    """Log training values to tensorboard."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)

    @override
    def log(self, name: str, value: float, step: int) -> None:
        self.tensorboard_writer.add_scalar(name, value, step)


class DictLogger(TrainLogger):
    """Log training values in the given dict.

    A reference to the dict is kept in `logs` attribute. You can access the logs through it.
    """

    def __init__(self, logs: dict[str, tuple[list[int], list[float]]] | None = None):
        super().__init__()
        if logs is None:
            self.logs = {}
        else:
            self.logs = logs

    @override
    def log(self, name: str, value: float, step: int) -> None:
        if name not in self.logs:
            self.logs[name] = ([], [])
        steps, values = self.logs[name]
        steps.append(step)
        values.append(value)


class MultiLogger(TrainLogger):
    """Redirects logs to several loggers."""

    def __init__(self, loggers: list[TrainLogger]):
        super().__init__()
        self.loggers = loggers

    @override
    def log(self, name: str, value: float, step: int) -> None:
        for logger in self.loggers:
            logger.log(name, value, step)
