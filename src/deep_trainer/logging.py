"""Training loggers.

This module defines a minimal scalar logging interface (`TrainLogger`) and a few concrete
implementations used by `PytorchTrainer`:

- `TensorBoardLogger`: writes scalars to TensorBoard.
- `DictLogger`: stores scalars in-memory for plotting/testing.
- `MultiLogger`: fans out logging calls to multiple loggers.

The logging contract is intentionally small: `log(name, value, step)`.
Names are typically hierarchical (e.g. "A_train_batch/Loss") so UIs can group them.
"""

import sys

import torch.utils.tensorboard

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class TrainLogger:
    """Abstract interface for logging scalar training signals.

    A logger receives scalar values identified by a string `name` at a given integer `step`.
    Implementations may write to disk, stream to a service, or store in memory.

    Implementers should treat `log()` as best-effort and avoid expensive blocking operations
    when called every batch.
    """

    def log(self, name: str, value: float, step: int) -> None:
        """Record a scalar value.

        Args:
            name (str): Metric name (often a hierarchical path like "A_train_batch/Loss").
            value (float): Scalar value to record.
            step (int): Global step index (monotonic increasing during training).
        """


class TensorBoardLogger(TrainLogger):
    """TensorBoard-backed logger for scalar values.

    Attributes:
        output_dir (str): Directory passed to `SummaryWriter(log_dir=...)`.
    """

    def __init__(self, output_dir: str):
        super().__init__()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)

    @override
    def log(self, name: str, value: float, step: int) -> None:
        self.tensorboard_writer.add_scalar(name, value, step)


class DictLogger(TrainLogger):
    """In-memory logger storing all scalars in a Python dict.

    Data structure:
        `logs[name] == (steps, values)` where:
        - `steps` is `list[int]`
        - `values` is `list[float]`

    This is useful for unit tests, notebooks, or custom plotting without TensorBoard.

    Attributes:
        logs (dict[str, tuple[list[int], list[float]]]): Logs for all metrics.
    """

    def __init__(self):
        super().__init__()
        self.logs: dict[str, tuple[list[int], list[float]]] = {}

    @override
    def log(self, name: str, value: float, step: int) -> None:
        if name not in self.logs:
            self.logs[name] = ([], [])
        steps, values = self.logs[name]
        steps.append(step)
        values.append(value)


class MultiLogger(TrainLogger):
    """Fan-out logger that forwards each log call to multiple loggers.

    Useful to combine, e.g., TensorBoard + in-memory logs.

    Args:
        loggers (list[TrainLogger]): List of logger instances to call sequentially.
    """

    def __init__(self, loggers: list[TrainLogger]):
        super().__init__()
        self.loggers = loggers

    @override
    def log(self, name: str, value: float, step: int) -> None:
        for logger in self.loggers:
            logger.log(name, value, step)
