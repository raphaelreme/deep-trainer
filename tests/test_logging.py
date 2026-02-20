from __future__ import annotations

import pytest

from deep_trainer import logging as dt_logging


def test_dict_logger_creates_series_and_appends():
    logger = dt_logging.DictLogger()

    logger.log("Loss", 1.0, 0)
    logger.log("Loss", 0.5, 1)
    logger.log("Accuracy", 0.25, 1)

    assert "Loss" in logger.logs
    assert "Accuracy" in logger.logs

    loss_steps, loss_vals = logger.logs["Loss"]
    acc_steps, acc_vals = logger.logs["Accuracy"]

    assert loss_steps == [0, 1]
    assert loss_vals == [1.0, 0.5]
    assert acc_steps == [1]
    assert acc_vals == [0.25]


def test_multi_logger_fanout():
    logger_a = dt_logging.DictLogger()
    logger_b = dt_logging.DictLogger()

    logger = dt_logging.MultiLogger([logger_a, logger_b])
    logger.log("M", 2.0, 10)

    assert logger_a.logs["M"][0] == [10]
    assert logger_a.logs["M"][1] == [2.0]
    assert logger_b.logs == logger_a.logs


def test_tensorboard_logger_calls_add_scalar(monkeypatch: pytest.MonkeyPatch):
    # Let's not test tensorboard and simply mock it.
    calls = []

    class FakeWriter:
        def __init__(self, log_dir: str):
            self.log_dir = log_dir

        def add_scalar(self, name, value, step):
            calls.append((name, value, step))

    monkeypatch.setattr(dt_logging.torch.utils.tensorboard, "SummaryWriter", FakeWriter)

    logger = dt_logging.TensorBoardLogger("logs")
    logger.log("Loss", 1.23, 4)

    assert calls == [("Loss", 1.23, 4)]


def test_old_module_is_depreciated():
    with pytest.warns(DeprecationWarning, match="deep_trainer.pytorch.logging is deprecated"):
        import deep_trainer.pytorch.logging  # noqa: F401, PLC0415
