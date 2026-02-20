import math
import pathlib

import pytest
import torch
from torch import nn

from deep_trainer import logging as dt_logging
from deep_trainer import metric as dt_metric
from deep_trainer.trainer import PytorchTrainer


def _seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _TinyDataset(torch.utils.data.Dataset):
    """Deterministic binary classification dataset (~ linearly separable)."""

    def __init__(self, n: int = 128, dim=2):
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn((n // 2, dim), generator=g) - 1.0
        x1 = torch.randn((n // 2, dim), generator=g) + 1.0
        self.x = torch.cat([x0, x1], dim=0)
        self.y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)], dim=0).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _make_loaders(batch_size=4):
    train = _TinyDataset(64)
    val = _TinyDataset(32)
    test = _TinyDataset(32)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _make_model():
    # Simple linear classifier
    return nn.Linear(2, 2)


def test_init_trainer_minimal(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    _seed_everything(0)
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)

    trainer = PytorchTrainer(model, optimizer)

    assert trainer.scheduler is None
    assert not trainer.metrics_handler.metrics
    assert not trainer.use_amp
    assert not trainer.scaler._enabled
    assert trainer.output_dir.resolve() == (tmp_path / "experiments").resolve()
    assert trainer.device.type == "cpu"


def test_train_step(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    handler = dt_metric.MetricsHandler(
        [dt_metric.Accuracy(train=True, evaluate=False), dt_metric.Error(train=False, evaluate=True)]
    )
    logger = dt_logging.DictLogger()
    trainer = PytorchTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics_handler=handler,
        device=torch.device("cpu"),
        logger=logger,
        output_dir=tmp_path,
        save_mode="never",
        use_amp=False,
    )

    train_loader, _, _ = _make_loaders()
    batch = next(iter(train_loader))
    criterion = nn.CrossEntropyLoss()

    handler.train()
    metrics = trainer.train_step(batch, criterion)

    assert "Loss" in metrics
    assert "Accuracy" in metrics
    assert "Error" not in metrics
    assert isinstance(metrics["Loss"], float)
    assert metrics["Accuracy"] == handler.metrics[0].last_value


def test_eval_step(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    handler = dt_metric.MetricsHandler(
        [dt_metric.Accuracy(train=True, evaluate=False), dt_metric.Error(train=False, evaluate=True)]
    )
    trainer = PytorchTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics_handler=handler,
        device=torch.device("cpu"),
        logger=None,
        output_dir=tmp_path,
        save_mode="never",
        use_amp=True,
    )

    assert trainer.scaler._enabled  # Quick check for GradScaler

    _, val_loader, _ = _make_loaders()
    batch = next(iter(val_loader))

    handler.eval()
    metrics = trainer.eval_step(batch)

    assert "Loss" not in metrics
    assert "Accuracy" not in metrics
    assert "Error" in metrics
    assert metrics["Error"] == handler.metrics[1].last_value


def test_complete_example(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    handler = dt_metric.MetricsHandler(
        [dt_metric.Accuracy(train=True, evaluate=False), dt_metric.Error(train=False, evaluate=True)]
    )
    logger = dt_logging.DictLogger()

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        scheduler=None,
        metrics_handler=handler,
        device=torch.device("cpu"),
        logger=logger,
        output_dir=tmp_path,
        save_mode="small",
        use_amp=False,
    )

    # Create a file in checkpoints that should not be cleaned
    (tmp_path / "checkpoints" / "do_not_remove_me.file").touch()

    train_loader, val_loader, test_loader = _make_loaders(batch_size=16)
    criterion = nn.CrossEntropyLoss()

    # Compute initial accuracy
    initial_metrics = trainer.evaluate(test_loader)
    assert len(initial_metrics) == 1
    assert "Accuracy" not in initial_metrics
    assert "Error" in initial_metrics

    # Then let's train a bit
    trainer.train(epochs=5, train_loader=train_loader, criterion=criterion, val_loader=val_loader)

    # In small mode, older epoch ckpts should have been cleaned (only best + last remain)
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {p.name for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"}

    assert ckpts == {"best.ckpt", f"{trainer.epoch}.ckpt"}
    assert (tmp_path / "checkpoints" / "do_not_remove_me.file").exists()

    # Check logs
    assert set(logger.logs) == {
        "A_train_aggregate/Accuracy",
        "A_train_batch/Accuracy",
        "A_train_aggregate/Loss",
        "A_train_batch/Loss",
        "B_val/Loss",  # Has been added, as no validation_metric is set
        "B_val/Error",
        "Z_other/scale",
        "Z_other/lr_0",
    }
    assert len(set(logger.logs["Z_other/lr_0"][1])) == 1  # Single lr for all the training
    assert logger.logs["Z_other/scale"][1][0] == 1.0  # AMP is disabled

    # Check that metrics have improved after training
    final_metrics = trainer.evaluate(test_loader)
    assert len(final_metrics) == 1
    assert "Accuracy" not in final_metrics
    assert "Error" in final_metrics

    assert final_metrics["Error"] < initial_metrics["Error"]
    threshold = 0.2  # Should reach less than 20% of errors as this is almost separable
    assert final_metrics["Error"] < threshold


def test_complete_example_with_validation_metric_and_save_all_and_amp(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    confusion = dt_metric.ConfusionMatrix(2)
    handler = dt_metric.MetricsHandler(
        [dt_metric.F1(confusion), dt_metric.Precision(confusion), dt_metric.Recall(confusion)]
    )
    handler.set_validation_metric(1)  # Use Precision as the main metric
    metric = handler.get_validation_metric().display_name  # type: ignore[union-attr]
    logger = dt_logging.DictLogger()

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        scheduler=None,
        metrics_handler=handler,
        device=torch.device("cpu"),
        logger=logger,
        output_dir=tmp_path,
        save_mode="all",
        use_amp=True,
    )

    train_loader, val_loader, _ = _make_loaders(batch_size=4)
    criterion = nn.CrossEntropyLoss()

    # Let's train on shorter epochs and shorter batch size to increase noise and have best != last
    trainer.train(epochs=5, train_loader=train_loader, criterion=criterion, val_loader=val_loader, epoch_size=2)

    # In all mode, we should find all the checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {p.name for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"}

    assert ckpts == {"best.ckpt", "1.ckpt", "2.ckpt", "3.ckpt", "4.ckpt", "5.ckpt"}

    # Check logs
    assert set(logger.logs) == {
        "A_train_aggregate/F1-macro",
        "A_train_batch/F1-macro",
        "A_train_aggregate/Precision-macro",
        "A_train_batch/Precision-macro",
        "A_train_aggregate/Recall-macro",
        "A_train_batch/Recall-macro",
        "A_train_aggregate/Loss",
        "A_train_batch/Loss",
        # "B_val/Loss",  # validation metric is set, no need to evaluate Loss
        "B_val/F1-macro",
        "B_val/Precision-macro",
        "B_val/Recall-macro",
        "Z_other/scale",
        "Z_other/lr_0",
    }
    assert len(set(logger.logs["Z_other/lr_0"][1])) == 1  # Single lr for all the training
    assert logger.logs["Z_other/scale"][1][0] != 1.0  # AMP is enabled

    # Final metrics
    final_metrics = trainer.evaluate(val_loader)
    assert metric in final_metrics

    best_epoch = trainer.best_epoch
    best_val = trainer.best_val

    # Reload best model over the 5 epochs
    trainer.load(tmp_path / "checkpoints" / "best.ckpt")

    assert trainer.epoch == best_epoch

    best_metrics = trainer.evaluate(val_loader)
    assert metric in best_metrics
    assert best_val == best_metrics[metric]  # F1 is used as validation metric

    # Check that this test runs for a case where best != last
    assert best_metrics[metric] > final_metrics[metric]


def test_complete_example_without_validation_nor_save_but_schedule(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    handler = dt_metric.MetricsHandler([dt_metric.Accuracy(train=True, evaluate=True)])
    logger = dt_logging.DictLogger()

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        scheduler=torch.optim.lr_scheduler.ConstantLR(opt, total_iters=5),
        metrics_handler=handler,
        device=torch.device("cpu"),
        logger=logger,
        output_dir=tmp_path,
        save_mode="never",
        use_amp=False,
    )

    train_loader, _, test_loader = _make_loaders(batch_size=4)
    criterion = nn.CrossEntropyLoss()

    # Let's train on shorter epochs and shorted batch size
    trainer.train(epochs=5, train_loader=train_loader, criterion=criterion)

    # In never mode, there are no checkpoints
    ckpt_dir = tmp_path / "checkpoints"
    ckpts = {p.name for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"}

    assert not ckpts

    # Check logs
    assert len(set(logger.logs["Z_other/lr_0"][1])) != 1  # Lr scheduling
    assert logger.logs["Z_other/scale"][1][0] == 1.0  # AMP is disabled

    # Final metrics
    final_metrics = trainer.evaluate(test_loader)

    threshold = 0.8  # Should reach more than 80% of accuracy as this is almost separable
    assert final_metrics["Accuracy"] > threshold


def test_save_and_load_restore_state(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        scheduler=torch.optim.lr_scheduler.ConstantLR(opt, total_iters=5),
        device=torch.device("cpu"),
        logger=dt_logging.DictLogger(),
        output_dir=tmp_path,
        save_mode="never",
        use_amp=True,
    )

    train_loader, val_loader, _ = _make_loaders(batch_size=16)
    criterion = nn.CrossEntropyLoss()

    trainer.train(epochs=1, train_loader=train_loader, criterion=criterion, val_loader=val_loader)
    ckpt_path = pathlib.Path(tmp_path) / "checkpoints" / "test.ckpt"
    trainer.save("test.ckpt")
    assert ckpt_path.exists()

    old_weight = model.weight.clone()

    with torch.no_grad():
        model.weight.add_(10.0)

    # Also mutate counters
    trainer.epoch = 123
    trainer.train_steps = -50
    trainer.val = float("nan")

    trainer.load(ckpt_path, strict=True)

    assert (model.weight == old_weight).all()

    # Loaded values should match checkpoint
    assert trainer.epoch == 1
    assert trainer.train_steps > 0
    assert not math.isnan(trainer.val)

    # best_* should be set to loaded baseline
    assert trainer.best_epoch == trainer.epoch
    assert trainer.best_val == trainer.val


def test_load_fails_or_warns_for_missing_keys(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(opt, total_iters=5)

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        device=torch.device("cpu"),
        logger=dt_logging.DictLogger(),
        output_dir=tmp_path,
        save_mode="never",
        use_amp=True,
    )

    train_loader, val_loader, _ = _make_loaders(batch_size=16)
    criterion = nn.CrossEntropyLoss()

    trainer.train(epochs=1, train_loader=train_loader, criterion=criterion, val_loader=val_loader)

    ckpt_path = pathlib.Path(tmp_path) / "checkpoints" / "test.ckpt"

    # Saving & Loading works
    trainer.save("test.ckpt")
    trainer.load(ckpt_path, strict=True)

    # Let's not save the scaler (amp = false)
    trainer.scaler._enabled = False  # Hacky way to disable amp
    trainer.save("test.ckpt")
    trainer.scaler._enabled = True
    with pytest.raises(ValueError, match="Missing scaler"):
        trainer.load(ckpt_path, strict=True)
    with pytest.warns(Warning, match="Missing scaler"):
        trainer.load(ckpt_path, strict=False)

    # Let's not save the scheduler
    trainer.scheduler = None
    trainer.save("test.ckpt")
    trainer.scheduler = scheduler

    with pytest.raises(ValueError, match="Missing scheduler"):
        trainer.load(ckpt_path, strict=True)
    with pytest.warns(Warning, match="Missing scheduler"):
        trainer.load(ckpt_path, strict=False)


def test_load_restore_optim_hp_false_keeps_current_lr(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        logger=dt_logging.DictLogger(),
        output_dir=tmp_path,
        save_mode="never",
    )

    # Save initial state
    trainer.save("test.ckpt")
    ckpt_path = pathlib.Path(tmp_path) / "checkpoints" / "test.ckpt"

    # Change LR then load without restoring HP: LR should remain changed
    for g in trainer.optimizer.param_groups:
        g["lr"] = 0.001

    trainer.load(str(ckpt_path), strict=True, restore_optim_hp=False)

    for g in trainer.optimizer.param_groups:
        assert g["lr"] == pytest.approx(0.001)


def test_trainer_with_user_added_metrics(tmp_path: pathlib.Path):
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = dt_logging.DictLogger()

    class Trainer(PytorchTrainer):
        def train_step(self, batch, criterion):
            metrics = super().train_step(batch, criterion)
            metrics["TEST"] = 0.5
            return metrics

        def eval_step(self, batch):
            metrics = super().eval_step(batch)
            metrics["TEST"] = 0.3
            metrics["TEST_EVAL"] = 0.7
            return metrics

    trainer = Trainer(
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        logger=logger,
        output_dir=tmp_path,
        save_mode="never",
    )

    train_loader, val_loader, test_loader = _make_loaders(batch_size=16)
    criterion = nn.CrossEntropyLoss()

    # Let's train on shorter epochs and shorted batch size
    trainer.train(epochs=2, train_loader=train_loader, criterion=criterion, val_loader=val_loader)

    assert "A_train_aggregate/TEST" in logger.logs
    assert "A_train_batch/TEST" in logger.logs
    assert "B_val/TEST" in logger.logs
    assert "B_val/TEST_EVAL" in logger.logs

    metrics = trainer.evaluate(test_loader)

    assert metrics["TEST"] == pytest.approx(0.3)
    assert metrics["TEST_EVAL"] == pytest.approx(0.7)


def test_noisy_training_to_trigger_last_lines(tmp_path: pathlib.Path):
    # Done to trigger a usecase with a metric to be minimized (i.e. loss here)
    # and a loss that is deacreasing between two consecutive epochs
    _seed_everything(0)
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = PytorchTrainer(
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        logger=dt_logging.DictLogger(),
        output_dir=tmp_path,
        save_mode="never",
        use_amp=True,
    )

    train_loader, val_loader, _ = _make_loaders(batch_size=1)
    criterion = nn.CrossEntropyLoss()

    trainer.train(epochs=5, train_loader=train_loader, criterion=criterion, val_loader=val_loader)


def test_old_module_is_depreciated():
    with pytest.warns(DeprecationWarning, match="deep_trainer.pytorch.trainer is deprecated"):
        import deep_trainer.pytorch.trainer  # noqa: F401, PLC0415
