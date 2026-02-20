"""PyTorch Trainer.

This module provides:

- Small helpers (`round_to_n`, `build_description`, `cyclic_iterator`) used for progress display.
- `PytorchTrainer`, a lightweight training loop wrapper around a `torch.nn.Module` plus:
  optimizer, (optional) scheduler, mixed precision (optional), metrics, logging, and checkpointing.

Design goals
------------
- Keep the trainer easy to subclass: override `process_*_batch`, `train_step`, `eval_step`, or `backward`.
- Work with any `torch.utils.data.DataLoader` that yields `(inputs, targets)` by default.
- Provide simple checkpoint formats (`save` / `load`) suitable for experiment iteration.

Limitations
-----------
- Distributed training / multi-device (DDP/FSDP/TPU) is not handled here.
"""

from __future__ import annotations

import math
import pathlib
import sys
import warnings
from typing import TYPE_CHECKING, cast

import torch
import torch.utils.data
import tqdm.auto as tqdm

from . import logging as dt_logging
from . import metric

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Generator, Iterable, Iterator

# TODO: Avg Losses assume that the batches are evenly sized. (was solved before but not anymore)
# TODO: Log instead of print for epochs + time monitoring (Split data time vs Model time ?)
# TODO: Split the class into a abstract class + an implementation for classification?


def round_to_n(x: float, n_digits: int) -> float:
    """Round a floating-point number to a given number of significant digits.

    This rounds based on *significant figures* (not decimal places). For example:
    - `round_to_n(1234.5, 2) -> 1200.0`
    - `round_to_n(0.012345, 2) -> 0.012`

    Special cases:
    - If `x` is `0.0`, `inf`, `-inf`, or `nan`, it is returned unchanged.

    Args:
        x (float): Value to round.
        n_digits (int): Number of significant digits to keep. Must be >= 1.

    Returns:
        float: The rounded value.
    """
    if not math.isfinite(x) or x == 0:
        return x
    main_digit = math.floor(math.log10(abs(x)))
    return round(x, -main_digit + n_digits - 1)


def build_description(name: str, metrics: dict[str, float]) -> str:
    """Build a readable progress-bar description from metric values.

    The output is intended for tqdm `set_description`, e.g.:

        "Training --- Accuracy:  0.9231, Loss:  0.1234"

    Metrics are sorted by name for stable display, and values are rounded to 4 significant digits.

    Args:
        name (str): Prefix displayed first (e.g. "Training", "Testing").
        metrics (dict[str, float]): Mapping from metric display name to numeric value.

    Returns:
        str: A single-line description string.
    """
    desc = name

    if metrics:
        desc += " --- "

        for metric_name in sorted(metrics):
            desc += f"{metric_name}: {round_to_n(metrics[metric_name], 4):7}, "
        desc = desc[:-2]

    return desc


def cyclic_iterator(iterable: Iterable) -> Generator:
    """Yield items from `iterable` forever, restarting on exhaustion.

    This is similar to `itertools.cycle`, except it does *not* cache the first pass in memory.
    That matters for:
    - large datasets
    - data loaders with randomness (reshuffling per epoch)
    - iterables that should be re-instantiated each cycle

    Note:
        The iterable must be restartable via `iter(iterable)`. For `DataLoader`, this is true.

    Args:
        iterable: Any restartable iterable.

    Yields:
        Items from `iterable` in an infinite loop.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:  # noqa: PERF203
            iterator = iter(iterable)


class PytorchTrainer:
    """A small, subclass-friendly trainer for PyTorch.

    `PytorchTrainer` orchestrates a standard training loop:

    - move model to a target device
    - iterate batches
    - forward pass + loss computation
    - (optional) automatic mixed precision via `torch.autocast` + `torch.GradScaler`
    - backward / optimizer step / (optional) scheduler step
    - update and aggregate metrics through `metric.MetricsHandler`
    - log scalars through a `logging.TrainLogger`
    - save / load checkpoints

    Typical usage
    -------------
    >>> train_loader, val_loader, test_loader = ...  # Define the dataset loading and splitting
    >>> model, optimizer, scheduler = ...  # Define the model and optimizer
    >>> metrics = metric.MetricsHandler([metric.Accuracy(evaluate=True)])  # Add some metrics
    >>> trainer = PytorchTrainer(model, optimizer, scheduler, metrics, save_mode="small")
    >>> trainer.train(epochs=10, train_loader, criterion=torch.nn.CrossEntropyLoss(), val_loader=val_loader)
    >>> scores = trainer.evaluate(test_loader)

    Customization points
    --------------------
    Override one of the following for custom behavior:

    - `process_train_batch` / `process_eval_batch`:
        adapt dataloader batch structure (e.g. dict batches, multiple inputs).
    - `train_step` / `eval_step`:
        implement non-standard forward passes, multi-loss, teacher forcing, etc.
    - `backward`:
        implement gradient accumulation, custom clipping, manual AMP, etc.

    Checkpointing
    -------------
    `save()` writes a dict to `{output_dir}/checkpoints/{filename}` containing model/optimizer state,
    and optionally scheduler + scaler state.

    `save_mode`:
        - "never": do not write checkpoints
        - "small": keep `{epoch}.ckpt` for latest epoch and `best.ckpt`
        - "all": keep all epoch checkpoints + `best.ckpt`
    """

    train_bar_name = "Training"
    eval_bar_name = "Testing "
    train_avg_name = "Avg Train Metrics"
    eval_avg_name = "Avg Eval Metrics "

    def __init__(  # noqa: PLR0913
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics_handler: metric.MetricsHandler | None = None,
        device: torch.device | None = None,
        logger: dt_logging.TrainLogger | None = None,
        output_dir: str | os.PathLike = "./experiments",
        save_mode: str = "never",
        use_amp: bool = False,
    ):
        """Create a Trainer for the training loop orchestration.

        Notes:
            - If `device` is None, the trainer selects CUDA if available, otherwise CPU.
            - The model is sent to the device (multi-device training is not supported yet)
            - The scheduler (when provided) is assumed to be stepped **per training step**
            (see `backward`), not per epoch.
            - Checkpoints are written under `{output_dir}/checkpoints/`.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): Optional learning rate scheduler.
                If provided, `scheduler.step()` is called after successful optimizer updates
                (i.e. when AMP does not skip the step). Note that it is called every training step by default
                and not every epoch.
            metrics_handler (metric.MetricsHandler): Handles metrics update/aggregation in train/eval loops.
                If None, an empty `MetricsHandler([])` is used.
                For training, an additional criterion should be given to compute a differentiable loss.
                To select the best validated model, the validation metric of the handler is used. If it is
                undefined the train criterion will be used instead.
            device (torch.device): Device on which to run the model and move batches. Defaults to CUDA if available.
                Only cpu and cuda devices are supported.
            logger (TrainLogger): Logger used to record scalar metrics. By default, a TensorBoard logger is created.
            output_dir (str | os.PathLike): Experiment directory where logs and checkpoints are stored.
            save_mode (str): Checkpoint retention policy:
                - "never": never write checkpoints
                - "small": keep `best.ckpt` and the last epoch checkpoint `{epoch}.ckpt`
                - "all": keep all epoch checkpoints `{epoch}.ckpt` plus `best.ckpt`
            use_amp (bool): Whether to enable Automatic Mixed Precision (AMP) with `torch.amp`
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_handler = metrics_handler or metric.MetricsHandler([])
        self.output_dir = pathlib.Path(output_dir)
        self.save_mode = save_mode

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model.to(self.device)
        self.scaler = torch.GradScaler(self.device.type, enabled=use_amp)

        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        self.logger: dt_logging.TrainLogger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = dt_logging.TensorBoardLogger(str(self.output_dir / "logs"))  # TODO: Support for Path?

        self.train_steps = 0
        self.epoch = 0
        self.val = float("nan")
        self.best_val = float("nan")
        self.best_epoch = -1

    @property
    def use_amp(self) -> bool:
        """Whether the trainer is using amp or not."""
        return self.scaler.is_enabled()

    def _default_process_batch(self, batch) -> tuple:
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return inputs, targets

    def process_train_batch(self, batch) -> tuple:
        """Prepare a batch for the training step.

        The default implementation expects `batch == (inputs, targets)` and moves both
        tensors to `self.device`.

        Override if your DataLoader yields:
        - dictionaries (e.g. HuggingFace-style)
        - multiple input tensors
        - additional metadata

        Args:
            batch (Any): A single batch produced by `train_loader`.

        Returns:
            tuple: A `(inputs, targets)` pair consumable by `self.model(inputs)` and `criterion(preds, targets)`.
        """
        return self._default_process_batch(batch)

    def process_eval_batch(self, batch) -> tuple:
        """Prepare a batch for the training step.

        The default implementation expects `batch == (inputs, targets)` and moves both
        tensors to `self.device`.

        Override if your DataLoader yields:
        - dictionaries (e.g. HuggingFace-style)
        - multiple input tensors
        - additional metadata

        Args:
            batch (Any): A single batch produced by `val_loader`.

        Returns:
            tuple: A `(inputs, targets)` pair consumable by `self.model(inputs)` and `criterion(preds, targets)`.
        """
        return self._default_process_batch(batch)

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward + optimizer step (+ optional scheduler step).

        Default behavior:
        - `optimizer.zero_grad()`
        - scaled backward if AMP is enabled (`GradScaler.scale(loss).backward()`)
        - `scaler.step(optimizer)` + `scaler.update()`
        - if the step was *not skipped* due to inf/nan gradients, call `scheduler.step()` (if provided)

        Notes:
            - This trainer assumes a *per-step* scheduler (stepped every batch).
              If you want epoch-based scheduling, override this method.
            - For gradient accumulation, clipping, or multiple optimizers, override this method.

        Args:
            loss: Scalar loss tensor for the current batch.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # XXX: torch.GradScaler does not provide any way to check whether the step is skipped
        # We could check some internal stuff like _found_inf_per_device but let's just check whether
        # scale has halved after the scaler update.
        previous_scale = self.scaler.get_scale()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if previous_scale <= self.scaler.get_scale() and self.scheduler:  # Step not skipped
            self.scheduler.step()

    def train_step(self, batch, criterion: Callable[..., torch.Tensor]) -> dict[str, float]:
        """Run one optimization step on a single batch.

        Default implementation performs:
            1) `inputs, targets = process_train_batch(batch)`
            2) forward pass: `predictions = model(inputs)` (under autocast if AMP enabled)
            3) compute loss: `loss = criterion(predictions, targets)`
            4) update metrics: `metrics_handler.update((inputs, targets), predictions.detach())`
            5) backward/step via `self.backward(loss)` (optimizer + scaler + optional scheduler)
            6) return a dict containing:
                - all metric last values (`metrics_handler.last_values`)
                - plus `"Loss": loss.item()`

        Override this method for customization:
            - specific batch processing
            - multiple losses / auxiliary heads
            - models needing extra inputs (attention masks, metadata, etc.)
            - custom metric update logic
            - manual optimization (multiple optimizers, accumulation)

        Args:
            batch (Any): A batch from the training DataLoader.
            criterion (Callable[..., torch.Tensor]): Differentiable loss function.

        Returns:
            dict[str, float]: Batch-level metric values. Is expected to contain a `"Loss"` key.
                The returned values will be logged.
        """
        inputs, targets = self.process_train_batch(batch)

        with torch.autocast(self.device.type, enabled=self.use_amp):
            predictions = self.model(inputs)
            loss = criterion(predictions, targets)
            self.metrics_handler.update((inputs, targets), predictions.detach())

        self.backward(loss)

        metrics = self.metrics_handler.last_values
        metrics["Loss"] = loss.item()

        return metrics

    def eval_step(self, batch) -> dict[str, float]:
        """Compute metrics on a single evaluation batch.

        Default implementation performs:
            1) `inputs, targets = process_eval_batch(batch)`
            2) forward pass: `predictions = model(inputs)` (under autocast if AMP enabled)
            3) update metrics: `metrics_handler.update((inputs, targets), predictions)`

        Note:
            The outer evaluation loop (`evaluate`) runs this under `torch.no_grad()`.

        Args:
            batch (Any): A batch from the evaluation DataLoader.

        Returns:
            dict[str, float]: Batch-level metric values (`metrics_handler.last_values`).
        """
        inputs, targets = self.process_eval_batch(batch)

        with torch.autocast(self.device.type, enabled=self.use_amp):
            predictions = self.model(inputs)
            self.metrics_handler.update((inputs, targets), predictions)

        return self.metrics_handler.last_values

    def _single_epoch_train(
        self, train_iterator: Iterator, criterion: Callable[..., torch.Tensor], epoch_size: int
    ) -> dict[str, float]:
        """Performs a single epoch training of the `train` method."""
        self.model.train()
        self.metrics_handler.train()
        self.metrics_handler.reset()
        metrics = self.metrics_handler.last_values
        metrics["Loss"] = float("nan")
        cum_metrics: dict[str, float] = {}

        progress_bar = tqdm.trange(epoch_size, file=sys.stdout)
        progress_bar.set_description(build_description(self.train_bar_name, metrics))
        for _ in progress_bar:
            batch = next(train_iterator)

            metrics = self.train_step(batch, criterion)

            for metric_name, metric_value in metrics.items():
                cum_metrics[metric_name] = cum_metrics.get(metric_name, 0) + metric_value
                self.logger.log(f"A_train_batch/{metric_name}", metric_value, self.train_steps)

            for i, group in enumerate(self.optimizer.param_groups):
                self.logger.log(f"Z_other/lr_{i}", group["lr"], self.train_steps)

            self.logger.log("Z_other/scale", self.scaler.get_scale(), self.train_steps)

            progress_bar.set_description(build_description(self.train_bar_name, metrics))
            self.train_steps += 1

        metrics = self.metrics_handler.aggregated_values

        # Add extra metric added by hand by the user
        for metric_name, metric_value in cum_metrics.items():
            if metric_name in metrics:
                continue  # Keep aggregated version
            metrics[metric_name] = metric_value / epoch_size

        return metrics

    def _handle_validation_metrics(self, metrics: dict[str, float]) -> None:
        """Handle validation metrics.

        If metrics are better, update best_val, best_epoch and save
        """
        val_metric = cast("metric.Metric", self.metrics_handler.get_validation_metric())

        self.val = metrics[val_metric.display_name]

        if not math.isnan(self.best_val):
            if val_metric.minimize and self.val >= self.best_val:
                return

            if not val_metric.minimize and self.val <= self.best_val:
                return

        self.best_val = self.val
        self.best_epoch = self.epoch
        if self.save_mode in ("small", "all"):
            self.save("best.ckpt")

    def train(
        self,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        criterion: Callable[..., torch.Tensor],
        val_loader: torch.utils.data.DataLoader | None = None,
        epoch_size: int = 0,
    ) -> PytorchTrainer:
        """Train the model for a number of epochs.

        Validation & "best checkpoint" logic:
        - If `val_loader` is provided and a validation metric is configured in `metrics_handler`,
          that metric is used to decide whether to save `best.ckpt`.
        - If no validation metric is configured, the trainer temporarily injects a `PytorchMetric`
          based on `criterion` under the name "Loss" and uses it as the validation metric.

        Args:
            epochs (int): Number of epochs to run. Must be > 0.
            train_loader (torch.utils.data.DataLoader): DataLoader for training batches.
            criterion (Callable): Callable used to compute a differentiable scalar loss from `(predictions, targets)`.
            val_loader (torch.utils.data.DataLoader | None): Optional DataLoader for validation (done every epoch).
            epoch_size (int): Number of training steps per epoch. If 0, defaults to `len(train_loader)`.

        Returns:
            Trainer: `self` (to allow chaining).
        """
        n_epochs = self.epoch + epochs

        if epoch_size == 0:
            epoch_size = len(train_loader)

        # If no validation metric, let's create one from the train criterion
        no_validation_metric = self.metrics_handler.get_validation_metric() is None
        if no_validation_metric:
            self.metrics_handler.metrics.insert(0, metric.PytorchMetric(criterion, "Loss", False, True, True))
            self.metrics_handler.set_validation_metric(0)

        train_iterator = cyclic_iterator(train_loader)
        while self.epoch < n_epochs:
            print(f"Epoch {self.epoch + 1}/{n_epochs}")  # noqa: T201
            metrics = self._single_epoch_train(train_iterator, criterion, epoch_size)
            self.epoch += 1

            for metric_name, metric_value in metrics.items():
                self.logger.log(f"A_train_aggregate/{metric_name}", metric_value, self.train_steps)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

                for metric_name, metric_value in val_metrics.items():
                    self.logger.log(f"B_val/{metric_name}", metric_value, self.train_steps)

                self._handle_validation_metrics(val_metrics)

                print(build_description(self.train_avg_name.format(self.epoch, n_epochs), metrics), flush=True)  # noqa: T201
                print(build_description(self.eval_avg_name.format(self.epoch, n_epochs), val_metrics), flush=True)  # noqa: T201
            else:
                print(build_description(self.train_avg_name.format(self.epoch, n_epochs), metrics), flush=True)  # noqa: T201

            print(flush=True)  # Let's jump a line  # noqa: T201

            if self.save_mode in ("small", "all"):
                self.save(f"{self.epoch}.ckpt")

                if self.save_mode == "small":
                    self._clean_checkpoints()

        if no_validation_metric:
            self.metrics_handler.metrics.pop(0)
            self.metrics_handler._validation_metric = None  # noqa: SLF001

        return self

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        """Evaluate the model on a dataset and return aggregated metrics.

        Extra metrics:
            The trainer also averages any values returned by `eval_step` that are *not*
            part of `metrics_handler.aggregated_values` (useful if a subclass adds custom
            per-batch metrics directly in `eval_step`).

        Args:
            dataloader (torch.utils.data.Dataloader): DataLoader yielding evaluation batches.

        Returns:
            Dict[str, float]: Aggregated metrics
        """
        self.model.eval()
        self.metrics_handler.eval()
        self.metrics_handler.reset()
        metrics = self.metrics_handler.last_values
        cum_metrics: dict[str, float] = {}

        with torch.no_grad():
            progress_bar = tqdm.tqdm(dataloader, file=sys.stdout)
            progress_bar.set_description(build_description(self.eval_bar_name, metrics))
            for batch in progress_bar:
                metrics = self.eval_step(batch)

                for metric_name, metric_value in metrics.items():
                    cum_metrics[metric_name] = cum_metrics.get(metric_name, 0) + metric_value

                progress_bar.set_description(build_description(self.eval_bar_name, metrics))

        metrics = self.metrics_handler.aggregated_values

        # Add extra metric added by hand by the user
        for metric_name, metric_value in cum_metrics.items():
            if metric_name in metrics:
                continue  # Keep aggregated version
            metrics[metric_name] = metric_value / len(dataloader)

        return metrics

    def _clean_checkpoints(self) -> None:
        """Delete all the checkpoints except the last one and 'best.ckpt'."""
        for checkpoint_file in (self.output_dir / "checkpoints").iterdir():
            if checkpoint_file.name[-5:] != ".ckpt":
                continue
            if checkpoint_file.name in {"best.ckpt", f"{self.epoch}.ckpt"}:
                continue
            checkpoint_file.unlink()

    def save(self, filename: str) -> None:
        """Save a checkpoint to disk.

        The checkpoint is saved as a `torch.save`-able dictionary at:

            `{output_dir}/checkpoints/{filename}`

        Format:
            {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "scheduler": scheduler_state_dict,   # only if `self.scheduler` is not None
                "scaler": scaler_state_dict,         # only if AMP enabled (`self.use_amp`)
                "epoch": int,                        # current epoch counter
                "step": int,                         # global training step counter
                "val_metric": float,                 # last tracked validation metric value (`self.val`)
            }

        Note:
            This stores the *current* validation metric value, not necessarily the best one.
            (`best_val` / `best_epoch` are not included.)

        Args:
            filename (str): Checkpoint file name (e.g. `"best.ckpt"`, `"10.ckpt"`).
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "step": self.train_steps,
            "val_metric": self.val,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, self.output_dir / "checkpoints" / filename)

    def load(self, path: str | os.PathLike, strict: bool = False, restore_optim_hp: bool = True) -> None:
        """Restore trainer state from a checkpoint.

        Loads (when present) the following from `path`:
        - model weights
        - optimizer state (and optionally its hyperparameters)
        - scheduler state (if `self.scheduler` exists and `restore_optim_hp=True`)
        - AMP scaler state (only if AMP is already enabled)
        - epoch counter, global step counter, and last validation metric value

        After loading `self.best_epoch` and `self.best_val` are set to the loaded `epoch` / `val_metric`
        (i.e. the checkpoint becomes the new "best so far" baseline)

        Args:
            path (str | os.PathLike): Path to a checkpoint produced by `save()`.
            strict (bool): Controls missing component behavior.
                For model weights: passed to `model.load_state_dict(..., strict=strict)`.
                For scaler/scheduler: if missing and `strict=True`, raise `ValueError`,
                otherwise emit a warning and keep the current state.
            restore_optim_hp: Whether to restore optimizer (and scheduler) hyperparameters
                (learning rate, weight decay, momentum, etc.).
                By default (True), it restores everything to restart the training as originally planned.
                If False, it only restores the optimizer states (e.g. momentum buffers) and keep the current
                set of hyper parameters. In particular, the scheduler is not restored, and another can be
                used instead.
        """
        state: dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict)

        if self.use_amp:
            if "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
            else:
                if strict:
                    raise ValueError("Missing scaler state dict")
                warnings.warn("Missing scaler state dict. Keeping the current state", stacklevel=2)

        # XXX: Investigate loading first scheduler, then optimizer as stated in the new documentation.
        if restore_optim_hp:
            self.optimizer.load_state_dict(state["optimizer"])
        else:
            hyper_parameters = self.optimizer.state_dict()["param_groups"]
            self.optimizer.load_state_dict({"state": state["optimizer"]["state"], "param_groups": hyper_parameters})

        if self.scheduler and restore_optim_hp:
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            else:
                if strict:
                    raise ValueError("Missing scheduler state dict")
                warnings.warn("Missing scheduler state dict. Keeping the current state", stacklevel=2)

        # Allowing time shifting if epoch/step is not here
        self.epoch = state.get("epoch", 0)
        self.train_steps = state.get("step", 0)
        self.val = state.get("val_metric", float("nan"))

        # When reloading a training ckpt, best val and epoch are the starting point
        self.best_epoch = self.epoch
        self.best_val = self.val

        # Ensure all the model is on device
        self.model.to(self.device)
