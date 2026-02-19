"""Provide the Trainer Class for PyTorch."""

from __future__ import annotations

import math
import pathlib
import sys
import warnings
from typing import TYPE_CHECKING

import torch
import torch.utils.data
import tqdm.auto as tqdm

from . import logging, metric

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Generator, Iterable, Iterator

# TODO: Avg Losses assume that the batches are evenly sized. (was solved before but not anymore)
# TODO: Log instead of print for epochs + time monitoring (Split data time vs Model time ?)
# TODO: Split the class into a abstract class + an implementation for classification?


def round_to_n(x: float, n_digits: int) -> float:
    """Round a floating point to n significant digits.

    Args:
        x (float): Number to round
        n_digits (int): Number of digits to keep

    Returns:
        float: Rounded version of x with n_digits digits
    """
    if not math.isfinite(x) or x == 0:
        return x
    main_digit = math.floor(math.log10(abs(x)))
    return round(x, -main_digit + n_digits - 1)


def build_description(name: str, metrics: dict[str, float]) -> str:
    """Create a description string from a name and some metrics.

    Args:
        name (str): Main name of the description
        metrics (Dict[str, float]): Metrics to print

    Returns:
        str: "{name} --- {metric_name}: {metric_value}, ..., {metric_name}: {metric_value}"
    """
    desc = name

    if metrics:
        desc += " --- "

        for metric_name in sorted(metrics):
            desc += f"{metric_name}: {round_to_n(metrics[metric_name], 4):7}, "
        desc = desc[:-2]

    return desc


def cyclic_iterator(iterable: Iterable) -> Generator:
    """Build a infinite cyclic iterator over an iterable.

    Different from `itertools.cycle`: It does not try to keep the first iteration in memory
    (heavy and we want to keep randomness in dataloaders)

    Args:
        iterable (Iterable): Iterable to wrap
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:  # noqa: PERF203
            iterator = iter(iterable)


class PytorchTrainer:
    """Base trainer for pytorch project.

    Wraps all the training procedures for a given model, optimizer and scheduler.
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
        logger: logging.TrainLogger | None = None,
        output_dir: str | os.PathLike = "./experiments",
        save_mode: str = "never",
        use_amp: bool = False,
    ):
        """Constructor.

        The model is sent to the device even though this should probably be done before

        By default there is a logger that will save all training metrics to tensorboard.
        You can overwrite this by assigning a new logger to `trainer.logger`. (For instance a DictLogger
        that will keep the metrics in Ram so that you can draw your own curves.)

        Args:
            model (torch.nn.Module): The model to be trained
            optimizer (torch.optim.Optimizer): Optimizer to use
            scheduler (torch.optim.lr_scheduler._LRScheduler): Optional learning rate scheduler.
                The `step` method is called at each training step. (More reliable than calling it at each epoch,
                though it can lead to compute epoch-equivalent steps).
                Default: None
            metrics_handler (metric.MetricdHandler): Handle a list of metrics to track.
                See the documentation of `Metric` for more info.
                In order to train, an additional criterion should be given to compute a differentiable loss.
                In order to select the best model on a validation metric, you should select the validation metric
                of the metrics_handler. Otherwise the train criterion will be used.
            device (torch.device): Torch device to use. If None, the default device will be cuda:0
                (or cpu if cuda is not available)
                TPU or Multi device training is not supported yet.
            logger (TrainLogger): Logger object. If not provided a default tensorboard logger will be created.
            output_dir (str | os.PathLike): output directory, where checkpoints, logs, [...] are saved
            save_mode ("never"|"small"|"all"): Checkpointing mode (see `save` and `load`)
                never: No checkpoint for this training
                small: Keep only the best checkpoint 'best.ckpt' and the last checkpoint
                       '{epoch}.ckpt' (automatically clean the previous ones)
                all  : Keep everything. '{epoch}.ckpt' for each epoch and 'best.ckpt'
            use_amp (bool): Whether to use Automatique Mixed Precision. (with `torch.cuda.amp`)
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

        self.logger: logging.TrainLogger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.TensorBoardLogger(str(self.output_dir / "logs"))  # TODO: Support for Path?

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
        """Process each training batch.

        Extract inputs for the model, targets for the loss, and batch size.

        Should be overwritten for special train batches.

        Args:
            batch (Any): The batch from the train_loader.

        Returns:
            inputs (Any): A valid input for the model
            targets (Any): A valid target for the loss
        """
        return self._default_process_batch(batch)

    def process_eval_batch(self, batch) -> tuple:
        """Process each eval batch.

        Extract inputs for the model, targets for the loss, and batch size.

        Should be overwritten for special eval batches.

        Args:
            batch (Any): The batch from the eval_loader.

        Returns:
            inputs (Any): A valid input for the model
            targets (Any): A valid target for the loss
        """
        return self._default_process_batch(batch)

    def backward(self, loss: torch.Tensor) -> None:
        """Do the backpropagation step.

        Called at each training steps with the computed loss, it handles:
          * The backward of the loss (to be scaled or not if use_amp)
          * The scheduler, optimizer and scaler step / update

        Should be overwritten for special behavior. Use the default implementation as an example.
        (For instance you could do gradient accumulation here)

        For more complex behavior you can avoid calling backward in the trainstep and handle everything yourself.

        Args:
            loss (torch.Tensor): Loss for the current batch.
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
        """Perform a training step.

        Should be overwritten for special behaviors. The steps are:
            - Processing the batch from the train dataloader (By default: `PytorchTrainer.process_train_batch`)
            - Compute model outputs (By default: `self.model(inputs)`)
            - Compute loss using the criterion
            - Backward as needed (By default: `PytorchTrainer.backward` which will handle optimizer and scheduler steps)
            - Update metrics (if needed) and return a dictionary of metrics with a "Loss" entry

        Args:
            batch (Any): The batch from the train loader
            criterion (Callable[[Any, Any], torch.Tensor]): Loss function given to the train method.
                See `PytorchTrainer.train` documentation.

        Returns:
            dict[str, float]: Evaluation of metrics on this batch
                Should contain a "Loss" entry.
                All the items (one by metric by default), will be logged
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
        """Perform an evaluation step with the metrics.

        Args:
            batch (Any): Batch from the dataloader given to evaluate

        Returns:
            dict[str, float]: Evaluation of the metrics on this batch
                Should contain a "Loss" entry
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
        val_metric = self.metrics_handler.get_validation_metric()
        if not val_metric:
            return  # FIXME: Warning rather than nothing

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
        """Train the model.

        Note:
            If no validation metric is set in the metrics handler, then the criterion will be used

        Args:
            epochs (int): Number of epochs to perform (Should be greater than 0)
            train_loader (torch.utils.data.Dataloader): Data loader for the training set
            criterion (Callable): Loss function which will be called with the model outputs and the
                targets for each batch. Should return a singled loss value on which to backpropagate.
                It should be differentiable.
            val_loader (torch.utils.data.Dataloader): Optional validation data loader. If provided the
                model will be evaluate after each epoch with it. If not, no validation is done.
            epoch_size (int): The number of training steps for each epochs.
                By default the length of the train_loader is used.

        Returns:
            Trainer: self
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
        """Evaluate the current model with the metrics on the dataloader.

        Args:
            dataloader (torch.utils.data.Dataloader): Dataloader over the dataset to evaluate

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
        """Save a checkpoint with the following format.

        {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict, (If any scheduler)
            "scaler": scaler_state_dict, (If use_amp)
            "epoch": epoch,
            "step": step,
            "val_metric": validation_metric, (self.val)
        }

        Saved at {output_dir}/checkpoints/filename

        Args:
            filename (str): Name of the checkpoint file
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

    def load(self, path: str, strict: bool = False, restore_optim_hp: bool = True) -> None:
        """Reset the trainer to a given checkpoint.

        It will reload model weights, optimizer state and hyper parameters (with scheduler),
        scaler state. Note that the best validation metric is not stored in the checkpoint,
        therefore it use the current validation metric as the best one.

        Args:
            path (str): Path to a valid checkpoint
            strict (bool): Allowing partial loading
                If True, will raise exception when missing keys are found
            restore_optim_hp (bool): Whether to restore optimizer and scheduler hyper parameters.
                Usual hyper parameters are learning rate (with its schedule), weight decay, momentum, etc
                By default restores everything to restart the training as originally planned.
                If False, it only restores the state of the optimizer and keep the current set of
                hyper parameters (For the scheduler, there is nothing to load and you can even use
                another scheduler).

                Warning: With weight decay, it's common to build two param groups (one with wd and one without).
                    When reloading with no weight decay, you should still split the params in these two groups, even
                    if each group has the same hyper parameters.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict)

        if self.use_amp:
            if state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])
            else:
                if strict:
                    raise ValueError("Missing scaler state dict")
                warnings.warn("Missing scaler state dict. Keeping the current state", stacklevel=2)

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
