"""Provide the Trainer Class for PyTorch."""

import math
import os
from typing import Any, Callable, Dict, Tuple

import torch
import torch.utils.data
import tqdm  # type: ignore

from . import logging
from . import metric


# TODO: Avg Losses assume that the batches are evenly sized. (was solved before but not anymore)
# TODO: Log instead of print for epochs + time monitoring (Split data time vs Model time ?)


def round_to_n(x: float, n_digits: int) -> float:
    """Round a floating point to n significant digits

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


def build_description(name: str, metrics: Dict[str, float]) -> str:
    """Create a description string from a name and some metrics

    Args:
        name (str): Main name of the description
        metrics (Dict[str, float]): Metrics to print

    Returns:
        str: "{name} --- {metric_name}: {metric_value}, ..., {metric_name}: {metric_value}"
    """
    desc = name

    if metrics:
        desc += " --- "

        for metric_name in metrics:
            desc += f"{metric_name}: {round_to_n(metrics[metric_name], 4):7}, "
        desc = desc[:-2]

    return desc


class PytorchTrainer:
    """Base trainer for pytorch project.

    Wraps all the training procedures for a given model, optimizer and scheduler.
    """

    epoch_description = "{desc} [{step}/{total_step}]\n"

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metrics_handler: metric.MetricsHandler = None,
        device: torch.device = None,
        output_dir: str = "./experiments",
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
                In addition of those metrics, the `train` and `evaluate` method expect a criterion to compute
                a loss. (the training one should be differentiable unless you do your own backpropagation)
            device (torch.device): Torch device to use. If None, the default device will be cuda:0
                (or cpu if cuda is not available)
                TPU or Multi device training is not supported yet.
            output_dir (str): output directory, where checkpoints, logs, [...] are saved
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.metrics_handler = metrics_handler if metrics_handler else metric.MetricsHandler([])
        self.device = device
        self.output_dir = output_dir
        self.save_mode = save_mode

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        self.logger = logging.TensorBoardLogger(os.path.join(self.output_dir, "logs"))

        self.train_steps = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.best_epoch = -1

    @property
    def use_amp(self) -> bool:
        """Whether the trainer is using amp or not"""
        return self.scaler.is_enabled()

    def _default_process_batch(self, batch: Any) -> Tuple[Any, Any]:
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return inputs, targets

    def process_train_batch(self, batch: Any) -> Tuple[Any, Any]:
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

    def process_eval_batch(self, batch: Any) -> Tuple[Any, Any]:
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

    def backward(self, loss: torch.Tensor):
        """Do the backpropagation step

        Called at each training steps with the computed loss, it handles:
          * The backward of the loss (to be scaled or not if use_amp)
          * The scheduler, optimizer and scaler step / update

        Should be overwritten for special behavior. Use the default implementation as an example.
        (For instance you could do gradient accumulation here)

        For more complexe behavior you can avoid calling backward in the trainstep and handle everything yourself.

        Args:
            loss (torch.Tensor)
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # XXX: torch.cuda.amp.GradScaler does not provide any way to check whether the step is skipped
        # We could check some internal stuff like _found_inf_per_device but let's just check whether
        # scale has halved after the scaler update.
        previous_scale = self.scaler.get_scale()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if previous_scale <= self.scaler.get_scale():  # Step not skipped
            if self.scheduler:
                self.scheduler.step()

    def train_step(self, batch: Any, criterion: Callable) -> Dict[str, float]:
        """Perform a training step.

        Should be overwritten for special behaviors. The steps are:
            - Processing the batch from the train dataloader (By default: `PytorchTrainer.process_train_batch`)
            - Compute model outputs (By default: `self.model(inputs)`)
            - Compute loss using the criterion
            - Backward as needed (By default: `PytorchTrainer.backward` which will handle optimizer and scheduler steps)
            - Update metrics (if needed) and returns a dictionnary of metrics with a "Loss" entry

        Args:
            batch (Any): The batch from the train loader
            criterion (Callable): Loss function given to the train method.
                See `PytorchTrainer.train` documentation.

        Returns:
            Dict[str, float]: Evaluation of metrics on this batch
                Should contain a "Loss" entry.
                All the items (one by metric by default), will be logged
        """
        inputs, targets = self.process_train_batch(batch)

        with torch.cuda.amp.autocast(self.use_amp):
            predictions = self.model(inputs)
            loss: torch.Tensor = criterion(predictions, targets)

        self.backward(loss)

        self.metrics_handler.update(predictions.detach(), targets)
        metrics = self.metrics_handler.last_values
        metrics["Loss"] = loss.item()

        return metrics

    def eval_step(self, batch: Any, criterion: Callable) -> Dict[str, float]:
        """Perform an evaluation step.

        Args:
            batch (Any): Batch from the dataloader given to evaluate
            criterion (Callable): Loss function given to the evaluate method.
                See `PytorchTrainer.evaluate` documentation.

        Returns:
            Dict[str, float]: Evaluation of the metrics on this batch
                Should contain a "Loss" entry
        """
        inputs, targets = self.process_eval_batch(batch)

        with torch.cuda.amp.autocast(self.use_amp):
            predictions = self.model(inputs)
            loss: torch.Tensor = criterion(predictions, targets)

        self.metrics_handler.update(predictions, targets)
        metrics = self.metrics_handler.last_values
        metrics["Loss"] = loss.item()

        return metrics

    def train(
        self,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        criterion: Callable,
        val_loader: torch.utils.data.DataLoader = None,
        val_criterion: Callable = None,
        epoch_size: int = 0,
    ) -> "PytorchTrainer":
        """Train the model

        Args:
            epochs (int): Number of epochs to perform (Should be greater than 0)
            train_loader (torch.utils.data.Dataloader): Data loader for the training set
            criterion (Callable): Loss function which will be called with the model outputs and the
                targets for each batch. Should return a singled loss value on which to backpropagate.
                It should be differentiable.
            val_loader (torch.utils.data.Dataloader): Optional validation data loader. If provided the
                model will be evaluate after each epoch with it. If not, no validation is done.
            val_criterion (Callable): Optional validation criterion. Used to compute a validation loss on the
                val_loader. It not given, criterion will be used instead.
            epoch_size (int): The number of training steps for each epochs.
                By default the length of the train_loader is used.

        Returns:
            Trainer: self
        """
        n_epochs = self.epoch + epochs

        if epoch_size == 0:
            epoch_size = len(train_loader)

        if val_criterion is None:
            val_criterion = criterion

        train_iterator = iter(train_loader)
        while self.epoch < n_epochs:
            self.model.train()
            self.metrics_handler.train()
            self.metrics_handler.reset()
            loss_cum = 0.0
            metrics = self.metrics_handler.last_values
            metrics["Loss"] = float("nan")

            progress_bar = tqdm.trange(epoch_size)
            progress_bar.set_description(build_description("Training", metrics))
            for _ in progress_bar:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                metrics = self.train_step(batch, criterion)
                loss_cum += metrics["Loss"]

                for metric_name in metrics:
                    self.logger.log(f"A_train/{metric_name}", metrics[metric_name], self.train_steps)

                for i, group in enumerate(self.optimizer.param_groups):
                    self.logger.log(f"Z_other/lr_{i}", group["lr"], self.train_steps)

                self.logger.log("Z_other/scale", self.scaler.get_scale(), self.train_steps)

                progress_bar.set_description(build_description("Training", metrics))
                self.train_steps += 1

            metrics = self.metrics_handler.aggregated_values
            self.epoch += 1

            if self.save_mode != "never":
                self.save(f"{self.epoch}.ckpt")

                if self.save_mode == "small":
                    self._clean_checkpoints()

            if val_loader is not None:
                metrics = self.evaluate(val_loader, val_criterion)
                for metric_name in metrics:
                    self.logger.log(f"B_val/{metric_name}", metrics[metric_name], self.train_steps)

                if metrics["Loss"] < self.best_val:
                    self.best_val = metrics["Loss"]
                    self.best_epoch = self.epoch
                    if self.save_mode != "never":
                        self.save("best.ckpt")

                metrics["Avg Val Loss"] = metrics.pop("Loss")  # Rename Loss

            # Use val metrics if possible (and only the avg train loss from train loop)
            metrics["Avg Train Loss"] = loss_cum / epoch_size  # Assume that batch are evenly sized
            print(build_description(f"Epoch {self.epoch}/{n_epochs}", metrics), end="\n\n", flush=True)

        return self

    def evaluate(self, dataloader: torch.utils.data.DataLoader, criterion: Callable) -> Dict[str, float]:
        """Evaluate the current model with the metrics on the dataloader

        Args:
            dataloader (torch.utils.data.Dataloader): Dataloader over the dataset to evaluate
            criterion (Callable): Main criterion of the evaluation. Should be minimized.

        Returns:
            Dict[str, float]: Aggregated metrics
        """
        self.model.eval()
        self.metrics_handler.eval()
        self.metrics_handler.reset()
        loss_cum = 0.0
        metrics = self.metrics_handler.last_values
        metrics["Loss"] = float("nan")

        with torch.no_grad():
            progress_bar = tqdm.tqdm(dataloader)
            progress_bar.set_description(build_description("Testing", metrics))
            for batch in progress_bar:
                metrics = self.eval_step(batch, criterion)
                loss_cum += metrics["Loss"]

                progress_bar.set_description(build_description("Testing", metrics))

        metrics = self.metrics_handler.aggregated_values
        metrics["Loss"] = loss_cum / len(dataloader)  # Assume that batch are evenly sized

        return metrics

    def _clean_checkpoints(self):
        """Delete all the checkpoints except the last one and 'best.ckpt'"""
        files = os.listdir(os.path.join(self.output_dir, "checkpoints"))
        for checkpoint_file in filter(lambda file: file[-5:] == ".ckpt", files):
            if checkpoint_file in {"best.ckpt", f"{self.epoch}.ckpt"}:
                continue
            os.remove(os.path.join(self.output_dir, "checkpoints", checkpoint_file))

    def save(self, filename: str):
        """Save a checkpoint with the following format

        {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict, (If any scheduler)
            "scaler": scaler_state_dict, (If use_amp)
            "epoch": epoch,
            "step": step,
        }

        Args:
            filename (str): Name of the checkpoint file
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "step": self.train_steps,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, os.path.join(self.output_dir, "checkpoints", filename))

    def load(self, path: str, strict: bool = True):
        """Reset the trainer to a given checkpoint

        Args:
            path (str): Path to a valid checkpoint
            strict (bool): Allowing partial loading
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler:
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            else:
                if strict:
                    raise ValueError("Missing scheduler state dict")
                print("Missing scheduler state dict. Keeping the current state")

        if self.use_amp:
            if state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])
            else:
                if strict:
                    raise ValueError("Missing scheduler state dict")
                print("Missing scaler state dict. Keeping the current state")

        # Allowing time shifting if epoch/step is not here
        self.epoch = state.get("epoch", 0)
        self.train_steps = state.get("step", 0)

        # Ensure all the model is on device
        self.model.to(self.device)
