"""An example of Deep Trainer with Cifar and torchvision."""

import argparse

import torch
import torch.utils.data
import torchvision  # type: ignore[import-untyped]
from torch_resnet import PreActResNet18

from deep_trainer import PytorchTrainer
from deep_trainer.pytorch import metric


def main(checkpoint, epochs, lr, batch_size, weight_decay, device, use_amp, override_optim_hp):  # noqa: PLR0913
    """Train & evaluate a PreActResNet18 on Cifar10."""
    # Data
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=8)

    # Model
    model = PreActResNet18(small_images=True)
    model.set_head(torch.nn.Linear(model.out_planes, len(trainset.classes)))
    model.to(device)

    # optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainset) / batch_size * epochs)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Metrics
    metrics_handler = metric.MetricsHandler(
        [
            metric.Accuracy(),
            metric.BalancedAccuracy(),
            metric.TopK(2),
        ]
    )

    # Trainer
    trainer = PytorchTrainer(model, optimizer, scheduler, metrics_handler, device, save_mode="small", use_amp=use_amp)

    if checkpoint:
        print(f"Reload from {checkpoint}")
        trainer.load(checkpoint, restore_optim_hp=not override_optim_hp)
        print("Compute metrics with the loaded checkpoint", flush=True)
        print(trainer.evaluate(test_loader))

    # Train
    print("Training....")
    trainer.train(epochs, train_loader, criterion, test_loader)

    # Evaluate
    print("Evaluate...")
    metrics = trainer.evaluate(test_loader)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example with Deep Trainer on Cifar")
    parser.add_argument("--e", default=200, type=int, help="epochs")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--amp", action="store_true", help="Use amp")
    parser.add_argument("--checkpoint", help="Checkpoint to restore")
    parser.add_argument(
        "--override-optim-hp",
        action="store_true",
        help="Override optimizer and scheduler hyper parameters from checkpoint",
    )

    args = parser.parse_args()

    print(args)

    main(
        args.checkpoint, args.e, args.lr, args.bs, args.wd, torch.device(args.device), args.amp, args.override_optim_hp
    )
