"""
Unified training script for all model variants.
Supports: ResNet18 (baseline/improved/frozen), EfficientNet (B0/B1/Stage2).
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    set_seed, get_device,
    default_epochs, initial_lr, weight_decay,
    early_stop_patience, early_stop_threshold,
)
from visualize import TrainingHistory, plot_learning_curves
from data.dataset import get_train_val_loaders, get_stage2_loaders
from models.resnet18 import (
    resnet18_baseline, resnet18_improved, resnet18_frozen, get_params_to_update,
)
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b0_stage2


MODEL_REGISTRY = {
    "resnet18_baseline": {
        "build": lambda nc: resnet18_baseline(nc, grayscale=True),
        "channels": 1, "use_sampler": False,
    },
    "resnet18_improved": {
        "build": lambda nc: resnet18_improved(nc, dropout=0.5, grayscale=True),
        "channels": 1, "use_sampler": True,
    },
    "resnet18_frozen": {
        "build": lambda nc: resnet18_frozen(nc, dropout=0.5, grayscale=True),
        "channels": 1, "use_sampler": True, "use_adamw": True,
    },
    "efficientnet_b0": {
        "build": lambda nc: efficientnet_b0(nc, pretrained=True),
        "channels": 3, "use_sampler": True,
    },
    "efficientnet_b0_tuned": {
        "build": lambda nc: efficientnet_b0(nc, pretrained=True, dropout=0.4),
        "channels": 3, "use_sampler": True,
    },
    "efficientnet_b1": {
        "build": lambda nc: efficientnet_b1(nc, pretrained=True),
        "channels": 3, "use_sampler": True,
    },
}


def train_one_epoch(model, loader, loss_crit, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_crit(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (preds == labels).sum().item()

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    return epoch_train_loss, epoch_train_acc


def validate(model, loader, loss_crit, device):
    model.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for inputs_val, labels_val in loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_val = model(inputs_val)
            loss_val = loss_crit(outputs_val, labels_val)

            running_val_loss += loss_val.item() * inputs_val.size(0)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val
    return epoch_val_loss, epoch_val_acc


def train(model, train_batches_dl, val_batches_dl, loss_crit, optimizer,
          scheduler, device, num_epochs, save_dir, model_name):
    """Training loop with early stopping, checkpointing, and history logging."""

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    history = TrainingHistory(model_name)

    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    last_model_path = os.path.join(save_dir, f"{model_name}_last.pth")

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = train_one_epoch(
            model, train_batches_dl, loss_crit, optimizer, device
        )
        epoch_val_loss, epoch_val_acc = validate(
            model, val_batches_dl, loss_crit, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history.record(epoch + 1, epoch_train_loss, epoch_train_acc,
                       epoch_val_loss, epoch_val_acc, current_lr)

        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc

        print(
            f"Epoch {epoch+1}/{num_epochs} || "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} || "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} || "
            f"LR: {current_lr:.1e}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss - early_stop_threshold:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), last_model_path)

    # Save training history and learning curves
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    history.save(history_path)
    plot_learning_curves(
        history, os.path.join(save_dir, f"{model_name}_curves.png")
    )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia classification models")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["stage2"])
    parser.add_argument("--data_root", required=True,
                        help="Path to chest_xray_3class directory")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--lr", type=float, default=initial_lr)
    parser.add_argument("--wd", type=float, default=weight_decay)
    parser.add_argument("--stage1_ckpt", default=None,
                        help="Stage 1 checkpoint path (required for stage2)")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"{device}")

    # Stage 2: hierarchical BACTERIAL/VIRAL classifier
    if args.model == "stage2":
        if not args.stage1_ckpt:
            raise ValueError("--stage1_ckpt required for stage2 training")

        from torchvision import datasets
        original_class_names = datasets.ImageFolder(
            os.path.join(args.data_root, "train")
        ).classes

        train_batches_dl, val_batches_dl, num_classes = get_stage2_loaders(
            args.data_root, original_class_names
        )
        model = efficientnet_b0_stage2(
            args.stage1_ckpt, num_classes_stage2=num_classes, device=device
        )
        model = model.to(device)
        loss_crit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=2
        )

        train(model, train_batches_dl, val_batches_dl, loss_crit, optimizer,
              scheduler, device, args.epochs, args.save_dir, "stage2_bv")
        return

    # Standard 3-class training
    cfg = MODEL_REGISTRY[args.model]
    num_channels = cfg["channels"]
    use_sampler = cfg["use_sampler"]

    train_batches_dl, val_batches_dl, class_names = get_train_val_loaders(
        args.data_root, num_channels=num_channels, use_sampler=use_sampler
    )
    num_classes = len(class_names)
    model = cfg["build"](num_classes)
    model = model.to(device)

    loss_crit = nn.CrossEntropyLoss()

    # Frozen models: only update trainable params with AdamW
    if cfg.get("use_adamw"):
        params_to_update = get_params_to_update(model)
        optimizer = optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    train(model, train_batches_dl, val_batches_dl, loss_crit, optimizer,
          scheduler, device, args.epochs, args.save_dir, args.model)


if __name__ == "__main__":
    main()
