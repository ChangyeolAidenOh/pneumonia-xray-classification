"""
Training history tracking and visualization.

Logs per-epoch metrics during training and generates:
- Learning curves (train/val loss and accuracy)
- Multi-model comparison charts
- Results summary table
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class TrainingHistory:
    """
    Records per-epoch metrics and serializes to JSON for reproducibility.
    Integrates with train.py's training loop.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.epochs = []
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.learning_rates = []

    def record(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)

    def save(self, path):
        data = {
            "model_name": self.model_name,
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        history = cls(data["model_name"])
        history.epochs = data["epochs"]
        history.train_loss = data["train_loss"]
        history.train_acc = data["train_acc"]
        history.val_loss = data["val_loss"]
        history.val_acc = data["val_acc"]
        history.learning_rates = data["learning_rates"]
        return history

    @property
    def best_val_accuracy(self):
        return max(self.val_acc) if self.val_acc else 0.0

    @property
    def best_val_loss(self):
        return min(self.val_loss) if self.val_loss else float("inf")

    @property
    def best_epoch(self):
        if not self.val_loss:
            return 0
        return self.epochs[np.argmin(self.val_loss)]


def plot_learning_curves(history, save_path=None):
    """
    Plot train/val loss and accuracy curves for a single model.
    Marks the best epoch with a vertical dashed line.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = history.epochs

    # Loss
    ax1.plot(epochs, history.train_loss, "b-", linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, history.val_loss, "r-", linewidth=1.5, label="Val Loss")
    ax1.axvline(x=history.best_epoch, color="gray", linestyle="--",
                alpha=0.6, label=f"Best epoch ({history.best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{history.model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Accuracy
    ax2.plot(epochs, history.train_acc, "b-", linewidth=1.5, label="Train Acc")
    ax2.plot(epochs, history.val_acc, "r-", linewidth=1.5, label="Val Acc")
    ax2.axvline(x=history.best_epoch, color="gray", linestyle="--",
                alpha=0.6, label=f"Best epoch ({history.best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{history.model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_model_comparison(histories, save_path=None):
    """
    Compare validation performance across multiple models.
    Top: val loss/acc curves overlaid. Bottom: bar charts of best metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, len(histories)))

    # Val loss comparison
    for h, c in zip(histories, colors):
        axes[0, 0].plot(h.epochs, h.val_loss, linewidth=1.5,
                        color=c, label=h.model_name)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Val Loss")
    axes[0, 0].set_title("Validation Loss Progression")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Val accuracy comparison
    for h, c in zip(histories, colors):
        axes[0, 1].plot(h.epochs, h.val_acc, linewidth=1.5,
                        color=c, label=h.model_name)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Val Accuracy")
    axes[0, 1].set_title("Validation Accuracy Progression")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Best accuracy bar chart
    names = [h.model_name for h in histories]
    best_accs = [h.best_val_accuracy for h in histories]
    bars = axes[1, 0].barh(names, best_accs, color=colors)
    axes[1, 0].set_xlabel("Best Val Accuracy")
    axes[1, 0].set_title("Best Validation Accuracy by Model")
    axes[1, 0].set_xlim(min(best_accs) - 0.05, 1.0)
    for bar, acc in zip(bars, best_accs):
        axes[1, 0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{acc:.1%}", va="center", fontsize=9)

    # Convergence speed
    best_epochs = [h.best_epoch for h in histories]
    bars2 = axes[1, 1].barh(names, best_epochs, color=colors)
    axes[1, 1].set_xlabel("Epoch of Best Val Loss")
    axes[1, 1].set_title("Convergence Speed")
    for bar, ep in zip(bars2, best_epochs):
        axes[1, 1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(ep), va="center", fontsize=9)

    plt.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def print_summary_table(histories):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 75)
    print(f"{'Model':<28} {'Best Val Acc':>12} {'Best Val Loss':>14} {'Best Epoch':>11}")
    print("-" * 75)
    for h in histories:
        print(f"{h.model_name:<28} {h.best_val_accuracy:>11.1%} "
              f"{h.best_val_loss:>14.4f} {h.best_epoch:>11}")
    print("=" * 75)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize training histories")
    parser.add_argument("--history_dir", required=True,
                        help="Directory containing history JSON files")
    parser.add_argument("--output_dir", default="./results")
    args = parser.parse_args()

    histories = []
    for fname in sorted(os.listdir(args.history_dir)):
        if fname.endswith(".json"):
            h = TrainingHistory.load(os.path.join(args.history_dir, fname))
            histories.append(h)

    if not histories:
        print("No history files found")
        exit(1)

    for h in histories:
        save_path = os.path.join(args.output_dir, f"curves_{h.model_name}.png")
        plot_learning_curves(h, save_path)

    if len(histories) > 1:
        plot_model_comparison(
            histories,
            os.path.join(args.output_dir, "model_comparison.png"),
        )

    print_summary_table(histories)
