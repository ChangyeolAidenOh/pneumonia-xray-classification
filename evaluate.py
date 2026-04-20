"""
Evaluation script: classification report, confusion matrix, AUROC, and AUPRC.
Supports all model variants including Stage 2 hierarchical classifier.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from config import set_seed, get_device, img_size, num_batch
from data.dataset import get_test_loader
from models.resnet18 import resnet18_baseline, resnet18_improved, resnet18_frozen
from models.efficientnet import efficientnet_b0, efficientnet_b1


MODEL_BUILDERS = {
    "resnet18_baseline": lambda nc: resnet18_baseline(nc, grayscale=True),
    "resnet18_improved": lambda nc: resnet18_improved(nc, dropout=0.5, grayscale=True),
    "resnet18_frozen": lambda nc: resnet18_frozen(nc, dropout=0.5, grayscale=True),
    "efficientnet_b0": lambda nc: efficientnet_b0(nc, pretrained=False),
    "efficientnet_b0_tuned": lambda nc: efficientnet_b0(nc, pretrained=False, dropout=0.4),
    "efficientnet_b1": lambda nc: efficientnet_b1(nc, pretrained=False),
}

CHANNEL_MAP = {
    "resnet18_baseline": 1,
    "resnet18_improved": 1,
    "resnet18_frozen": 1,
    "efficientnet_b0": 3,
    "efficientnet_b0_tuned": 3,
    "efficientnet_b1": 3,
}


def evaluate(model, test_loader, device, class_names, output_dir=None):
    """Run evaluation and print metrics."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # AUROC
    try:
        auroc_weighted = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="weighted"
        )
        auroc_macro = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
        print(f"AUROC (weighted): {auroc_weighted:.4f}")
        print(f"AUROC (macro):    {auroc_macro:.4f}")
    except ValueError:
        print("AUROC: could not compute")

    # AUPRC per class
    print("\nAUPRC per class:")
    for i, name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        precision_vals, recall_vals, _ = precision_recall_curve(
            binary_labels, all_probs[:, i]
        )
        auprc = auc(recall_vals, precision_vals)
        print(f"  {name}: {auprc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    num_channels = CHANNEL_MAP[args.model]

    test_loader, class_names = get_test_loader(args.data_root, num_channels=num_channels)
    num_classes = len(class_names)

    model = MODEL_BUILDERS[args.model](num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    evaluate(model, test_loader, device, class_names, args.output_dir)


if __name__ == "__main__":
    main()
