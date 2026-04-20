"""
ResNet18 model variants for chest X-ray classification.

Baseline  : Grayscale 1-ch input, no regularization
Improved  : Dropout 0.5, class-weighted loss, augmentation
Frozen    : Freeze all layers except layer4 + fc (feature extraction)
"""

import torch.nn as nn
from torchvision import models


def resnet18_baseline(num_classes=3, grayscale=True):
    """
    Minimal ResNet18 baseline.
    Grayscale (1-channel) input via modified conv1, direct FC output.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_improved(num_classes=3, dropout=0.5, grayscale=True):
    """
    Improved ResNet18 with dropout regularization.
    Designed to be used with class-weighted CrossEntropyLoss.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    classifier_input_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(classifier_input_dim, num_classes),
    )
    return model


def resnet18_frozen(num_classes=3, dropout=0.5, grayscale=True):
    """
    ResNet18 with frozen early layers (feature extraction mode).
    Only layer4 and classifier head are trainable.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Unfreeze layer4 for fine-tuning deeper features
    for param in model.layer4.parameters():
        param.requires_grad = True

    classifier_input_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(classifier_input_dim, num_classes),
    )
    return model


def get_params_to_update(model):
    """Return list of parameters that require gradients."""
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update
