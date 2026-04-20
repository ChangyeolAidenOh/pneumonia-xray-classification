"""
EfficientNet model variants for chest X-ray classification.

B0/B1      : Full fine-tuning with 3-channel input (ImageNet-pretrained)
Stage2 head: 2-class (BACTERIAL/VIRAL) classifier initialized from Stage1 features
"""

import torch
import torch.nn as nn
from torchvision import models


def efficientnet_b0(num_classes=3, pretrained=True, dropout=None):
    """
    EfficientNet-B0 with optional custom dropout.
    Uses 3-channel input with ImageNet normalization.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    classifier_input_dim = model.classifier[1].in_features
    if dropout is not None:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(classifier_input_dim, num_classes),
        )
    else:
        model.classifier[1] = nn.Linear(classifier_input_dim, num_classes)

    return model


def efficientnet_b1(num_classes=3, pretrained=True):
    """
    EfficientNet-B1 via Compound Scaling.
    Larger capacity than B0 with higher resolution and depth.
    """
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b1(weights=weights)

    classifier_input_dim = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_input_dim, num_classes)

    return model


def efficientnet_b0_stage2(stage1_checkpoint, num_classes_stage2=2, device="cpu"):
    """
    Stage 2 hierarchical classifier.
    Loads feature extractor from Stage 1 (3-class) checkpoint,
    replaces head with a 2-class (BACTERIAL vs VIRAL) classifier.
    """
    # Build Stage 2 model shell
    model = models.efficientnet_b0(weights=None)
    classifier_input_dim = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_input_dim, num_classes_stage2)

    # Load Stage 1 model to extract pretrained features
    temp_stage1_model = models.efficientnet_b0(weights=None)
    temp_stage1_model.classifier[1] = nn.Linear(
        temp_stage1_model.classifier[1].in_features, 3  # Stage 1: 3 classes
    )
    stage1_state_dict = torch.load(stage1_checkpoint, map_location=device)
    temp_stage1_model.load_state_dict(stage1_state_dict)

    # Transfer feature backbone from Stage 1
    model.features = temp_stage1_model.features
    return model
