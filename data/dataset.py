"""
Data loading utilities: transforms, train/val split, WeightedRandomSampler,
and Stage2 custom dataset for hierarchical BACTERIAL/VIRAL classification.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import img_size, num_batch, val_ratio, num_workers, SEED


# ---- Transforms ----

def get_train_transformation(num_channels=3):
    """Augmentation pipeline for training."""
    norm_mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5]
    norm_std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def get_val_test_transformation(num_channels=3):
    """No-augmentation pipeline for validation and test."""
    norm_mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5]
    norm_std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


# ---- Weighted sampler ----

def build_weighted_sampler(labels):
    """Create WeightedRandomSampler to handle class imbalance."""
    class_sample_counts = np.bincount(labels)
    class_weights_for_sampler = np.zeros_like(class_sample_counts, dtype=float)
    non_zero_counts_mask = class_sample_counts > 0
    class_weights_for_sampler[non_zero_counts_mask] = 1.0 / class_sample_counts[non_zero_counts_mask]

    samples_weights = np.array([class_weights_for_sampler[t] for t in labels])
    samples_weights = torch.from_numpy(samples_weights).double()
    return WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
    )


# ---- Data loaders (3-class) ----

def get_train_val_loaders(data_root, num_channels=3, use_sampler=True):
    """
    Returns (train_batches_dl, val_batches_dl, class_names).
    Stratified 80/20 split with optional WeightedRandomSampler.
    """
    val_test_transformation = get_val_test_transformation(num_channels)
    train_transformation = get_train_transformation(num_channels)

    base_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=val_test_transformation
    )
    class_names = base_dataset.classes
    targets = base_dataset.targets

    train_idx, val_idx = train_test_split(
        list(range(len(targets))),
        test_size=val_ratio,
        stratify=targets,
        random_state=SEED,
    )

    val_data = Subset(
        datasets.ImageFolder(os.path.join(data_root, "train"),
                             transform=val_test_transformation),
        val_idx,
    )
    train_data = Subset(
        datasets.ImageFolder(os.path.join(data_root, "train"),
                             transform=train_transformation),
        train_idx,
    )

    if use_sampler:
        train_subset_labels = [targets[i] for i in train_idx]
        sampler = build_weighted_sampler(train_subset_labels)
        train_batches_dl = DataLoader(train_data, batch_size=num_batch,
                                      sampler=sampler, num_workers=num_workers)
    else:
        train_batches_dl = DataLoader(train_data, batch_size=num_batch,
                                      shuffle=True, num_workers=num_workers)

    val_batches_dl = DataLoader(val_data, batch_size=num_batch,
                                shuffle=False, num_workers=num_workers)

    return train_batches_dl, val_batches_dl, class_names


def get_test_loader(data_root, num_channels=3):
    """Returns (test_loader, class_names)."""
    val_test_transformation = get_val_test_transformation(num_channels)
    test_data = datasets.ImageFolder(
        os.path.join(data_root, "test"), transform=val_test_transformation
    )
    test_loader = DataLoader(test_data, batch_size=num_batch,
                             shuffle=False, num_workers=num_workers)
    return test_loader, test_data.classes


# ---- Stage 2 dataset (BACTERIAL vs VIRAL only) ----

class Stage2_BV_Dataset(Dataset):
    """
    Filters the training set to only BACTERIAL and VIRAL samples,
    re-labeling them as 0 and 1 for binary fine-tuning.
    """

    new_classes = ["BACTERIAL_stage2", "VIRAL_stage2"]

    def __init__(self, root_dir, original_class_names, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        temp_img_folder = datasets.ImageFolder(os.path.join(self.root_dir, "train"))

        bacterial_original_idx = original_class_names.index("BACTERIAL")
        viral_original_idx = original_class_names.index("VIRAL")

        for img_path, original_label_idx in temp_img_folder.samples:
            if original_label_idx == bacterial_original_idx:
                self.samples.append((img_path, 0))  # BACTERIAL -> 0
            elif original_label_idx == viral_original_idx:
                self.samples.append((img_path, 1))  # VIRAL -> 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_targets_for_split(self):
        return [s[1] for s in self.samples]


def get_stage2_loaders(data_root, original_class_names):
    """
    Returns (train_batches_dl, val_batches_dl, num_classes) for Stage 2
    (BACTERIAL vs VIRAL binary classification).
    """
    train_transformation = get_train_transformation(num_channels=3)
    val_test_transformation = get_val_test_transformation(num_channels=3)

    full_bv_dataset = Stage2_BV_Dataset(data_root, original_class_names, transform=None)
    stage2_targets = full_bv_dataset.get_targets_for_split()
    num_classes_stage2 = len(Stage2_BV_Dataset.new_classes)

    train_idx_stage2, val_idx_stage2 = train_test_split(
        list(range(len(full_bv_dataset))),
        test_size=val_ratio,
        stratify=stage2_targets,
        random_state=SEED,
    )

    train_data_stage2 = Subset(
        Stage2_BV_Dataset(data_root, original_class_names, transform=train_transformation),
        train_idx_stage2,
    )
    val_data_stage2 = Subset(
        Stage2_BV_Dataset(data_root, original_class_names, transform=val_test_transformation),
        val_idx_stage2,
    )

    train_stage2_labels = [stage2_targets[i] for i in train_idx_stage2]
    sampler_stage2 = build_weighted_sampler(train_stage2_labels)

    train_batches_dl = DataLoader(train_data_stage2, batch_size=num_batch,
                                  sampler=sampler_stage2, num_workers=num_workers)
    val_batches_dl = DataLoader(val_data_stage2, batch_size=num_batch,
                                shuffle=False, num_workers=num_workers)

    return train_batches_dl, val_batches_dl, num_classes_stage2
