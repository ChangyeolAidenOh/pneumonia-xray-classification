import torch
import random
import numpy as np


SEED = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Data
img_size = 224
num_batch = 32
val_ratio = 0.2
num_workers = 2

# Training
default_epochs = 25
initial_lr = 1e-4
weight_decay = 1e-5
early_stop_patience = 5
early_stop_threshold = 1e-4
