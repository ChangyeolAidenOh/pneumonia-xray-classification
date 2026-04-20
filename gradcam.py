"""
Grad-CAM visualization for chest X-ray classification models.

Generates class activation heatmaps to show which regions of the X-ray
the model focuses on when making predictions. Validates that the model
attends to clinically relevant areas (lung fields) rather than artifacts.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import set_seed, get_device, img_size


class GradCAM:
    """
    Grad-CAM extractor for CNN models.

    Hooks into a target convolutional layer to capture activations
    and gradients, then computes a weighted activation map highlighting
    regions most relevant to the predicted class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, C, H, W)
            target_class: class index to visualize (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) normalized to [0, 1]
            predicted_class: int
            confidence: float
        """
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients -> channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size and normalize
        cam = F.interpolate(cam, size=(img_size, img_size),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, confidence


def get_target_layer(model, model_name):
    """Return the last convolutional layer for Grad-CAM."""
    if "resnet" in model_name:
        return model.layer4[-1].conv2
    elif "efficientnet" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_and_preprocess(image_path, num_channels=3):
    """Load a single image and return (original_image, input_tensor)."""
    original = Image.open(image_path).convert("RGB")

    norm_mean = [0.485, 0.456, 0.406] if num_channels == 3 else [0.5]
    norm_std = [0.229, 0.224, 0.225] if num_channels == 3 else [0.5]

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    tensor = preprocess(original).unsqueeze(0)
    return original, tensor


def visualize_gradcam(original_image, heatmap, predicted_class,
                      confidence, class_names, save_path=None):
    """
    Overlay Grad-CAM heatmap on original X-ray image.
    Layout: original | heatmap | overlay with prediction.
    """
    original_resized = original_image.resize((img_size, img_size))
    original_array = np.array(original_resized)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Original image
    axes[0].imshow(original_array, cmap="gray" if original_array.ndim == 2 else None)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    if original_array.ndim == 3:
        gray = np.mean(original_array, axis=2)
    else:
        gray = original_array
    gray_normalized = gray.astype(float) / 255.0

    axes[2].imshow(gray_normalized, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title(
        f"Prediction: {class_names[predicted_class]} ({confidence:.1%})"
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def visualize_batch(image_dir, model, model_name, device, class_names,
                    num_channels=3, num_images=8, save_dir=None):
    """
    Generate Grad-CAM for multiple images from each class.
    Creates a grid showing model attention patterns across classes.
    """
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)

    num_cols = num_images // len(class_names) + 1
    fig, axes = plt.subplots(len(class_names), num_cols,
                             figsize=(16, 4 * len(class_names)))

    for row, cls_name in enumerate(class_names):
        cls_dir = os.path.join(image_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        images = sorted(os.listdir(cls_dir))[:num_cols]

        for col, fname in enumerate(images):
            if col >= axes.shape[1]:
                break
            img_path = os.path.join(cls_dir, fname)
            original, tensor = load_and_preprocess(img_path, num_channels)
            tensor = tensor.to(device)

            heatmap, pred_cls, conf = gradcam.generate(tensor)

            original_resized = original.resize((img_size, img_size))
            gray = np.array(original_resized.convert("L")) / 255.0

            axes[row, col].imshow(gray, cmap="gray")
            axes[row, col].imshow(heatmap, cmap="jet", alpha=0.4)

            pred_label = class_names[pred_cls]
            color = "green" if pred_label == cls_name else "red"
            axes[row, col].set_title(f"{pred_label} ({conf:.0%})",
                                     fontsize=9, color=color)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(f"True: {cls_name}", fontsize=11, fontweight="bold")

    plt.suptitle("Grad-CAM Attention Maps by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "gradcam_grid.png"), dpi=150)
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model", required=True,
                        choices=["resnet18_baseline", "resnet18_improved",
                                 "resnet18_frozen", "efficientnet_b0",
                                 "efficientnet_b0_tuned", "efficientnet_b1"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image_dir", default=None,
                        help="Directory with class subdirectories for batch mode")
    parser.add_argument("--output_dir", default="./results/gradcam")
    args = parser.parse_args()

    set_seed()
    device = get_device()

    num_channels = 1 if "resnet" in args.model else 3

    from evaluate import MODEL_BUILDERS
    class_names = ["BACTERIAL", "NORMAL", "VIRAL"]
    model = MODEL_BUILDERS[args.model](len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    if args.image:
        target_layer = get_target_layer(model, args.model)
        gradcam = GradCAM(model, target_layer)

        original, tensor = load_and_preprocess(args.image, num_channels)
        tensor = tensor.to(device)
        heatmap, pred_cls, conf = gradcam.generate(tensor)

        save_path = os.path.join(args.output_dir, "gradcam_single.png")
        visualize_gradcam(original, heatmap, pred_cls, conf,
                          class_names, save_path)

    elif args.image_dir:
        visualize_batch(args.image_dir, model, args.model, device,
                        class_names, num_channels, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
