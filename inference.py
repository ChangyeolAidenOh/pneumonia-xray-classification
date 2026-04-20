"""
Single-image inference for pneumonia classification.

Loads a trained model and predicts the class of a given chest X-ray image
with confidence scores for all three classes. Optionally generates
a Grad-CAM overlay to visualize model attention.

Usage:
    python inference.py --model efficientnet_b0 \
        --checkpoint ./checkpoints/efficientnet_b0_best.pth \
        --image /path/to/xray.jpeg \
        --gradcam
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import set_seed, get_device, img_size
from gradcam import GradCAM, get_target_layer, load_and_preprocess
from evaluate import MODEL_BUILDERS, CHANNEL_MAP


CLASS_NAMES_3CLASS = ["BACTERIAL", "NORMAL", "VIRAL"]


def predict(model, image_tensor, device, class_names):
    """
    Run inference on a single preprocessed image tensor.

    Returns:
        predicted_class: str
        confidence: float
        all_probs: dict mapping class name to probability
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

    pred_idx = np.argmax(probs)
    all_probs = {name: float(p) for name, p in zip(class_names, probs)}

    return class_names[pred_idx], probs[pred_idx], all_probs


def display_prediction(original_image, predicted_class, confidence,
                       all_probs, heatmap=None, save_path=None):
    """Display prediction results with optional Grad-CAM overlay."""

    num_cols = 3 if heatmap is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5),
                             gridspec_kw={"width_ratios": [1] * num_cols})

    if num_cols == 2:
        ax_img, ax_bar = axes
    else:
        ax_img, ax_cam, ax_bar = axes

    # Original image
    original_resized = original_image.resize((img_size, img_size))
    ax_img.imshow(np.array(original_resized), cmap="gray")
    ax_img.set_title("Input X-ray", fontsize=12)
    ax_img.axis("off")

    # Grad-CAM overlay
    if heatmap is not None:
        gray = np.array(original_resized.convert("L")) / 255.0
        ax_cam.imshow(gray, cmap="gray")
        ax_cam.imshow(heatmap, cmap="jet", alpha=0.4)
        ax_cam.set_title("Grad-CAM Attention", fontsize=12)
        ax_cam.axis("off")

    # Confidence bar chart
    names = list(all_probs.keys())
    probs_vals = list(all_probs.values())
    colors = ["#e74c3c" if n == predicted_class else "#bdc3c7" for n in names]
    bars = ax_bar.barh(names, probs_vals, color=colors)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Confidence")
    ax_bar.set_title(f"Prediction: {predicted_class} ({confidence:.1%})", fontsize=12)

    for bar, p in zip(bars, probs_vals):
        ax_bar.text(bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{p:.1%}", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single-image pneumonia inference")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualization")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    num_channels = CHANNEL_MAP[args.model]
    class_names = CLASS_NAMES_3CLASS

    model = MODEL_BUILDERS[args.model](len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    original, tensor = load_and_preprocess(args.image, num_channels)
    tensor = tensor.to(device)

    pred_class, confidence, all_probs = predict(
        model, tensor, device, class_names
    )

    print(f"Prediction: {pred_class} ({confidence:.1%})")
    for name, prob in all_probs.items():
        print(f"  {name}: {prob:.1%}")

    heatmap = None
    if args.gradcam:
        target_layer = get_target_layer(model, args.model)
        gradcam_extractor = GradCAM(model, target_layer)
        heatmap, _, _ = gradcam_extractor.generate(tensor)

    save_path = args.output or os.path.join("results", "inference_result.png")
    display_prediction(original, pred_class, confidence, all_probs,
                       heatmap=heatmap, save_path=save_path)


if __name__ == "__main__":
    main()
