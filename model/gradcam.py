"""
Grad-CAM implementation for explaining CNN predictions.

Works with any ResNet-style model. Hooks into the last convolutional
layer to compute class-discriminative localization maps.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: torch.nn.Module in eval mode
        target_layer: the conv layer to hook (e.g. model.layer4[-1] for ResNet)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1, C, H, W)
        class_idx: which class to explain; if None, uses the predicted class
        Returns: (heatmap as HxW numpy array in [0,1], predicted class idx, confidence)
        """
        self.model.eval()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx, confidence


def overlay_heatmap(original_img_bgr, cam, alpha=0.45):
    """
    original_img_bgr: HxWx3 numpy array (BGR, as read by cv2)
    cam: HxW numpy array in [0,1] (from GradCAM.generate)
    Returns: HxWx3 numpy array with heatmap overlaid
    """
    h, w = original_img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, original_img_bgr, 1 - alpha, 0)
    return overlay


def estimate_region(cam, threshold=0.7):
    """
    Rough textual description of where the model focused, based on the
    centroid of the highest-activation region. Used to give the LLM
    something concrete to describe in the generated report.
    """
    h, w = cam.shape
    mask = cam >= threshold
    if not mask.any():
        return "diffuse region without a sharply localized focus"

    ys, xs = np.where(mask)
    cy, cx = ys.mean() / h, xs.mean() / w

    vertical = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "middle")
    horizontal = "left" if cx < 0.4 else ("right" if cx > 0.6 else "central")

    return f"{vertical} {horizontal} lung field"
