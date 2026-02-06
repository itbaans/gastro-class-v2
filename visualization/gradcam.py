"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
Visualizes which regions of an image are important for model predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Trained model
            target_layer: Layer to compute gradients from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)
        
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to focus on positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        """Make the class callable"""
        return self.generate_cam(input_tensor, target_class)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image (H, W, 3) in RGB, values [0, 255]
        heatmap: Heatmap array (H, W), values [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
    
    Returns:
        Image with heatmap overlay (H, W, 3) in RGB
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image_original: np.ndarray,
    target_layer: nn.Module,
    class_names: List[str],
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
    device: torch.device = None
) -> Tuple[np.ndarray, int, float]:
    """
    Generate and visualize Grad-CAM for an image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        image_original: Original image for overlay (H, W, 3) RGB
        target_layer: Target layer for Grad-CAM
        class_names: List of class names
        target_class: Target class (None for predicted class)
        save_path: Path to save visualization
        device: Device to run on
    
    Returns:
        Tuple of (overlayed_image, predicted_class, confidence)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move to device
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_class = predicted.item()
        confidence_score = confidence.item()
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(image_tensor, target_class=target_class if target_class is not None else predicted_class)
    
    # Overlay heatmap
    overlayed = overlay_heatmap(image_original, cam, alpha=0.5)
    
    # Visualization
    if save_path or save_path is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlayed)
        title_class = class_names[target_class] if target_class is not None else class_names[predicted_class]
        axes[2].set_title(
            f'Overlay\nPredicted: {class_names[predicted_class]}\n'
            f'Confidence: {confidence_score:.2%}',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return overlayed, predicted_class, confidence_score


def process_image_for_gradcam(image_path: str, transform) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and process image for Grad-CAM
    
    Args:
        image_path: Path to image file
        transform: Transform to apply to image
    
    Returns:
        Tuple of (transformed_tensor, original_image_array)
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Apply transform
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image_np


def batch_gradcam_analysis(
    model: nn.Module,
    image_paths: List[str],
    transform,
    target_layer: nn.Module,
    class_names: List[str],
    output_dir: str,
    device: torch.device = None
):
    """
    Generate Grad-CAM for a batch of images
    
    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Transform for preprocessing
        target_layer: Target layer for Grad-CAM
        class_names: List of class names
        output_dir: Directory to save outputs
        device: Device to run on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        # Process image
        image_tensor, image_original = process_image_for_gradcam(image_path, transform)
        
        # Generate Grad-CAM
        image_name = Path(image_path).stem
        save_path = output_dir / f'{image_name}_gradcam.png'
        
        try:
            visualize_gradcam(
                model=model,
                image_tensor=image_tensor,
                image_original=image_original,
                target_layer=target_layer,
                class_names=class_names,
                save_path=str(save_path),
                device=device
            )
            print(f"[{i+1}/{len(image_paths)}] Saved: {save_path}")
        except Exception as e:
            print(f"[{i+1}/{len(image_paths)}] Error processing {image_path}: {e}")
    
    print(f"\nGrad-CAM analysis complete. Results saved to {output_dir}")