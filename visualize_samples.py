"""
Generate Grad-CAM visualizations for random samples from each class
Run with: python visualize_samples.py --config configs/visualize_gradcam.yaml
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import numpy as np

# Setup matplotlib for inline display in notebooks
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for figure generation
import matplotlib.pyplot as plt

# Try to import IPython display for better notebook support
try:
    from IPython.display import display, Image as IPImage
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model.resnet import GastroResNet
from visualization.gradcam import visualize_gradcam, process_image_for_gradcam
from data.dataset import get_transforms


def load_model_from_checkpoint(checkpoint_path, model_name, num_classes, device):
    """Load model from checkpoint"""
    # Create model
    from model.resnet import GastroResNet
    model = GastroResNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove DataParallel wrapper if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def get_random_samples_per_class(data_dir, nested_classes=False, samples_per_class=3, seed=42):
    """Get random image paths for each class"""
    data_dir = Path(data_dir)
    random.seed(seed)
    
    # Get class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    samples = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Handle nested structure
        if nested_classes:
            nested_dir = class_dir / class_name
            if nested_dir.exists() and nested_dir.is_dir():
                class_dir = nested_dir
        
        # Get all images in class
        image_extensions = ['.jpg', '.jpeg', '.png']
        images = [
            str(img) for img in class_dir.glob('*')
            if img.suffix.lower() in image_extensions
        ]
        
        # Sample random images
        if images:
            num_samples = min(samples_per_class, len(images))
            selected = random.sample(images, num_samples)
            samples[class_name] = selected
            print(f"Class '{class_name}': Selected {num_samples}/{len(images)} images")
        else:
            print(f"Warning: No images found for class '{class_name}'")
    
    return samples


def colormap_str_to_cv2(colormap_name):
    """Convert colormap string to OpenCV constant"""
    import cv2
    colormap_mapping = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'cool': cv2.COLORMAP_COOL,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'bone': cv2.COLORMAP_BONE,
    }
    return colormap_mapping.get(colormap_name.lower(), cv2.COLORMAP_JET)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("GRAD-CAM VISUALIZATION")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get random samples per class
    print("Selecting random samples...")
    samples_dict = get_random_samples_per_class(
        data_dir=config['data']['root_dir'],
        nested_classes=config['data']['nested_classes'],
        samples_per_class=config['visualization']['samples_per_class'],
        seed=config['visualization']['seed']
    )
    
    class_names = sorted(samples_dict.keys())
    
    # Load model
    print(f"\nLoading model from {config['model']['checkpoint_path']}...")
    
    # Infer num_classes if not specified
    num_classes = config['model']['num_classes']
    if num_classes is None:
        num_classes = len(class_names)
    
    model = load_model_from_checkpoint(
        checkpoint_path=config['model']['checkpoint_path'],
        model_name=config['model']['model_name'],
        num_classes=num_classes,
        device=device
    )
    print(f"Model loaded successfully!")
    
    # Get target layer for Grad-CAM
    target_layer = model.get_layer_for_gradcam()
    
    # Get transform
    transform = get_transforms(
        image_size=config['data']['image_size'],
        augment=False
    )
    
    # Create output directory if saving
    if config['visualization']['save_images']:
        output_dir = Path(config['visualization']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {output_dir}")
    
    # Process each class
    print(f"\n{'='*80}")
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    total_images = sum(len(imgs) for imgs in samples_dict.values())
    processed = 0
    
    for class_name in class_names:
        image_paths = samples_dict[class_name]
        
        print(f"\nClass: {class_name} ({len(image_paths)} samples)")
        print("-" * 80)
        
        for img_path in image_paths:
            processed += 1
            img_name = Path(img_path).name
            
            try:
                # Process image
                image_tensor, image_original = process_image_for_gradcam(img_path, transform)
                
                # Save path (if saving)
                save_path = None
                if config['visualization']['save_images']:
                    save_path = str(output_dir / f"{class_name}_{img_name}_gradcam.png")
                else:
                    # Create temporary file for display even if not saving permanently
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    save_path = temp_file.name
                
                # Generate Grad-CAM
                print(f"  [{processed}/{total_images}] Processing: {img_name}")
                
                overlayed, pred_class, confidence = visualize_gradcam(
                    model=model,
                    image_tensor=image_tensor,
                    image_original=image_original,
                    target_layer=target_layer,
                    class_names=class_names,
                    save_path=save_path,  # Always save to show
                    device=device
                )
                
                print(f"    Predicted: {class_names[pred_class]} (Confidence: {confidence:.2%})")
                
                # Display the saved figure in Kaggle/Jupyter
                if config['visualization']['display_inline']:
                    if IPYTHON_AVAILABLE:
                        # Use IPython display for better rendering - force display
                        from IPython.display import display as ipy_display
                        ipy_display(IPImage(filename=save_path))
                    else:
                        # Fallback: Use PIL to show image
                        from PIL import Image as PILImage
                        img = PILImage.open(save_path)
                        plt.figure(figsize=config['visualization']['figsize'])
                        plt.imshow(img)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        plt.close()
                
                # Clean up temporary file if not saving permanently
                if not config['visualization']['save_images']:
                    import os
                    import time
                    time.sleep(0.1)  # Small delay to ensure display is complete
                    try:
                        os.unlink(save_path)
                    except:
                        pass  # Ignore if file is still in use
                
            except Exception as e:
                print(f"    Error: {e}")
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    
    if config['visualization']['save_images']:
        print(f"Results saved to: {config['visualization']['output_dir']}")
    else:
        print("Visualizations displayed inline (not saved)")


if __name__ == '__main__':
    main()
