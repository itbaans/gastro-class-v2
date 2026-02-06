"""
Dataset loader for GastroClassTraining
Handles image loading, preprocessing, and train/validation split
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


class GastroDataset(Dataset):
    """
    Custom Dataset for Gastro Classification
    Loads images from class-organized folders
    """
    
    def __init__(self, root_dir: str, transform=None, class_names: Optional[List[str]] = None, nested_classes: bool = False):
        """
        Args:
            root_dir: Root directory containing class folders
            transform: Optional transform to be applied on images
            class_names: Optional list of class names to use (for filtering)
            nested_classes: If True, handles nested structure like datadir/class1/class1/images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.nested_classes = nested_classes
        
        # Get class folders
        if class_names is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        else:
            self.classes = class_names
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            # Handle nested class structure if specified
            if nested_classes:
                # Check if there's a nested directory with the same name
                nested_dir = class_dir / class_name
                if nested_dir.exists() and nested_dir.is_dir():
                    class_dir = nested_dir
            
            # Find all images in the class directory
            image_count = 0
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    image_count += 1
            
            if image_count == 0:
                print(f"Warning: No images found for class '{class_name}' in {class_dir}")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index"""
        return self.idx_to_class[idx]
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1
        return distribution


def get_transforms(image_size: int = 224, augment: bool = True):
    """
    Get data transforms for training and validation
    
    Args:
        image_size: Size to resize images to
        augment: Whether to apply data augmentation (for training)
    
    Returns:
        Transform composition
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
    nested_classes: bool = False
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Directory containing class folders
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        nested_classes: If True, handles nested structure like datadir/class1/class1/images
    
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Create full dataset with validation transforms first to get classes
    temp_dataset = GastroDataset(
        root_dir=data_dir,
        transform=get_transforms(image_size, augment=False),
        nested_classes=nested_classes
    )
    
    class_names = temp_dataset.classes
    
    # Calculate split sizes
    total_size = len(temp_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    torch.manual_seed(seed)
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size]
    )
    
    # Create separate datasets with appropriate transforms
    train_dataset_full = GastroDataset(
        root_dir=data_dir,
        transform=get_transforms(image_size, augment=True),
        nested_classes=nested_classes
    )
    
    val_dataset_full = GastroDataset(
        root_dir=data_dir,
        transform=get_transforms(image_size, augment=False),
        nested_classes=nested_classes
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)
    
    # Create data loaders
    # Only use pin_memory when CUDA is available (avoids warning on CPU)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Class distribution: {temp_dataset.get_class_distribution()}")
    
    return train_loader, val_loader, class_names