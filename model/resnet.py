"""
ResNet model wrapper for GastroClassTraining
Supports multiple ResNet architectures with custom classifier
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class GastroResNet(nn.Module):
    """
    ResNet wrapper for gastro classification
    Supports ResNet18, 34, 50, 101, 152 with custom classifier head
    """
    
    AVAILABLE_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_layers: int = 0,
        custom_pretrained_path: Optional[str] = None,
        freeze_features: bool = False
    ):
        """
        Args:
            model_name: Name of ResNet architecture ('resnet18', 'resnet34', etc.)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            freeze_layers: Number of layer groups to freeze (0-4)
            custom_pretrained_path: Path to custom pretrained weights (e.g., GastroNet-5M)
            freeze_features: If True, freeze entire feature extractor (all layers except fc)
        """
        super(GastroResNet, self).__init__()
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not available. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load base model
        if custom_pretrained_path:
            # Load custom pretrained weights (e.g., GastroNet-5M)
            print(f"Loading custom pretrained weights from {custom_pretrained_path}")
            self.model = self.AVAILABLE_MODELS[model_name](weights=None)
            self.load_custom_pretrained(custom_pretrained_path)
        elif pretrained:
            weights = 'IMAGENET1K_V1'
            self.model = self.AVAILABLE_MODELS[model_name](weights=weights)
        else:
            self.model = self.AVAILABLE_MODELS[model_name](weights=None)
        
        # Get the number of features from the last layer
        num_features = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Freeze entire feature extractor if specified
        if freeze_features:
            self._freeze_feature_extractor()
        # Otherwise freeze specific layer groups if specified
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_groups: int):
        """
        Freeze initial layer groups
        
        Args:
            num_groups: Number of layer groups to freeze (0-4)
                       0: No freezing
                       1: Freeze conv1, bn1
                       2: Freeze + layer1
                       3: Freeze + layer2
                       4: Freeze + layer3
        """
        layers_to_freeze = []
        
        if num_groups >= 1:
            layers_to_freeze.extend([self.model.conv1, self.model.bn1])
        if num_groups >= 2:
            layers_to_freeze.append(self.model.layer1)
        if num_groups >= 3:
            layers_to_freeze.append(self.model.layer2)
        if num_groups >= 4:
            layers_to_freeze.append(self.model.layer3)
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        print(f"Froze {num_groups} layer group(s)")
    
    def _freeze_feature_extractor(self):
        """
        Freeze all layers except the final classifier
        Used for transfer learning with custom pretrained weights
        """
        for name, param in self.model.named_parameters():
            if 'fc' not in name:  # Freeze everything except fc layer
                param.requires_grad = False
        
        print("Froze entire feature extractor (training only classification head)")
    
    def load_custom_pretrained(self, checkpoint_path: str):
        """
        Load custom pretrained weights (e.g., GastroNet-5M)
        
        Args:
            checkpoint_path: Path to pretrained checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights into model (ignore fc layer as we'll replace it)
        # Create a new state dict without fc layer
        model_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_state_dict and 'fc' not in k}
        
        # Update model state dict
        model_state_dict.update(pretrained_dict)
        self.model.load_state_dict(model_state_dict, strict=False)
        
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} layers from custom pretrained weights")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def get_feature_extractor(self):
        """
        Get the feature extractor (all layers except fc)
        Useful for Grad-CAM
        """
        return nn.Sequential(*list(self.model.children())[:-1])
    
    def get_classifier(self):
        """Get the classifier layer"""
        return self.model.fc
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    def get_layer_for_gradcam(self) -> nn.Module:
        """
        Get the recommended layer for Grad-CAM visualization
        Returns the last convolutional layer (layer4)
        """
        return self.model.layer4
    
    def summary(self):
        """Print model summary"""
        params = self.count_parameters()
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Frozen parameters: {params['frozen']:,}")
        print(f"{'='*60}\n")


def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 6,
    pretrained: bool = True,
    freeze_layers: int = 0,
    custom_pretrained_path: Optional[str] = None,
    freeze_features: bool = False,
    device: Optional[torch.device] = None
) -> GastroResNet:
    """
    Factory function to create and initialize a model
    
    Args:
        model_name: Name of ResNet architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_layers: Number of layers to freeze
        custom_pretrained_path: Path to custom pretrained weights
        freeze_features: Freeze entire feature extractor (only train head)
        device: Device to move model to
    
    Returns:
        Initialized GastroResNet model
    """
    model = GastroResNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_layers=freeze_layers,
        custom_pretrained_path=custom_pretrained_path,
        freeze_features=freeze_features
    )
    
    if device is not None:
        model = model.to(device)
    
    model.summary()
    
    return model