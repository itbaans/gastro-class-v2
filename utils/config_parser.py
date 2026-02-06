"""
YAML configuration parser for training setup
Loads and validates configuration files with sensible defaults
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigValidator:
    """Validates configuration parameters"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ['data', 'model', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate data config
        if 'root_dir' not in config['data']:
            raise ValueError("data.root_dir is required")
        
        # Validate model config
        valid_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        model_name = config['model'].get('name', 'resnet50')
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {valid_models}")
        
        # Validate training config
        if config['training']['epochs'] <= 0:
            raise ValueError("training.epochs must be positive")
        if config['training']['batch_size'] <= 0:
            raise ValueError("training.batch_size must be positive")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        'data': {
            'root_dir': None,
            'nested_classes': False,
            'image_size': 224,
            'train_split': 0.8,
            'num_workers': 4,
            'seed': 42,
        },
        'augmentation': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation': 20,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'translate': [0.1, 0.1]
        },
        'model': {
            'name': 'resnet50',
            'num_classes': None,  # Will be inferred from data
            'pretrained': True,
            'pretrained_path': None,  # Path to custom pretrained weights (e.g., models/model_name.pth)
            'freeze_features': False,  # Freeze entire feature extractor
            'freeze_layers': 0  # Number of layer groups to freeze (0-4)
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'adam',  # adam or sgd
            'momentum': 0.9,  # Only for SGD
            'scheduler': {
                'type': 'step',  # step, cosine, or None
                'step_size': 10,  # For StepLR
                'gamma': 0.1,  # For StepLR
                'min_lr': 1e-6  # For CosineAnnealingLR (float, not string)
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            }
        },
        'checkpoint': {
            'save_dir': 'checkpoints',
            'save_best': True,
            'save_every': 5,  # Save every N epochs
            'resume_from': None  # Path to checkpoint to resume from
        },
        'logging': {
            'log_dir': 'logs',
            'tensorboard': False,
            'print_freq': 10  # Print every N batches
        }
    }


def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user config with default config
    
    Args:
        default: Default configuration
        user: User-provided configuration
        
    Returns:
        Merged configuration
    """
    merged = default.copy()
    
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary with defaults filled in
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load user config
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    if user_config is None:
        user_config = {}
    
    # Merge with defaults
    config = merge_configs(get_default_config(), user_config)
    
    # Validate
    ConfigValidator.validate_config(config)
    
    # Expand paths
    if config['data']['root_dir']:
        config['data']['root_dir'] = os.path.expanduser(config['data']['root_dir'])
    
    if config['model']['pretrained_path']:
        config['model']['pretrained_path'] = os.path.expanduser(config['model']['pretrained_path'])
    
    if config['checkpoint']['resume_from']:
        config['checkpoint']['resume_from'] = os.path.expanduser(config['checkpoint']['resume_from'])
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
