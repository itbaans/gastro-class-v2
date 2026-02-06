"""
Main training script for GastroClassTraining
Run with: python main_train.py --config configs/train_config_example.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import get_data_loaders
from model.resnet import create_model
from train.trainer import Trainer, EarlyStopping
from utils.config_parser import load_config, print_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GastroClassification model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (auto-detect if not specified)'
    )
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print_config(config)
    print("="*80 + "\n")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        train_split=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        seed=config['data']['seed'],
        nested_classes=config['data']['nested_classes']
    )
    
    # Update num_classes in config if not specified
    if config['model']['num_classes'] is None:
        config['model']['num_classes'] = len(class_names)
    
    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}\n")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        custom_pretrained_path=config['model']['pretrained_path'],
        freeze_features=config['model']['freeze_features'],
        freeze_layers=config['model']['freeze_layers'],
        device=device
    )
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    print(f"Optimizer: {config['training']['optimizer'].upper()}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create learning rate scheduler
    scheduler = None
    if config['training']['scheduler']['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler']['step_size'],
            gamma=config['training']['scheduler']['gamma']
        )
        print(f"Scheduler: StepLR (step_size={config['training']['scheduler']['step_size']}, gamma={config['training']['scheduler']['gamma']})")
    elif config['training']['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
        print(f"Scheduler: CosineAnnealingLR (min_lr={config['training']['scheduler']['min_lr']})")
    
    # Create early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        print(f"Early stopping: Enabled (patience={config['training']['early_stopping']['patience']})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=class_names,
        scheduler=scheduler,
        checkpoint_dir=config['checkpoint']['save_dir'],
        save_every=config['checkpoint']['save_every'],
        print_freq=config['logging']['print_freq']
    )
    
    # Train
    metrics_tracker = trainer.train(
        num_epochs=config['training']['epochs'],
        early_stopping=early_stopping,
        resume_from=config['checkpoint']['resume_from']
    )
    
    # Save training history plots
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    history_plot_path = log_dir / 'training_history.png'
    metrics_tracker.plot_history(save_path=str(history_plot_path))
    
    print(f"\nTraining history saved to: {history_plot_path}")
    print(f"Checkpoints saved to: {config['checkpoint']['save_dir']}")
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
