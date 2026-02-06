"""
Training loop and utilities for GastroClassTraining
Handles training, validation, checkpointing, and early stopping
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.metrics import MetricsCalculator, MetricsTracker


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """
    Trainer class for model training and validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        class_names: list,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = 'checkpoints',
        save_every: int = 5,
        print_freq: int = 10
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            class_names: List of class names
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            print_freq: Print training stats every N batches
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every = save_every
        self.print_freq = print_freq
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics_calculator = MetricsCalculator(class_names)
        self.metrics_tracker = MetricsTracker()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if (batch_idx + 1) % self.print_freq == 0:
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """
        Validate the model
        
        Returns:
            Tuple of (average loss, accuracy, metrics dict)
        """
        self.model.eval()
        running_loss = 0.0
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Update metrics
                running_loss += loss.item()
                self.metrics_calculator.update(predicted, labels, probs)
        
        # Compute metrics
        epoch_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calculator.compute()
        epoch_acc = metrics['accuracy'] * 100
        
        return epoch_loss, epoch_acc, metrics
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
            filename: Custom filename (if None, uses epoch number)
        """
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch + 1}.pth'
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'class_names': self.class_names
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Best model saved: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Resumed from epoch {self.current_epoch} with best val acc: {self.best_val_acc:.2f}%')
    
    def train(
        self,
        num_epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        resume_from: Optional[str] = None
    ) -> MetricsTracker:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping: Early stopping object (optional)
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            MetricsTracker with training history
        """
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        start_epoch = self.current_epoch
        
        print("\n" + "="*80)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*80 + "\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics tracker
            self.metrics_tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        
        # Save final checkpoint
        self.save_checkpoint(filename='final_model.pth')
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("="*80 + "\n")
        
        return self.metrics_tracker
