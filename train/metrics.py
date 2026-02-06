"""
Metrics calculation for model evaluation
Computes accuracy, precision, recall, F1-score, and confusion matrix
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate and track evaluation metrics
    """
    
    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: List of class names in order
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and labels"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """
        Update with new predictions and labels
        
        Args:
            preds: Predicted class indices
            labels: True class indices
            probs: Predicted probabilities (optional)
        """
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict:
        """
        Compute all metrics
        
        Returns:
            Dictionary containing all computed metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall metrics
        accuracy = accuracy_score(labels, preds)
        
        # Per-class metrics with zero_division handling
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': {
                self.class_names[i]: precision_per_class[i] 
                for i in range(self.num_classes)
            },
            'recall_per_class': {
                self.class_names[i]: recall_per_class[i] 
                for i in range(self.num_classes)
            },
            'f1_per_class': {
                self.class_names[i]: f1_per_class[i] 
                for i in range(self.num_classes)
            },
            'confusion_matrix': cm
        }
        
        return metrics
    
    def get_classification_report(self) -> str:
        """
        Get sklearn classification report as string
        
        Returns:
            Classification report string
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        return classification_report(
            labels,
            preds,
            target_names=self.class_names,
            zero_division=0
        )
    
    def print_metrics(self, metrics: Optional[Dict] = None):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Metrics dictionary (if None, compute it)
        """
        if metrics is None:
            metrics = self.compute()
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision_macro']:.4f}")
        print(f"Recall:       {metrics['recall_macro']:.4f}")
        print(f"F1-Score:     {metrics['f1_macro']:.4f}")
        print("\nPer-Class Metrics:")
        print("-"*60)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:20s} | "
                  f"P: {metrics['precision_per_class'][class_name]:.4f} | "
                  f"R: {metrics['recall_per_class'][class_name]:.4f} | "
                  f"F1: {metrics['f1_per_class'][class_name]:.4f}")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = False):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot (if None, display only)
            normalize: Whether to normalize the confusion matrix
        """
        metrics = self.compute()
        cm = metrics['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MetricsTracker:
    """
    Track metrics history over epochs
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float
    ):
        """Update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        axes[1, 1].plot(epochs, loss_diff, 'm-', linewidth=2)
        axes[1, 1].set_title('Train-Val Loss Gap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
        
        plt.close()