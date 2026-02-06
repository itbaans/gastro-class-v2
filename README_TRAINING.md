# GastroClassTraining - Training Guide

## Quick Start

### 1. Prepare Your Configuration

Edit one of the example configs in `configs/` or create your own:

```yaml
data:
  root_dir: "path/to/your/dataset"  # Your dataset path
  nested_classes: true  # Set to true if structure is: datadir/class1/class1/images

model:
  name: "resnet50"
  pretrained: true  # Use ImageNet pretrained weights
  pretrained_path: null  # Or specify: models/your_model.pth

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

### 2. Run Training

```bash
# Basic training
python train.py --config configs/train_config_example.yaml

# Fine-tune with pretrained model
python train.py --config configs/finetune_pretrained.yaml

# Quick test run (override epochs)
python train.py --config configs/quick_train.yaml --epochs 5
```

## Dataset Structure

### Standard Structure
```
dataset/
  class1/
    image1.jpg
    image2.jpg
  class2/
    image1.jpg
    image2.jpg
```
Set `nested_classes: false` in config.

### Nested Structure  
```
dataset/
  class1/
    class1/
      image1.jpg
      image2.jpg
  class2/
    class2/
      image1.jpg
      image2.jpg
```
Set `nested_classes: true` in config.

## Configuration Options

### Model Configuration

**Available Models**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

**Pretrained Options**:
- `pretrained: true` - Use ImageNet pretrained weights
- `pretrained_path: models/model_name.pth` - Use custom pretrained weights
- `freeze_features: true` - Freeze backbone, train only classifier
- `freeze_layers: 0-4` - Freeze specific layer groups

### Training Options

**Optimizers**: `adam`, `sgd`

**Learning Rate Schedulers**:
- `step` - Reduce LR every N epochs
- `cosine` - Cosine annealing
- `null` - No scheduler

**Early Stopping**:
```yaml
early_stopping:
  enabled: true
  patience: 10  # Stop after 10 epochs without improvement
  min_delta: 0.001  # Minimum improvement threshold
```

## Example Workflows

### 1. Train from Scratch with ImageNet Weights
```bash
python train.py --config configs/train_config_example.yaml
```

### 2. Fine-tune Custom Pretrained Model
```bash
# Edit configs/finetune_pretrained.yaml:
# - Set pretrained_path: models/gastronet_5m.pth
# - Set freeze_features: true (optional)
python train.py --config configs/finetune_pretrained.yaml
```

### 3. Resume Training from Checkpoint
```bash
# Edit config:
# checkpoint:
#   resume_from: checkpoints/checkpoint_epoch_20.pth
python train.py --config your_config.yaml
```

## Output Files

### Checkpoints
Saved in `checkpoint_dir` (default: `checkpoints/`):
- `best_model.pth` - Best model based on validation accuracy
- `final_model.pth` - Final model after training
- `checkpoint_epoch_N.pth` - Periodic checkpoints

### Logs
Saved in `log_dir` (default: `logs/`):
- `training_history.png` - Training curves (loss, accuracy, LR)

## Command Line Arguments

```bash
python train.py \
  --config configs/your_config.yaml \  # Required
  --epochs 100 \                        # Override config epochs
  --batch-size 64 \                     # Override config batch size
  --device cuda                         # Force device (cuda/cpu)
```

## Tips

1. **Start with a quick test**: Use `configs/quick_train.yaml` with `--epochs 2` to verify setup
2. **Monitor GPU memory**: Reduce batch size if you get OOM errors
3. **Use nested_classes: true**: If your dataset has the `class/class/images` structure
4. **Custom pretrained models**: Place in `models/` directory and reference in config
5. **Best practices for fine-tuning**:
   - Use lower learning rate (e.g., 0.0001)
   - Consider freezing feature extractor initially
   - Use cosine annealing scheduler

## Troubleshooting

### "No images found for class"
- Check `nested_classes` setting in config
- Verify dataset path is correct
- Ensure images are .jpg, .jpeg, or .png

### Out of Memory
- Reduce `batch_size` in config
- Use smaller model (resnet18 instead of resnet50)
- Reduce `num_workers`

### Low accuracy
- Increase training epochs
- Try different learning rates (0.0001 - 0.01)
- Enable data augmentation
- Use pretrained weights
