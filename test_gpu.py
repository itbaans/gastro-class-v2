"""
Quick script to test GPU availability
Run this in Kaggle to verify GPU is detected
"""

import torch

print("="*60)
print("GPU Detection Test")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    # Test tensor on GPU
    print("\nTesting GPU tensor operations...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✓ Successfully performed matrix multiplication on GPU")
    print(f"  Result device: {z.device}")
else:
    print("\n⚠ WARNING: CUDA not available!")
    print("  Check that you're using a GPU runtime in Kaggle")
    print("  Go to: Settings > Accelerator > GPU T4 x2")

print("\n" + "="*60)
