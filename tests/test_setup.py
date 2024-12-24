import torch
import numpy as np
from PIL import Image
import librosa
import transformers
import umap
import plotly

def test_imports():
    """Test if all required packages are properly installed"""
    print("Testing imports...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    test_imports()
    test_gpu() 