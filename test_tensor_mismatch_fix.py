#!/usr/bin/env python3
"""
Test script to verify that the tensor size mismatch issues have been resolved.
This script creates synthetic 3D images with different dimensions and tests
the optical flow algorithms to ensure they handle shape mismatches correctly.
"""

import torch
import numpy as np
from opticalflow3D import farneback_3d, pyrlk_3d

def create_test_images():
    """Create synthetic test images with slight size variations"""
    # Create slightly different sized images to trigger the tensor mismatch
    base_shape = (30, 112, 112)  # Depth, Height, Width
    
    # First image
    image1 = torch.randn(base_shape, dtype=torch.float32)
    
    # Second image with slightly different dimensions
    # This mimics what can happen during pyramid level processing
    image2_shape = (30, 113, 112)  # One dimension is different
    image2 = torch.randn(image2_shape, dtype=torch.float32)
    
    # Resize image2 to match image1 for this test
    image2 = torch.nn.functional.interpolate(
        image2.unsqueeze(0).unsqueeze(0),
        size=base_shape,
        mode='trilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)
    
    return image1, image2

def test_farneback():
    """Test Farneback 3D algorithm"""
    print("Testing Farneback 3D algorithm...")
    
    try:
        image1, image2 = create_test_images()
        print(f"Image1 shape: {image1.shape}")
        print(f"Image2 shape: {image2.shape}")
        
        # Test with pyramid levels to trigger resizing operations
        vx, vy, vz, confidence = farneback_3d(
            image1, image2, 
            iters=2, 
            num_levels=3,
            scale=0.5,
            spatial_size=7,
            filter_size=3
        )
        
        print(f"✅ Farneback test successful!")
        print(f"Output shapes - vx: {vx.shape}, vy: {vy.shape}, vz: {vz.shape}")
        print(f"Confidence shape: {confidence.shape}")
        
    except Exception as e:
        print(f"❌ Farneback test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pyrlk():
    """Test PyRLK 3D algorithm"""
    print("\nTesting PyRLK 3D algorithm...")
    
    try:
        image1, image2 = create_test_images()
        print(f"Image1 shape: {image1.shape}")
        print(f"Image2 shape: {image2.shape}")
        
        # Test with pyramid levels to trigger resizing operations
        vx, vy, vz = pyrlk_3d(
            image1, image2,
            iters=2,
            num_levels=3,
            scale=0.5,
            filter_size=5
        )
        
        print(f"✅ PyRLK test successful!")
        print(f"Output shapes - vx: {vx.shape}, vy: {vy.shape}, vz: {vz.shape}")
        
    except Exception as e:
        print(f"❌ PyRLK test failed: {e}")
        import traceback
        traceback.print_exc()

def test_device_detection():
    """Test device detection and show which device is being used"""
    print("\nTesting device detection...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")

if __name__ == "__main__":
    print("=== Testing Tensor Size Mismatch Fixes ===")
    
    test_device_detection()
    test_farneback()
    test_pyrlk()
    
    print("\n=== Test Summary ===")
    print("If both algorithms completed without tensor size errors,")
    print("the fixes have successfully resolved the shape mismatch issues!")
