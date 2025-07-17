#!/usr/bin/env python3
"""
Test script to verify that the OpticalFlow3D PyTorch conversion is working correctly.
"""

import numpy as np
import torch
import sys
import os

# Add the opticalflow3D directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'opticalflow3D'))

try:
    import opticalflow3D
    print("✓ Successfully imported opticalflow3D")
except ImportError as e:
    print(f"✗ Failed to import opticalflow3D: {e}")
    sys.exit(1)

def test_pytorch_availability():
    """Test PyTorch availability and CUDA support."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    print()

def test_farneback_3d():
    """Test Farneback3D class instantiation and basic functionality."""
    print("Testing Farneback3D...")
    
    try:
        # Test with CPU first
        farneback_cpu = opticalflow3D.Farneback3D(
            iters=2,
            num_levels=2,
            scale=0.5,
            spatial_size=5,
            filter_size=9,
            device='cpu'
        )
        print("✓ Farneback3D CPU instantiation successful")
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            farneback_cuda = opticalflow3D.Farneback3D(
                iters=2,
                num_levels=2,
                scale=0.5,
                spatial_size=5,
                filter_size=9,
                device='cuda'
            )
            print("✓ Farneback3D CUDA instantiation successful")
        
        # Test with small synthetic data
        np.random.seed(42)
        img1 = np.random.rand(32, 32, 32).astype(np.float32)
        img2 = np.random.rand(32, 32, 32).astype(np.float32)
        
        print("Testing Farneback3D with synthetic data...")
        vz, vy, vx, confidence = farneback_cpu.calculate_flow(img1, img2)
        print(f"✓ Farneback3D flow calculation successful. Output shapes: {vz.shape}, {vy.shape}, {vx.shape}, {confidence.shape}")
        
    except Exception as e:
        print(f"✗ Farneback3D test failed: {e}")
        return False
    
    return True

def test_pyrlk_3d():
    """Test PyrLK3D class instantiation and basic functionality."""
    print("Testing PyrLK3D...")
    
    try:
        # Test with CPU first
        pyrlk_cpu = opticalflow3D.PyrLK3D(
            iters=2,
            num_levels=2,
            scale=0.5,
            tau=0.1,
            alpha=0.1,
            filter_size=9,
            device='cpu'
        )
        print("✓ PyrLK3D CPU instantiation successful")
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            pyrlk_cuda = opticalflow3D.PyrLK3D(
                iters=2,
                num_levels=2,
                scale=0.5,
                tau=0.1,
                alpha=0.1,
                filter_size=9,
                device='cuda'
            )
            print("✓ PyrLK3D CUDA instantiation successful")
        
        # Test with small synthetic data
        np.random.seed(42)
        img1 = np.random.rand(32, 32, 32).astype(np.float32)
        img2 = np.random.rand(32, 32, 32).astype(np.float32)
        
        print("Testing PyrLK3D with synthetic data...")
        vz, vy, vx = pyrlk_cpu.calculate_flow(img1, img2)
        print(f"✓ PyrLK3D flow calculation successful. Output shapes: {vz.shape}, {vy.shape}, {vx.shape}")
        
    except Exception as e:
        print(f"✗ PyrLK3D test failed: {e}")
        return False
    
    return True

def test_helper_functions():
    """Test some helper functions."""
    print("Testing helper functions...")
    
    try:
        # Test gaussian_pyramid_3d
        from opticalflow3D.helpers.helpers import gaussian_pyramid_3d
        
        img = np.random.rand(64, 64, 64).astype(np.float32)
        pyramid = gaussian_pyramid_3d(img, levels=3, scale=0.5)
        print(f"✓ gaussian_pyramid_3d successful. Pyramid levels: {len(pyramid)}")
        
        # Test imresize_3d
        from opticalflow3D.helpers.helpers import imresize_3d
        
        resized = imresize_3d(img, (32, 32, 32))
        print(f"✓ imresize_3d successful. Resized shape: {resized.shape}")
        
    except Exception as e:
        print(f"✗ Helper functions test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("OpticalFlow3D PyTorch Conversion Test")
    print("=" * 40)
    
    test_pytorch_availability()
    
    success = True
    success &= test_farneback_3d()
    success &= test_pyrlk_3d()
    success &= test_helper_functions()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! PyTorch conversion successful.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
