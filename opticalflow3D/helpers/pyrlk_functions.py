import math
import typing

import torch
import torch.nn.functional as F
import numpy as np
from opticalflow3D.helpers.helpers import imresize_3d, gaussian_pyramid_3d


##################################
# PyTorch implementations of matrix operations
##################################

def solve_3x3_system_torch(A, B, reg, threshold):
    """Solve 3x3 linear system using PyTorch operations
    
    Args:
        A: 3x3 matrix (tensor)
        B: 3x1 vector (tensor) 
        reg: regularization matrix (tensor)
        threshold: eigenvalue threshold (float)
        
    Returns:
        solution vector or zeros if eigenvalue threshold not met
    """
    # Add regularization
    A_reg = A + reg
    
    # Check eigenvalue condition (simplified check using determinant)
    det = torch.det(A)
    
    if torch.abs(det) > threshold:
        try:
            solution = torch.linalg.solve(A_reg, B)
            return solution
        except:
            return torch.zeros_like(B)
    else:
        return torch.zeros_like(B)


##################################
# 3D Lucas Kanade Functions
##################################
def calculate_derivatives(image):
    """ Calculates the derivative of the image using predefined kernels with PyTorch

    The smoothing kernel is [0.036, 0.249, 0.437, 0.249, 0.036]
    The differentiation kernel is [-0.108, -0.283, 0, 0.283, 0.108]

    Args:
        image (torch.Tensor): Image to calculate derivatives from

    Returns:
        Ix (torch.Tensor): Image derivative in x direction
        Iy (torch.Tensor): Image derivative in y direction
        Iz (torch.Tensor): Image derivative in z direction
    """
    device = image.device
    p5 = torch.tensor([0.036, 0.249, 0.437, 0.249, 0.036], device=device)
    d5 = torch.tensor([-0.108, -0.283, 0, 0.283, 0.108], device=device)

    # Helper function for 1D convolution along specific axis
    def conv1d_axis(input_tensor, kernel, axis):
        # Add batch and channel dimensions
        input_5d = input_tensor.unsqueeze(0).unsqueeze(0)
        
        # Create 3D kernel for the specific axis
        if axis == 0:  # z-axis
            kernel_3d = kernel.view(1, 1, len(kernel), 1, 1)
            padding = (0, 0, 0, 0, len(kernel)//2, len(kernel)//2)
        elif axis == 1:  # y-axis
            kernel_3d = kernel.view(1, 1, 1, len(kernel), 1)
            padding = (0, 0, len(kernel)//2, len(kernel)//2, 0, 0)
        else:  # x-axis
            kernel_3d = kernel.view(1, 1, 1, 1, len(kernel))
            padding = (len(kernel)//2, len(kernel)//2, 0, 0, 0, 0)
        
        # Apply padding
        padded_input = F.pad(input_5d, padding, mode='reflect')
        
        # Convolve
        output = F.conv3d(padded_input, kernel_3d)
        
        # Remove batch and channel dimensions
        return output.squeeze(0).squeeze(0)

    # calculate Ix
    Ix = conv1d_axis(image, p5, axis=1)  # y direction smoothing
    Ix = conv1d_axis(Ix, p5, axis=0)     # z direction smoothing
    Ix = conv1d_axis(Ix, d5, axis=2)     # x direction differentiation

    # calculate Iy
    Iy = conv1d_axis(image, p5, axis=2)  # x direction smoothing
    Iz = conv1d_axis(Iy, p5, axis=1)     # y direction smoothing (for Iz)
    Iz = conv1d_axis(Iz, d5, axis=0)     # z direction differentiation
    
    # finish Iy calculation
    Iy = conv1d_axis(Iy, p5, axis=0)     # z direction smoothing
    Iy = conv1d_axis(Iy, d5, axis=1)     # y direction differentiation

    return Ix, Iy, Iz


def calculate_difference_torch(image1, image2, vx, vy, vz):
    """ Calculates the difference in image intensity across the time frames using PyTorch

    Args:
        image1 (torch.Tensor): First image in the sequence
        image2 (torch.Tensor): Second image in the sequence
        vx (torch.Tensor): Displacement in x direction
        vy (torch.Tensor): Displacement in y direction
        vz (torch.Tensor): Displacement in z direction

    Returns:
        It (torch.Tensor): Image derivative in t direction
    """
    device = image1.device
    
    # Ensure all tensors have the same shape
    target_shape = image1.shape
    
    def ensure_shape_match(tensor, target_shape):
        if tensor.shape != target_shape:
            return F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        return tensor
    
    # Resize velocity fields to match image shape
    vx = ensure_shape_match(vx, target_shape)
    vy = ensure_shape_match(vy, target_shape)
    vz = ensure_shape_match(vz, target_shape)
    
    depth, length, width = target_shape
    
    # Create coordinate grids
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(length, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Calculate target coordinates
    fx = x_coords.float() + vx
    fy = y_coords.float() + vy
    fz = z_coords.float() + vz
    
    # Normalize coordinates to [-1, 1] for grid_sample
    D, H, W = image2.shape
    coords_norm = torch.stack([
        2.0 * fx / (W - 1) - 1.0,  # x
        2.0 * fy / (H - 1) - 1.0,  # y  
        2.0 * fz / (D - 1) - 1.0   # z
    ], dim=-1)
    
    # Reshape for grid_sample: (N, D, H, W, 3)
    coords_norm = coords_norm.unsqueeze(0)
    
    # Apply grid sampling to interpolate image2
    warped_image2 = F.grid_sample(
        image2.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
        coords_norm,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(0)
    
    # Calculate temporal difference
    It = image1 - warped_image2
    
    return It


def calculate_gradients(Ix, Iy, Iz, filter_fn):
    """ Calculates the component of the A^T W^2 A matrix using PyTorch

    Args:
        Ix (torch.Tensor): Image derivative in x direction
        Iy (torch.Tensor): Image derivative in y direction
        Iz (torch.Tensor): Image derivative in z direction
        filter_fn: Function to determine the window size of the neighbourhood as well as the weights of the window

    Returns:
        Ix2 (torch.Tensor): Image derivative in x*x direction
        IxIy (torch.Tensor): Image derivative in x*y direction
        IxIz (torch.Tensor): Image derivative in x*z direction
        Iy2 (torch.Tensor): Image derivative in y*y direction
        IyIz (torch.Tensor): Image derivative in y*z direction
        Iz2 (torch.Tensor): Image derivative in z*z direction
    """
    Ix2 = filter_fn(Ix * Ix)
    IxIy = filter_fn(Ix * Iy)
    IxIz = filter_fn(Ix * Iz)
    Iy2 = filter_fn(Iy * Iy)
    IyIz = filter_fn(Iy * Iz)
    Iz2 = filter_fn(Iz * Iz)

    return Ix2, IxIy, IxIz, Iy2, IyIz, Iz2


def calculate_mismatch(Ix, Iy, Iz, It, filter_fn):
    """ Calculates the image mismatch b vector using PyTorch

    Args:
        Ix (torch.Tensor): Image derivative in x direction
        Iy (torch.Tensor): Image derivative in y direction
        Iz (torch.Tensor): Image derivative in z direction
        It (torch.Tensor): Image derivative in t direction
        filter_fn: Function to determine the window size of the neighbourhood as well as the weights of the window

    Returns:
        IxIt (torch.Tensor): Image derivative in x*t direction
        IyIt (torch.Tensor): Image derivative in y*t direction
        IzIt (torch.Tensor): Image derivative in z*t direction
    """
    IxIt = filter_fn(Ix * It)
    IyIt = filter_fn(Iy * It)
    IzIt = filter_fn(Iz * It)

    return IxIt, IyIt, IzIt


def calculate_vector_torch(vx, vy, vz, Ix2, IxIy, IxIz, Iy2, IyIz, Iz2, IxIt, IyIt, IzIt, reg, threshold):
    """ Update the displacement field using the calculated image gradients with PyTorch

    Args:
        vx (torch.Tensor): Displacement in x direction (modified in place)
        vy (torch.Tensor): Displacement in y direction (modified in place) 
        vz (torch.Tensor): Displacement in z direction (modified in place)
        Ix2, IxIy, IxIz, Iy2, IyIz, Iz2: Image gradient products
        IxIt, IyIt, IzIt: Image mismatch terms
        reg (torch.Tensor): Regularization matrix
        threshold (float): Eigenvalue threshold

    Returns:
        None (updates vx, vy, vz in place)
    """
    # Ensure all tensors have the same shape as velocity tensors
    def ensure_shape_match(tensor, target_shape):
        if tensor.shape != target_shape:
            return F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        return tensor
    
    target_shape = vx.shape
    
    # Ensure all gradient tensors match velocity shape
    Ix2 = ensure_shape_match(Ix2, target_shape)
    IxIy = ensure_shape_match(IxIy, target_shape)
    IxIz = ensure_shape_match(IxIz, target_shape)
    Iy2 = ensure_shape_match(Iy2, target_shape)
    IyIz = ensure_shape_match(IyIz, target_shape)
    Iz2 = ensure_shape_match(Iz2, target_shape)
    IxIt = ensure_shape_match(IxIt, target_shape)
    IyIt = ensure_shape_match(IyIt, target_shape)
    IzIt = ensure_shape_match(IzIt, target_shape)
    
    # Stack the gradient matrices
    A = torch.stack([
        torch.stack([Ix2, IxIy, IxIz], dim=-1),
        torch.stack([IxIy, Iy2, IyIz], dim=-1),
        torch.stack([IxIz, IyIz, Iz2], dim=-1)
    ], dim=-2)  # Shape: [..., 3, 3]
    
    B = torch.stack([-IxIt, -IyIt, -IzIt], dim=-1)  # Shape: [..., 3]
    
    # Add regularization
    reg_expanded = reg.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(A)
    A_reg = A + reg_expanded
    
    # Calculate eigenvalues for threshold checking (simplified)
    det = torch.det(A)
    
    # Solve the linear system where det > threshold
    valid_mask = torch.abs(det) > threshold
    
    try:
        # Solve Ax = B
        solution = torch.linalg.solve(A_reg, B.unsqueeze(-1)).squeeze(-1)
        
        # Update velocities where solution is valid
        vx += torch.where(valid_mask, solution[..., 0], torch.zeros_like(solution[..., 0]))
        vy += torch.where(valid_mask, solution[..., 1], torch.zeros_like(solution[..., 1]))
        vz += torch.where(valid_mask, solution[..., 2], torch.zeros_like(solution[..., 2]))
    except:
        # If solving fails, don't update velocities
        pass


def pyrlk_3d(image1, image2, iters: int, num_levels: int,
             scale: float = 0.5,
             tau: float = 0.1, alpha: float = 0.1,
             filter_type: str = "gaussian", filter_size: int = 15,
             presmoothing: int = None, threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8)):
    """ Implementation of Pyramidal Lucas Kanade for 3D images using PyTorch

    Args:
        image1 (array): First image in the sequence
        image2 (array): Second image in the sequence
        iters (int): number of iterations
        num_levels (int): number of pyramid levels
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        tau (float): Threshold value to accept calculated displacement. Defaults to 0.1
        alpha (float): Regularization parameter. Defaults to 0.1
        filter_type (int): Defines the type of filter used to average the calculated matrices. Defaults to "gaussian"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 15
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        threadsperblock (typing.Tuple[int, int, int]): Legacy parameter for CUDA compatibility. Ignored in PyTorch version.
        
    Returns:
        vx (torch.Tensor): Displacement in x direction
        vy (torch.Tensor): Displacement in y direction
        vz (torch.Tensor): Displacement in z direction
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert images to torch tensors
    if not isinstance(image1, torch.Tensor):
        image1 = torch.tensor(image1, dtype=torch.float32, device=device)
    if not isinstance(image2, torch.Tensor):
        image2 = torch.tensor(image2, dtype=torch.float32, device=device)
    
    # Ensure images are on the correct device
    image1 = image1.to(device)
    image2 = image2.to(device)

    if presmoothing is not None:
        from opticalflow3D.helpers.farneback_functions import gaussian_filter_3d_torch
        image1 = gaussian_filter_3d_torch(image1, presmoothing)
        image2 = gaussian_filter_3d_torch(image2, presmoothing)

    # initialize variables
    reg = alpha ** 2 * torch.eye(3, dtype=torch.float32, device=device)

    assert filter_type.lower() in ["gaussian", "box"]
    if filter_type.lower() == "gaussian":
        def filter_fn(x):
            from opticalflow3D.helpers.farneback_functions import gaussian_filter_3d_torch
            return gaussian_filter_3d_torch(x, filter_size / 2 * 0.3)
    elif filter_type.lower() == "box":
        def filter_fn(x):
            from opticalflow3D.helpers.farneback_functions import uniform_filter_3d_torch
            return uniform_filter_3d_torch(x, filter_size)

    # initialize gaussian pyramid
    gauss_pyramid_1 = {1: image1}
    gauss_pyramid_2 = {1: image2}
    true_scale_dict = {}
    for pyr_lvl in range(1, num_levels + 1):
        if pyr_lvl == 1:
            gauss_pyramid_1 = {pyr_lvl: image1}
            gauss_pyramid_2 = {pyr_lvl: image2}
        else:
            gauss_pyramid_1[pyr_lvl], true_scale_dict[pyr_lvl] = gaussian_pyramid_3d(gauss_pyramid_1[pyr_lvl - 1],
                                                                                     sigma=1, scale=scale)
            gauss_pyramid_2[pyr_lvl], _ = gaussian_pyramid_3d(gauss_pyramid_2[pyr_lvl - 1], sigma=1, scale=scale)

    # LK code
    for lvl in range(num_levels, 0, -1):
        #         print("Currently working on pyramid level: {}".format(lvl))
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels:
            # initialize velocities
            vx = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
            vy = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
            vz = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
        else:
            # Resize velocity fields and ensure exact size match
            target_shape = lvl_image_1.shape
            vx = 1 / true_scale_dict[lvl + 1][2] * imresize_3d(vx, scale=true_scale_dict[lvl + 1])
            vy = 1 / true_scale_dict[lvl + 1][1] * imresize_3d(vy, scale=true_scale_dict[lvl + 1])
            vz = 1 / true_scale_dict[lvl + 1][0] * imresize_3d(vz, scale=true_scale_dict[lvl + 1])
            
            # Ensure exact size match by using interpolation if necessary
            def match_size(tensor, target_shape):
                current_shape = tensor.shape
                if current_shape != target_shape:
                    return F.interpolate(
                        tensor.unsqueeze(0).unsqueeze(0),
                        size=target_shape,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                return tensor
            
            vx = match_size(vx, target_shape)
            vy = match_size(vy, target_shape)
            vz = match_size(vz, target_shape)

        Ix, Iy, Iz = calculate_derivatives(lvl_image_1)

        Ix2, IxIy, IxIz, Iy2, IyIz, Iz2 = calculate_gradients(Ix, Iy, Iz, filter_fn)

        for _ in range(iters):
            It = calculate_difference_torch(lvl_image_1, lvl_image_2, vx, vy, vz)

            IxIt, IyIt, IzIt = calculate_mismatch(Ix, Iy, Iz, It, filter_fn)

            calculate_vector_torch(vx, vy, vz,
                                  Ix2, IxIy, IxIz, Iy2, IyIz, Iz2, IxIt, IyIt, IzIt,
                                  reg, tau)

    return vx, vy, vz
