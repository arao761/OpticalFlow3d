import math
import typing

import numpy as np
import torch
import torch.nn.functional as F
from opticalflow3D.helpers.helpers import imresize_3d, gaussian_pyramid_3d


def make_abc_fast(signal,
                  spatial_size: int = 9,
                  sigma_k: float = 0.15):
    """Calculates the polynomial expansion coefficients using PyTorch

    Args:
        signal: tensor containing the pixel values of the 3D image.
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.

    Returns:
        Returns the A array in this format as it is symmetrical. This saves memory space.
        a = [[a_00, a_01, a_02],
             [a_01, a_11, a_12],
             [a_02, a_12, a_22]]

        Returns the B array in this format.
        b = [[b_0],
             [b_1],
             [b_2]]

        b_0 (torch.Tensor): tensor containing the first value of the B array
        b_1 (torch.Tensor): tensor containing the second value of the B array
        b_2 (torch.Tensor): tensor containing the third value of the B array
        a_00 (torch.Tensor): tensor containing the values of the A array
        a_01 (torch.Tensor): tensor containing the values of the A array
        a_02 (torch.Tensor): tensor containing the values of the A array
        a_11 (torch.Tensor): tensor containing the values of the A array
        a_12 (torch.Tensor): tensor containing the values of the A array
        a_22 (torch.Tensor): tensor containing the values of the A array

    Raises:
        AssertionError: signal must be array with 3 dimensions
    """
    # Convert to torch tensor if needed
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    
    device = signal.device
    
    if signal.ndim != 3:
        raise AssertionError("signal must be array with 3 dimensions")

    sigma = sigma_k * (spatial_size - 1)

    n = int((spatial_size - 1) / 2)
    a = np.exp(-(np.arange(-n, n + 1, dtype=np.float32) ** 2) / (2 * sigma ** 2))

    # Set up applicability and basis functions
    applicability = np.multiply.outer(np.multiply.outer(a, a), a)
    z, y, x = np.mgrid[-n:n + 1, -n:n + 1, -n:n + 1]

    basis = np.stack((np.ones(x.shape), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z), axis=3)
    nb = basis.shape[3]

    # Compute the inverse metric
    q = np.zeros((nb, nb), dtype=np.float32)
    for i in range(nb):
        for j in range(i, nb):
            q[i, j] = np.sum(basis[..., i] * applicability * basis[..., j])
            q[j, i] = q[i, j]

    del basis, applicability, x, y, z
    qinv = np.linalg.inv(q)

    # Convert kernels to torch tensors
    kernel_0 = torch.tensor(a, dtype=torch.float32, device=device)
    kernel_1 = torch.tensor(np.arange(-n, n + 1, dtype=np.float32) * a, dtype=torch.float32, device=device)
    kernel_2 = torch.tensor(np.arange(-n, n + 1, dtype=np.float32) ** 2 * a, dtype=torch.float32, device=device)

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

    # convolutions in z
    conv_z0 = conv1d_axis(signal, kernel_0, axis=0)
    conv_z1 = conv1d_axis(signal, kernel_1, axis=0)
    conv_z2 = conv1d_axis(signal, kernel_2, axis=0)

    # convolutions in y
    conv_z0y0 = conv1d_axis(conv_z0, kernel_0, axis=1)
    conv_z0y1 = conv1d_axis(conv_z0, kernel_1, axis=1)
    conv_z0y2 = conv1d_axis(conv_z0, kernel_2, axis=1)
    del conv_z0

    conv_z1y0 = conv1d_axis(conv_z1, kernel_0, axis=1)
    conv_z1y1 = conv1d_axis(conv_z1, kernel_1, axis=1)
    del conv_z1

    conv_z2y0 = conv1d_axis(conv_z2, kernel_0, axis=1)
    del conv_z2

    # convolutions in x
    conv_z0y0x0 = conv1d_axis(conv_z0y0, kernel_0, axis=2)
    b_0 = qinv[1, 1] * conv1d_axis(conv_z0y0, kernel_1, axis=2)
    a_00 = qinv[4, 4] * conv1d_axis(conv_z0y0, kernel_2, axis=2) + qinv[4, 0] * conv_z0y0x0
    del conv_z0y0

    b_1 = qinv[2, 2] * conv1d_axis(conv_z0y1, kernel_0, axis=2)
    a_01 = qinv[7, 7] * conv1d_axis(conv_z0y1, kernel_1, axis=2) / 2
    del conv_z0y1

    a_11 = qinv[5, 5] * conv1d_axis(conv_z0y2, kernel_0, axis=2) + qinv[5, 0] * conv_z0y0x0
    del conv_z0y2

    b_2 = qinv[3, 3] * conv1d_axis(conv_z1y0, kernel_0, axis=2)
    a_02 = qinv[8, 8] * conv1d_axis(conv_z1y0, kernel_1, axis=2) / 2
    del conv_z1y0

    a_12 = qinv[9, 9] * conv1d_axis(conv_z1y1, kernel_0, axis=2) / 2
    del conv_z1y1

    a_22 = qinv[6, 6] * conv1d_axis(conv_z2y0, kernel_0, axis=2) + qinv[6, 0] * conv_z0y0x0
    del conv_z2y0, conv_z0y0x0

    return b_0, b_1, b_2, a_00, a_01, a_02, a_11, a_12, a_22


def update_matrices_torch(b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22,
                         b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22,
                         vx, vy, vz, border):
    """Sets up the matrices that can be used to solve for the velocities using PyTorch

        Matrices are in the format [[g00, g01, g02], and [[h0],
                                    [g01, g11, g12],      [h1],
                                    [g02, g12, g22]]      [h2]]

    Args:
        All arguments are torch tensors containing the polynomial expansion coefficients,
        velocities, and border weights.

    Returns:
        h0, h1, h2, g00, g01, g02, g11, g12, g22: Updated matrices as torch tensors
    """
    device = vx.device
    depth, length, width = vx.shape
    
    # Ensure all input tensors have the same shape as velocity tensors
    def ensure_shape_match(tensor, target_shape):
        if tensor.shape != target_shape:
            return F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        return tensor
    
    target_shape = (depth, length, width)
    
    # Ensure all coefficient tensors match velocity shape
    b1_0 = ensure_shape_match(b1_0, target_shape)
    b1_1 = ensure_shape_match(b1_1, target_shape)
    b1_2 = ensure_shape_match(b1_2, target_shape)
    a1_00 = ensure_shape_match(a1_00, target_shape)
    a1_01 = ensure_shape_match(a1_01, target_shape)
    a1_02 = ensure_shape_match(a1_02, target_shape)
    a1_11 = ensure_shape_match(a1_11, target_shape)
    a1_12 = ensure_shape_match(a1_12, target_shape)
    a1_22 = ensure_shape_match(a1_22, target_shape)
    
    b2_0 = ensure_shape_match(b2_0, target_shape)
    b2_1 = ensure_shape_match(b2_1, target_shape)
    b2_2 = ensure_shape_match(b2_2, target_shape)
    a2_00 = ensure_shape_match(a2_00, target_shape)
    a2_01 = ensure_shape_match(a2_01, target_shape)
    a2_02 = ensure_shape_match(a2_02, target_shape)
    a2_11 = ensure_shape_match(a2_11, target_shape)
    a2_12 = ensure_shape_match(a2_12, target_shape)
    a2_22 = ensure_shape_match(a2_22, target_shape)
    
    # Initialize output matrices
    h0 = torch.zeros_like(vx)
    h1 = torch.zeros_like(vx)
    h2 = torch.zeros_like(vx)
    g00 = torch.zeros_like(vx)
    g01 = torch.zeros_like(vx)
    g02 = torch.zeros_like(vx)
    g11 = torch.zeros_like(vx)
    g12 = torch.zeros_like(vx)
    g22 = torch.zeros_like(vx)
    
    # Create coordinate grids
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(length, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Get displacements
    dx = vx
    dy = vy
    dz = vz
    
    # Calculate target coordinates
    fx = x_coords.float() + dx
    fy = y_coords.float() + dy
    fz = z_coords.float() + dz
    
    # Floor coordinates for interpolation
    x1 = fx.floor().long()
    y1 = fy.floor().long()
    z1 = fz.floor().long()
    
    # Fractional parts
    fx_frac = fx - x1.float()
    fy_frac = fy - y1.float()
    fz_frac = fz - z1.float()
    
    # Check bounds
    valid_mask = (x1 >= 0) & (y1 >= 0) & (z1 >= 0) & \
                 (x1 < (width - 1)) & (y1 < (length - 1)) & (z1 < (depth - 1))
    
    # Initialize interpolation coefficients
    a000 = (1.0 - fx_frac) * (1.0 - fy_frac) * (1.0 - fz_frac)
    a001 = fx_frac * (1.0 - fy_frac) * (1.0 - fz_frac)
    a010 = (1.0 - fx_frac) * fy_frac * (1.0 - fz_frac)
    a100 = (1.0 - fx_frac) * (1.0 - fy_frac) * fz_frac
    a011 = fx_frac * fy_frac * (1.0 - fz_frac)
    a101 = fx_frac * (1.0 - fy_frac) * fz_frac
    a110 = (1.0 - fx_frac) * fy_frac * fz_frac
    a111 = fx_frac * fy_frac * fz_frac
    
    # Initialize r arrays for interpolation results
    r = torch.zeros((depth, length, width, 9), device=device)
    
    # Vectorized interpolation for each component where valid
    if valid_mask.any():
        # Extract valid indices
        valid_z, valid_y, valid_x = torch.where(valid_mask)
        
        # Get the 8 corner indices for trilinear interpolation
        z1_v = z1[valid_mask]
        y1_v = y1[valid_mask]
        x1_v = x1[valid_mask]
        
        # Interpolate all b2 and a2 components
        for idx, tensor in enumerate([b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22]):
            interp_val = (a000[valid_mask] * tensor[z1_v, y1_v, x1_v] +
                         a001[valid_mask] * tensor[z1_v, y1_v, x1_v + 1] +
                         a010[valid_mask] * tensor[z1_v, y1_v + 1, x1_v] +
                         a100[valid_mask] * tensor[z1_v + 1, y1_v, x1_v] +
                         a011[valid_mask] * tensor[z1_v, y1_v + 1, x1_v + 1] +
                         a101[valid_mask] * tensor[z1_v + 1, y1_v, x1_v + 1] +
                         a110[valid_mask] * tensor[z1_v + 1, y1_v + 1, x1_v] +
                         a111[valid_mask] * tensor[z1_v + 1, y1_v + 1, x1_v + 1])
            r[valid_z, valid_y, valid_x, idx] = interp_val
    
    # Combine with first image coefficients
    r[:, :, :, 3] = torch.where(valid_mask, (a1_00 + r[:, :, :, 3]) * 0.5, a1_00)
    r[:, :, :, 4] = torch.where(valid_mask, (a1_01 + r[:, :, :, 4]) * 0.25, a1_01 * 0.5)
    r[:, :, :, 5] = torch.where(valid_mask, (a1_02 + r[:, :, :, 5]) * 0.25, a1_02 * 0.5)
    r[:, :, :, 6] = torch.where(valid_mask, (a1_11 + r[:, :, :, 6]) * 0.5, a1_11)
    r[:, :, :, 7] = torch.where(valid_mask, (a1_12 + r[:, :, :, 7]) * 0.25, a1_12 * 0.5)
    r[:, :, :, 8] = torch.where(valid_mask, (a1_22 + r[:, :, :, 8]) * 0.5, a1_22)
    
    # Calculate residuals
    r[:, :, :, 0] = torch.where(valid_mask, 
                               ((b1_0 - r[:, :, :, 0]) * 0.5) + (r[:, :, :, 3] * dx + r[:, :, :, 4] * dy + r[:, :, :, 5] * dz),
                               b1_0)
    r[:, :, :, 1] = torch.where(valid_mask,
                               ((b1_1 - r[:, :, :, 1]) * 0.5) + (r[:, :, :, 4] * dx + r[:, :, :, 6] * dy + r[:, :, :, 7] * dz),
                               b1_1)
    r[:, :, :, 2] = torch.where(valid_mask,
                               ((b1_2 - r[:, :, :, 2]) * 0.5) + (r[:, :, :, 5] * dx + r[:, :, :, 7] * dy + r[:, :, :, 8] * dz),
                               b1_2)
    
    # Apply border scaling
    border_size = len(border) - 1
    scale_x = torch.minimum(x_coords, torch.tensor(border_size, device=device))
    scale_x = torch.minimum(scale_x, width - x_coords - 1)
    scale_x = torch.minimum(scale_x, torch.tensor(border_size, device=device))
    
    scale_y = torch.minimum(y_coords, torch.tensor(border_size, device=device))
    scale_y = torch.minimum(scale_y, length - y_coords - 1)
    scale_y = torch.minimum(scale_y, torch.tensor(border_size, device=device))
    
    scale_z = torch.minimum(z_coords, torch.tensor(border_size, device=device))
    scale_z = torch.minimum(scale_z, depth - z_coords - 1)
    scale_z = torch.minimum(scale_z, torch.tensor(border_size, device=device))
    
    scale = border[scale_x] * border[scale_y] * border[scale_z]
    scale = scale.unsqueeze(-1).expand_as(r)
    
    r = r * scale
    
    # Calculate G and H matrices
    g00 = r[:, :, :, 3] * r[:, :, :, 3] + r[:, :, :, 4] * r[:, :, :, 4] + r[:, :, :, 5] * r[:, :, :, 5]
    g01 = r[:, :, :, 3] * r[:, :, :, 4] + r[:, :, :, 4] * r[:, :, :, 6] + r[:, :, :, 5] * r[:, :, :, 7]
    g02 = r[:, :, :, 3] * r[:, :, :, 5] + r[:, :, :, 4] * r[:, :, :, 7] + r[:, :, :, 5] * r[:, :, :, 8]
    g11 = r[:, :, :, 4] * r[:, :, :, 4] + r[:, :, :, 6] * r[:, :, :, 6] + r[:, :, :, 7] * r[:, :, :, 7]
    g12 = r[:, :, :, 4] * r[:, :, :, 5] + r[:, :, :, 6] * r[:, :, :, 7] + r[:, :, :, 7] * r[:, :, :, 8]
    g22 = r[:, :, :, 5] * r[:, :, :, 5] + r[:, :, :, 7] * r[:, :, :, 7] + r[:, :, :, 8] * r[:, :, :, 8]
    
    h0 = r[:, :, :, 3] * r[:, :, :, 0] + r[:, :, :, 4] * r[:, :, :, 1] + r[:, :, :, 5] * r[:, :, :, 2]
    h1 = r[:, :, :, 4] * r[:, :, :, 0] + r[:, :, :, 6] * r[:, :, :, 1] + r[:, :, :, 7] * r[:, :, :, 2]
    h2 = r[:, :, :, 5] * r[:, :, :, 0] + r[:, :, :, 7] * r[:, :, :, 1] + r[:, :, :, 8] * r[:, :, :, 2]
    
    return h0, h1, h2, g00, g01, g02, g11, g12, g22


def calculate_confidence_torch(h0, h1, h2, g00, g01, g02, g11, g12, g22,
                              vx, vy, vz):
    """ Calculates the confidence of the Farneback algorithm using PyTorch. 
    Smaller values indicate that the algorithm is more confident.

    Args:
        h0, h1, h2: torch tensors containing the h matrix values
        g00, g01, g02, g11, g12, g22: torch tensors containing the g matrix values
        vx, vy, vz: torch tensors containing the displacements

    Returns:
        confidence: torch tensor containing the calculated confidence
    """
    confidence = (h0 ** 2 + h1 ** 2 + h2 ** 2) - \
                 (vx * (g00 * h0 + g01 * h1 + g02 * h2) +
                  vy * (g01 * h0 + g11 * h1 + g12 * h2) +
                  vz * (g02 * h0 + g12 * h1 + g22 * h2))
    
    return confidence


def update_flow_torch(h0, h1, h2, g00, g01, g02, g11, g12, g22,
                     vx, vy, vz):
    """ Updates the displacements using the calculated matrices with PyTorch.

    Args:
        h0, h1, h2: torch tensors containing the h matrix values
        g00, g01, g02, g11, g12, g22: torch tensors containing the g matrix values
        vx, vy, vz: torch tensors containing the displacements (modified in place)

    Returns:
        None (modifies vx, vy, vz in place)
    """
    # Calculate determinant
    det = g00 * (g11 * g22 - g12 * g12) - \
          g01 * (g01 * g22 - g02 * g12) + \
          g02 * (g01 * g12 - g02 * g11)
    
    # Avoid division by zero
    det = torch.where(torch.abs(det) < 1e-10, torch.ones_like(det) * 1e-10, det)
    
    # Calculate new velocities using Cramer's rule
    vx_new = (h0 * (g11 * g22 - g12 * g12) -
              g01 * (h1 * g22 - h2 * g12) +
              g02 * (h1 * g12 - h2 * g11)) / det
    
    vy_new = (g00 * (h1 * g22 - h2 * g12) -
              h0 * (g01 * g22 - g02 * g12) +
              g02 * (g01 * h2 - g02 * h1)) / det
    
    vz_new = (g00 * (g11 * h2 - g12 * h1) -
              g01 * (g01 * h2 - g02 * h1) +
              h0 * (g01 * g12 - g02 * g11)) / det
    
    # Update velocities in place
    vx.copy_(vx_new)
    vy.copy_(vy_new)
    vz.copy_(vz_new)


def farneback_3d(image1, image2, iters: int, num_levels: int,
                 scale: float = 0.5, spatial_size: int = 9, sigma_k: float = 0.15,
                 filter_type: str = "box", filter_size: int = 5,
                 presmoothing: int = None, threadsperblock: typing.Tuple[int, int, int] = (8, 8, 8)):
    """ Estimates the displacement across image1 and image2 using the 3D Farneback two frame algorithm

    Args:
        image1 (array): first image
        image2 (array): second image
        iters (int): number of iterations
        num_levels (int): number of pyramid levels
        scale (float): Scaling factor used to generate the pyramid levels. Defaults to 0.5
        spatial_size (int): size of the support used in the calculation of the standard deviation of the Gaussian
            applicability. Defaults to 9.
        sigma_k (float): scaling factor used to calculate the standard deviation of the Gaussian applicability. The
            formula to calculate sigma is sigma_k*(spatial_size - 1). Defaults to 0.15.
        filter_type (int): Defines the type of filter used to average the calculated matrices. Defaults to "box"
        filter_size (int): Size of the filter used to average the matrices. Defaults to 5
        presmoothing (int): Standard deviation used to perform Gaussian smoothing of the images. Defaults to None
        threadsperblock (typing.Tuple[int, int, int]): Legacy parameter for CUDA compatibility. Ignored in PyTorch version.

    Returns:
        vx (torch.Tensor): tensor containing the displacements in the x direction
        vy (torch.Tensor): tensor containing the displacements in the y direction
        vz (torch.Tensor): tensor containing the displacements in the z direction
        confidence (torch.Tensor): tensor containing the calculated confidence of the Farneback algorithm
    """
    assert filter_type.lower() in ["gaussian", "box"]
    
    # Determine device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define filter function
    if filter_type.lower() == "gaussian":
        def filter_fn(x):
            sigma = filter_size / 2 * 0.3
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            return gaussian_filter_3d_torch(x, sigma, kernel_size)
    elif filter_type.lower() == "box":
        def filter_fn(x):
            return uniform_filter_3d_torch(x, filter_size)

    # Convert images to torch tensors
    if not isinstance(image1, torch.Tensor):
        image1 = torch.tensor(image1, dtype=torch.float32, device=device)
    if not isinstance(image2, torch.Tensor):
        image2 = torch.tensor(image2, dtype=torch.float32, device=device)
    
    # Ensure images are on the correct device
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    if presmoothing is not None:
        image1 = gaussian_filter_3d_torch(image1, presmoothing)
        image2 = gaussian_filter_3d_torch(image2, presmoothing)

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

    if type(iters) != list:
        iters = [iters] * num_levels

    # Pyr code
    for lvl in range(num_levels, 0, -1):
        # print("Currently working on pyramid level: {}".format(lvl))
        lvl_image_1 = gauss_pyramid_1[lvl]
        lvl_image_2 = gauss_pyramid_2[lvl]

        if lvl == num_levels:
            # initialize velocities
            vx = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
            vy = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
            vz = torch.zeros(lvl_image_1.shape, dtype=torch.float32, device=device)
        else:
            # check if nan values are present
            vx[torch.isnan(vx)] = 0
            vy[torch.isnan(vy)] = 0
            vz[torch.isnan(vz)] = 0

            # Remove large confidence values
            mask = torch.abs(confidence) > 1
            vx[mask] = 0
            vy[mask] = 0
            vz[mask] = 0
            del confidence

            # Resize velocity fields to match current level image size exactly
            target_shape = lvl_image_1.shape
            vx = 1 / true_scale_dict[lvl + 1][2] * imresize_3d(vx, scale=true_scale_dict[lvl + 1])
            vy = 1 / true_scale_dict[lvl + 1][1] * imresize_3d(vy, scale=true_scale_dict[lvl + 1])
            vz = 1 / true_scale_dict[lvl + 1][0] * imresize_3d(vz, scale=true_scale_dict[lvl + 1])
            
            # Ensure exact size match by cropping or padding if necessary
            def match_size(tensor, target_shape):
                current_shape = tensor.shape
                if current_shape != target_shape:
                    # Use interpolation to get exact target shape
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

        b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22 = make_abc_fast(lvl_image_1, spatial_size,
                                                                                   sigma_k=sigma_k)
        b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22 = make_abc_fast(lvl_image_2, spatial_size,
                                                                                   sigma_k=sigma_k)

        border = torch.tensor([0.14, 0.14, 0.4472, 0.4472, 0.4472, 1], dtype=torch.float32, device=device)

        for i in range(iters[lvl - 1]):
            h0, h1, h2, g00, g01, g02, g11, g12, g22 = update_matrices_torch(
                b1_0, b1_1, b1_2, a1_00, a1_01, a1_02, a1_11, a1_12, a1_22,
                b2_0, b2_1, b2_2, a2_00, a2_01, a2_02, a2_11, a2_12, a2_22,
                vx, vy, vz, border)

            h0 = filter_fn(h0)
            h1 = filter_fn(h1)
            h2 = filter_fn(h2)
            g00 = filter_fn(g00)
            g01 = filter_fn(g01)
            g02 = filter_fn(g02)
            g11 = filter_fn(g11)
            g12 = filter_fn(g12)
            g22 = filter_fn(g22)

            update_flow_torch(h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz)

            if i == iters[lvl - 1] - 1:
                confidence = calculate_confidence_torch(h0, h1, h2, g00, g01, g02, g11, g12, g22, vx, vy, vz)

    return vx, vy, vz, confidence


def gaussian_filter_3d_torch(input_tensor, sigma, kernel_size=None):
    """Apply 3D Gaussian filter using PyTorch"""
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    
    # Make sure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply separable convolution
    # Add batch and channel dimensions
    tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution along each axis
    pad = kernel_size // 2
    
    # Z-axis
    kernel_z = kernel_1d.view(1, 1, kernel_size, 1, 1)
    tensor = F.conv3d(F.pad(tensor, (0, 0, 0, 0, pad, pad), mode='replicate'), kernel_z)
    
    # Y-axis
    kernel_y = kernel_1d.view(1, 1, 1, kernel_size, 1)
    tensor = F.conv3d(F.pad(tensor, (0, 0, pad, pad, 0, 0), mode='replicate'), kernel_y)
    
    # X-axis
    kernel_x = kernel_1d.view(1, 1, 1, 1, kernel_size)
    tensor = F.conv3d(F.pad(tensor, (pad, pad, 0, 0, 0, 0), mode='replicate'), kernel_x)
    
    return tensor.squeeze(0).squeeze(0)


def uniform_filter_3d_torch(input_tensor, size):
    """Apply 3D uniform (box) filter using PyTorch"""
    # Create uniform kernel
    kernel_1d = torch.ones(size, dtype=torch.float32, device=input_tensor.device) / size
    
    # Apply separable convolution
    tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    pad = size // 2
    
    # Z-axis
    kernel_z = kernel_1d.view(1, 1, size, 1, 1)
    tensor = F.conv3d(F.pad(tensor, (0, 0, 0, 0, pad, pad), mode='replicate'), kernel_z)
    
    # Y-axis
    kernel_y = kernel_1d.view(1, 1, 1, size, 1)
    tensor = F.conv3d(F.pad(tensor, (0, 0, pad, pad, 0, 0), mode='replicate'), kernel_y)
    
    # X-axis
    kernel_x = kernel_1d.view(1, 1, 1, 1, size)
    tensor = F.conv3d(F.pad(tensor, (pad, pad, 0, 0, 0, 0), mode='replicate'), kernel_x)
    
    return tensor.squeeze(0).squeeze(0)
