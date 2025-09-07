import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# Multi-user MSE loss summation
class Sum_MSE(nn.Module):
    def __init__(self):
        super(Sum_MSE, self).__init__()
        self.mse_loss = nn.MSELoss() # Enables global calling, MSELoss is already normalized
    def forward(self, x_hat, x):           # x_hat(batch,2,C,W,H)
        num_devices = x_hat.shape[1]    # Number of user devices
        return torch.stack(
            [
                self.mse_loss(x_hat[:, i, ...], x[:, i, ...]) for i in range(num_devices)
            ]
        ).sum()


class Sum_MAE(nn.Module):
    def __init__(self):
        super(Sum_MAE, self).__init__()
        self.mae_loss = nn.L1Loss()  # Replace MSE with MAE

    def forward(self, x_hat, x):  # x_hat (batch, num_devices, C, W, H)
        num_devices = x_hat.shape[1]  # Number of user devices
        return torch.stack(
            [
                self.mae_loss(x_hat[:, i, ...], x[:, i, ...]) for i in range(num_devices)
            ]
        ).sum()


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse_loss = nn.MSELoss() # Enables global calling, MSELoss is already normalized
    def forward(self, x_hat, x):           # x_hat(batch,C,W,H)
        return self.mse_loss(x_hat , x)


# Tensor has been flattened
class Average_PSNR(nn.Module):
    def __init__(self):  
        super(Average_PSNR, self).__init__()
    def forward(self, x_hat, x): # x:(batch, num_devices, Cin, Win, Hin)
        # num_devices = x.shape[1]      # Number of user devices
        # Cin, Hin, Win = x.shape[2], x.shape[3], x.shape[4]
        psnr = torch.sum(-10*torch.log10(torch.mean((x_hat - x)**2,[-1,-2,-3])))
        return psnr / x.shape[0]


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        """
        Args:
            max_val (float): Maximum pixel value of the image, default 1.0 (for images normalized to [0,1]).
                             If image pixel range is [0, 255], set to 255.0.
        """
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): Reconstructed image, shape can be (batch_size, C, H, W) or (batch_size, num_devices, C, H, W)
            img2 (torch.Tensor): Original image, same shape as img1
        Returns:
            torch.Tensor: PSNR value, scalar (single image) or tensor (batch images)
        """
        # Ensure input dimensions match
        if img1.shape != img2.shape:
            raise ValueError(f"Input images must have the same shape, got {img1.shape} and {img2.shape}")

        # Calculate mean squared error (MSE)
        mse = torch.mean((img1 - img2) ** 2, dim=[-3, -2, -1])  # Average over channel, height, width dimensions

        # Handle multi-user case
        if img1.dim() == 5:  # (batch_size, num_devices, C, H, W)
            mse = mse.mean(dim=1)  # Average over num_devices dimension, resulting in (batch_size,)

        # Calculate PSNR
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-10))  # Avoid division by 0 or log(0)

        # If batch calculation, return average PSNR
        if psnr.dim() > 0:
            return psnr.mean()
        return psnr


# Create 1D Gaussian kernel
def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


# Gaussian filtering
def gaussian_filter(input: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(f"Unsupported input shape: {input.shape}")
    
    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
    return out


# SSIM core calculation
def _ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    data_range: float,
    win: torch.Tensor,
    K: Tuple[float, float] = (0.01, 0.03)
) -> torch.Tensor:
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    return ssim_map.mean(dim=[-3, -2, -1])  # Average over channel, height, width


# SSIM class
class SSIM(nn.Module):
    def __init__(self, data_range: float = 1.0, win_size: int = 11, win_sigma: float = 1.5):
        super(SSIM, self).__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = _fspecial_gauss_1d(win_size, win_sigma)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.shape != Y.shape:
            raise ValueError(f"Input shapes must match, got {X.shape} and {Y.shape}")
        
        # Handle multi-user case
        if X.dim() == 5:  # (batch_size, num_devices, C, H, W)
            X = X.view(-1, *X.shape[2:])  # Merge batch and num_devices
            Y = Y.view(-1, *Y.shape[2:])

        # Dynamically adjust window channel count
        C = X.shape[1]
        win = self.win.repeat([C, 1] + [1] * (X.dim() - 2)).to(X.device)
        ssim_val = _ssim(X, Y, data_range=self.data_range, win=win)
        return ssim_val.mean() if ssim_val.dim() > 0 else ssim_val


# MS-SSIM class
class MS_SSIM(nn.Module):
    def __init__(self, data_range: float = 1.0, win_size: int = 11, win_sigma: float = 1.5, 
                 weights: Optional[List[float]] = None):
        super(MS_SSIM, self).__init__()
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.win = _fspecial_gauss_1d(win_size, win_sigma)
        self.weights = weights if weights is not None else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights_tensor = torch.tensor(self.weights, dtype=torch.float32)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.shape != Y.shape:
            raise ValueError(f"Input shapes must match, got {X.shape} and {Y.shape}")
        
        # Handle multi-user case
        if X.dim() == 5:  # (batch_size, num_devices, C, H, W)
            X = X.view(-1, *X.shape[2:])
            Y = Y.view(-1, *Y.shape[2:])

        C = X.shape[1]
        win = self.win.repeat([C, 1] + [1] * (X.dim() - 2)).to(X.device)
        levels = len(self.weights)
        mcs = []
        avg_pool = F.avg_pool2d if X.dim() == 4 else F.avg_pool3d

        for i in range(levels):
            ssim_map = _ssim(X, Y, data_range=self.data_range, win=win)
            if i < levels - 1:
                mcs.append(torch.relu(ssim_map))
                padding = [s % 2 for s in X.shape[2:]]
                X = avg_pool(X, kernel_size=2, padding=padding)
                Y = avg_pool(Y, kernel_size=2, padding=padding)
            else:
                mcs.append(torch.relu(ssim_map))

        mcs_and_ssim = torch.stack(mcs, dim=0)  # (levels, batch*num_devices)
        ms_ssim_val = torch.prod(mcs_and_ssim ** self.weights_tensor.view(-1, 1), dim=0)
        return ms_ssim_val.mean() if ms_ssim_val.dim() > 0 else ms_ssim_val
