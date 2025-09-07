"""
This part of the code is built based on the project:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import torch
torch.cuda.current_device()
import torch.nn as nn

import utils_eval

import matplotlib.pyplot as plt
import numpy as np
from vgg16 import vgg
import lpips

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        # 初始化 LPIPS 损失（使用默认的 VGG 网络）
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)  # 确保在指定设备上运行

    def forward(self, x_hat, x):
        """
        计算 LPIPS 损失，直接对整个批量进行计算。
        
        Args:
            x_hat (torch.Tensor): 生成图像，形状为 [batch*num_devices, C, H, W]
            x (torch.Tensor): 真实图像，形状为 [batch*num_devices, C, H, W]
        
        Returns:
            torch.Tensor: 整个批量的 LPIPS 损失均值
        """
        # 直接计算整个批量的 LPIPS 损失并取均值
        loss = self.lpips_loss(x_hat, x).mean()
        return loss

'''
class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        # 初始化 LPIPS 损失（使用默认的 VGG 网络）
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)  # 如果在 CPU 上运行，则去掉 `.cuda()`

    def forward(self, x_hat, x):  # x_hat(batch, 2, C, W, H)
        #num_devices = x_hat.shape[1]  # 用户设备数量

        # 逐设备计算 LPIPS 损失
        return self.lpips_loss(x_hat, x).mean()
'''
class MinimumPixelLoss():

    def __init__(self, opt=2):

        self.criterion = None
        if opt == 1:
            self.criterion = nn.L1Loss(reduce=False)
        elif opt == 2:
            self.criterion = nn.MSELoss(reduce=False)
        else:
            raise NotImplementedError('opt expected to be 1 (L1 loss) or '
                                      '2 (L2 loss) but received %d' % opt)

    def forward(self, batch, G_pred1, G_pred2):    #(image, csi) = batch
        image, csi = batch
        num_devices = image.shape[1]

        target_1 = (image[:,0,:,:,:]).to(device)
        target_2 = (image[:,1,:,:,:]).to(device)

        # compute loss on each img
        loss_1 = torch.mean(self.criterion(G_pred1, target_1), dim=[1, 2, 3]) \
                 + torch.mean(self.criterion(G_pred2, target_2), dim=[1, 2, 3])

        # exchange order and compute loss
        loss_2 = torch.mean(self.criterion(G_pred1, target_2), dim=[1, 2, 3]) \
                 + torch.mean(self.criterion(G_pred2, target_1), dim=[1, 2, 3])

        loss_min = torch.min(loss_1, loss_2)

        return torch.mean(loss_min)

class MinimumSemanticfeatureLoss():

    def __init__(self, opt=2):

        self.criterion = None
        if opt == 1:
            self.criterion = nn.L1Loss(reduce=False)
        elif opt == 2:
            self.criterion = nn.MSELoss(reduce=False)
        else:
            raise NotImplementedError('opt expected to be 1 (L1 loss) or '
                                      '2 (L2 loss) but received %d' % opt)

    def forward(self, Se_real, G_pred1, G_pred2):    #(image, csi) = batch
        num_devices = Se_real.shape[1]

        target_1 = (Se_real[:,0,:,:,:]).to(device)
        target_2 = (Se_real[:,1,:,:,:]).to(device)

        # compute loss on each img
        loss_1 = torch.mean(self.criterion(G_pred1, target_1), dim=[1, 2, 3]) \
                 + torch.mean(self.criterion(G_pred2, target_2), dim=[1, 2, 3])

        # exchange order and compute loss
        loss_2 = torch.mean(self.criterion(G_pred1, target_2), dim=[1, 2, 3]) \
                 + torch.mean(self.criterion(G_pred2, target_1), dim=[1, 2, 3])

        loss_min = torch.min(loss_1, loss_2)

        return torch.mean(loss_min)



class PixelLoss():

    def __init__(self, opt=2):

        self.criterion = None
        if opt == 1:
            self.criterion = nn.L1Loss()
        elif opt == 2:
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('opt expected to be 1 (L1 loss) or '
                                      '2 (L2 loss) but received %d' % opt)

    def forward(self, batch, G_pred1, G_pred2):

        target_1 = (batch['gt1']).to(device)
        target_2 = (batch['gt2']).to(device)

        # compute loss on each img
        loss = self.criterion(G_pred1, target_1) + self.criterion(G_pred2, target_2)

        return loss



class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

    def forward(self, G_pred1, G_pred2):
        img1 = G_pred1.to(device)
        img2 = G_pred2.to(device)
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0




class KurtosisLoss():

    def __init__(self):
        return

    def kurtosis(self, img):
        y = img - torch.mean(img)
        a = torch.mean(torch.pow(y, 4))
        b = torch.mean(y ** 2) ** 2
        k = a / (b + 1e-9)
        return k / 50.

    def forward(self, G_pred1, G_pred2):

        k1 = self.kurtosis(G_pred1)
        k2 = self.kurtosis(G_pred2)
        loss = k1 + k2

        return loss

'''
class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.mse_loss = nn.MSELoss() #这样可以进行全局调用，MSEloss里已经进行归一化了
        self.vgg16 = vgg.cuda()
    def forward(self, x_hat, x):           #x_hat(batch,2,C,W,H)
        num_devices = x_hat.shape[1]    #用户设备数量
        return torch.stack(
            [
                self.mse_loss(self.vgg16(x_hat[:, i, ...]), self.vgg16(x[:, i, ...])) for i in range(num_devices)
            ]
        ).sum()
'''

class Sum_MAE(nn.Module):
    def __init__(self):
        super(Sum_MAE, self).__init__()
        self.mae_loss = nn.L1Loss()  # 替换 MSE 为 MAE

    def forward(self, x_hat, x):  # x_hat (batch, num_devices, C, W, H)
        num_devices = x_hat.shape[1]  # 用户设备数量
        return torch.stack(
            [
                self.mae_loss(x_hat[:, i, ...], x[:, i, ...]) for i in range(num_devices)
            ]
        ).sum()

class Sum_MSE(nn.Module):
    def __init__(self):
        super(Sum_MSE, self).__init__()
        self.mse_loss = nn.MSELoss() #这样可以进行全局调用，MSEloss里已经进行归一化了
    def forward(self, x_hat, x):           #x_hat(batch,2,C,W,H)
        num_devices = x_hat.shape[1]    #用户设备数量
        return torch.stack(
            [
                self.mse_loss(x_hat[:, i, ...], x[:, i, ...]) for i in range(num_devices)
            ]
        ).sum()

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse_loss = nn.MSELoss() #这样可以进行全局调用，MSEloss里已经进行归一化了
    def forward(self, x_hat, x):           #x_hat(batch,2,C,W,H)
        #num_devices = x_hat.shape[1]    #用户设备数量
        return self.mse_loss(x_hat, x)

class Average_PSNR(nn.Module):
    def __init__(self):  
        super(Average_PSNR, self).__init__()
    def forward(self, x_hat, x): #x:(batch, num_devices, Cin, Win, Hin)
        #num_devices = x.shape[1]      #用户设备数目
        #Cin, Hin, Win = x.shape[2], x.shape[3], x.shape[4]
        psnr = torch.sum(-10*torch.log10(torch.mean((x_hat - x)**2,[-1,-2,-3])))
        return psnr / x.shape[0]

class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        """
        Args:
            max_val (float): 图像像素值的最大值，默认1.0（适用于归一化到[0,1]的图像）。
                             如果图像像素值范围是[0, 255]，则设为255.0。
        """
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): 重构图像，形状可以是 (batch_size, C, H, W) 或 (batch_size, num_devices, C, H, W)
            img2 (torch.Tensor): 原始图像，形状与img1相同
        Returns:
            torch.Tensor: PSNR值，标量（单张图片）或张量（批量图片）
        """
        # 确保输入维度匹配
        if img1.shape != img2.shape:
            raise ValueError(f"Input images must have the same shape, got {img1.shape} and {img2.shape}")

        # 计算均方误差 (MSE)
        mse = torch.mean((img1 - img2) ** 2, dim=[-3, -2, -1])  # 对通道、高、宽维度求均值

        # 处理多用户情况
        if img1.dim() == 5:  # (batch_size, num_devices, C, H, W)
            mse = mse.mean(dim=1)  # 对num_devices维度求均值，得到 (batch_size,)

        # 计算PSNR
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-10))  # 避免除以0或log(0)

        # 如果是批量计算，返回平均PSNR
        if psnr.dim() > 0:
            return psnr.mean()
        return psnr

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# 创建1D高斯核
def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

# 高斯滤波
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

# SSIM核心计算
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
    return ssim_map.mean(dim=[-3, -2, -1])  # 对通道、高、宽求均值

# SSIM类
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
        
        # 处理多用户情况
        if X.dim() == 5:  # (batch_size, num_devices, C, H, W)
            X = X.view(-1, *X.shape[2:])  # 合并batch和num_devices
            Y = Y.view(-1, *Y.shape[2:])

        # 动态调整窗口通道数
        C = X.shape[1]
        win = self.win.repeat([C, 1] + [1] * (X.dim() - 2)).to(X.device)
        ssim_val = _ssim(X, Y, data_range=self.data_range, win=win)
        return ssim_val.mean() if ssim_val.dim() > 0 else ssim_val

def calculate_psnr(x_real, x_hat, max_val=1.0):
    mse = ((x_real - x_hat) ** 2).mean()  # MSE计算
    if mse == 0:  # 防止除以零
        return float('inf')  # 如果MSE为0，表示完全匹配，PSNR为无穷大
    return 10 * torch.log10((max_val ** 2) / mse)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):   #ground truth is for real image or fake image
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None





