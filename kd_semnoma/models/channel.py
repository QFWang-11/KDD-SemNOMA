import torch
import torch.nn as nn

# Complex AWGN Channel for Multiple Access Channel (MAC)
class ComplexAWGNMAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch

        # inputs: BxCxWxH
        # snr: Bx1
        if x.dim()==5:
            x = torch.sum(x, 1)
        else:
            x = x

        awgn = torch.randn_like(x) * torch.sqrt(10.0 ** (-snr[..., None, None] / 10.0))
        
        awgn = awgn * torch.sqrt(torch.tensor(0.5, device=x.device))

        x = x + awgn

        return x


class ComplexRayleighMAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr, h = batch  # x: (B, Devices, C, W, H), snr: (B,), h: (B, Devices) 复数标量
        
        old_shape = x.shape  # e.g., (32, 2, 16, 8, 8)
        B, D, C, W, H = old_shape
        
        # 将信号重塑为复数形式
        x = x.contiguous().view(B, D, 2, -1)  # (B, D, 2, CWH/2), e.g., (32, 2, 2, 512)
        x = torch.complex(x[:, :, 0, :], x[:, :, 1, :])  # (B, D, CWH/2), e.g., (32, 2, 512)
        
        # 应用瑞利衰落 (每个设备独立的 h_i)
        h = h.view(B, D, 1)  # (B, D, 1)
        faded_signal = x * h  # (B, D, CWH/2)，每个设备乘以对应 h_i
        
        # 叠加所有设备信号
        faded_signal = torch.sum(faded_signal, dim=1)  # (B, CWH/2)，沿设备维度求和
        
        # 计算噪声功率
        noise_power = 10.0 ** (-snr / 10.0)  # (B, 1)
        noise_std = torch.sqrt(noise_power * 0.5)  # (B, 1)
        
        # 生成复高斯噪声
        noise_real = torch.randn_like(faded_signal.real) * noise_std  # (32, 1024)
        noise_imag = torch.randn_like(faded_signal.imag) * noise_std  # (32, 1024)
        noise = torch.complex(noise_real, noise_imag)  # (32, 1024)
        
        # 添加噪声
        noisy_signal = faded_signal + noise  # (32, 1024)
        noisy_signal = torch.stack([noisy_signal.real, noisy_signal.imag], dim=1)  # (B, 2, CWH/2)
        noisy_signal = noisy_signal.view(B, C, W, H)  # (B, C, W, H)，去掉设备维度
        
        return noisy_signal

class ComplexRayleighMAC_for_baseline(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch  # x: (B, Devices, C, W, H), snr: (B, 1)
        
        old_shape = x.shape  # e.g., (32, 2, 16, 8, 8)
        B, D, C, W, H = old_shape
        
        # 生成每个设备的瑞利衰落因子 h_i
        h_real = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h = torch.complex(h_real, h_imag)  # (B, D)，每个设备一个复数标量
        
        # 将信号重塑为复数形式
        num_elements = C * W * H // 2  # 复数信号长度，e.g., 1024
        x = x.contiguous().view(B, D, 2, num_elements)  # (B, D, 2, CWH/2), e.g., (32, 2, 2, 1024)
        x = torch.complex(x[:, :, 0, :], x[:, :, 1, :])  # (B, D, CWH/2), e.g., (32, 2, 1024)
        
        # 应用瑞利衰落 (每个设备独立的 h_i)
        h = h.view(B, D, 1)  # (B, D, 1)
        faded_signal = x * h  # (B, D, CWH/2)，每个设备乘以对应 h_i
        
        # 叠加所有设备信号
        faded_signal = torch.sum(faded_signal, dim=1)  # (B, CWH/2)，沿设备维度求和
        
        # 计算噪声功率
        noise_power = 10.0 ** (-snr / 10.0)  # (B, 1)
        noise_std = torch.sqrt(noise_power * 0.5)  # (B, 1)
        
        # 生成复高斯噪声
        noise_real = torch.randn_like(faded_signal.real) * noise_std  # (B, CWH/2)
        noise_imag = torch.randn_like(faded_signal.imag) * noise_std  # (B, CWH/2)
        noise = torch.complex(noise_real, noise_imag)  # (B, CWH/2)
        
        # 添加噪声
        noisy_signal = faded_signal + noise  # (B, CWH/2)
        noisy_signal = torch.stack([noisy_signal.real, noisy_signal.imag], dim=1)  # (B, 2, CWH/2)
        noisy_signal = noisy_signal.view(B, C, W, H)  # (B, C, W, H)
        
        return noisy_signal

class ComplexRayleighOrthogonalChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Args:
            batch: Tuple of (x, snr, h)
                - x: (B, D, C, W, H), input signals
                - snr: (B, 1), signal-to-noise ratio
                - h: (B, D), complex fading coefficients
        Returns:
            torch.Tensor: (B, D, C, W, H), faded signals with noise per device
        """
        x, snr, h = batch  # x: (B, D, C, W, H), snr: (B, 1), h: (B, D)
        B, D, C, W, H = x.shape

        x = x.contiguous().view(B, D, 2, -1)  # (B, D, 2, CWH/2)
        x = torch.complex(x[:, :, 0, :], x[:, :, 1, :])  # (B, D, CWH/2)


        h = h.view(B, D, 1)  # (B, D, 1)
        faded_signal = x * h  # (B, D, CWH/2)

        noise_power = 10.0 ** (-snr / 10.0)  # (B, 1)
        noise_std = torch.sqrt(noise_power * 0.5)  # (B, 1)


        noise_real = torch.randn_like(faded_signal.real) * noise_std.view(B, 1, 1)
        noise_imag = torch.randn_like(faded_signal.imag) * noise_std.view(B, 1, 1)
        noise = torch.complex(noise_real, noise_imag)  # (B, D, CWH/2)


        noisy_signal = faded_signal + noise  # (B, D, CWH/2)
        noisy_signal = torch.stack([noisy_signal.real, noisy_signal.imag], dim=2)  # (B, D, 2, CWH/2)
        noisy_signal = noisy_signal.view(B, D, C, W, H)  # (B, D, C, W, H)

        return noisy_signal