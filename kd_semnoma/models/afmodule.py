import torch
from torch import nn


class AFModule(nn.Module):
    def __init__(self, N, num_dim):
        super().__init__()

        self.c_in = N

        self.layers = nn.Sequential(
            nn.Linear(in_features=N + num_dim, out_features=N),
            nn.LeakyReLU(),
            nn.Linear(in_features=N, out_features=N),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x, side_info = x
        #x(batch, 3, W, H)
        #side_info(batch, 1, 1, 1)
        context = torch.mean(x, dim=(2, 3))

        context_input = torch.cat([context, side_info], dim=1)
        #context_input(batch, 4, W, H)
        mask = self.layers(context_input).view(-1, self.c_in, 1, 1)

        out = mask * x
        return out


'''
class ECA_AFModule(nn.Module):
    def __init__(self, N, num_dim, k_size=3):
        super().__init__()
        self.c_in = N
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.proj = nn.Linear(num_dim, N)  

    def forward(self, x):
        x, side_info = x
        context = self.squeeze(x).squeeze(-1).squeeze(-1)  # (batch, N)
        context = context.unsqueeze(1)  # (batch, 1, N)
        mask = torch.sigmoid(self.excite(context)).view(-1, self.c_in, 1, 1)
        
        side_scale = torch.sigmoid(self.proj(side_info)).view(-1, self.c_in, 1, 1)
        return mask * side_scale * x

class CA_AFModule(nn.Module):
    def __init__(self, N, num_dim, reduction=16):
        super().__init__()
        self.c_in = N
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = N // reduction
        self.conv1 = nn.Conv2d(N, mid_channels, kernel_size=1)
        self.conv_h = nn.Conv2d(mid_channels, N, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, N, kernel_size=1)
        self.side_proj = nn.Linear(num_dim, N)

    def forward(self, x):
        x, side_info = x
        h_pool = self.pool_h(x)  # (batch, N, H, 1)
        w_pool = self.pool_w(x)  # (batch, N, 1, W)
        context = torch.cat([h_pool, w_pool], dim=2)  # (batch, N, H+1, 1 or W)
        context = self.conv1(context)
        h_attn = torch.sigmoid(self.conv_h(context[:, :, :h_pool.size(2), :]))
        w_attn = torch.sigmoid(self.conv_w(context[:, :, h_pool.size(2):, :]))
        mask = h_attn * w_attn
        
    
        side_scale = torch.sigmoid(self.side_proj(side_info)).view(-1, self.c_in, 1, 1)
        return mask * side_scale * x

class CBAM_AFModule(nn.Module):
    def __init__(self, N, num_dim, reduction=16):
        super().__init__()
        self.c_in = N

        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(N + num_dim, N // reduction),
            nn.ReLU(),
            nn.Linear(N // reduction, N)
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        x, side_info = x
        avg_pool = self.channel_avg(x).squeeze(-1).squeeze(-1)
        max_pool = self.channel_max(x).squeeze(-1).squeeze(-1)
        context = torch.cat([avg_pool, side_info], dim=1)
        channel_attn = torch.sigmoid(self.channel_fc(context)).view(-1, self.c_in, 1, 1)
        

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1)))
        
        return channel_attn * spatial_attn * x
'''

# CAMBlock
'''
class CAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),  # Down-sample
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1), # Up-sample
            nn.Sigmoid()
        )

    def forward(self, x):
        V = self.conv1(x)
        V = self.conv2(V)
        

        Z = self.global_pool(V)

        
        S = self.attention(Z)
        

        V_hat = S * V
        

        U_hat = V_hat + x
        
        return U_hat

cam = CAMBlock(channels=16)
image = torch.randn(1,16,8,8)
out = cam(image)
print('out',out.size())'
'''