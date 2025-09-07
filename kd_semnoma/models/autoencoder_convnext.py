import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from timm.layers import trunc_normal_, DropPath

# Enhanced Adaptive Feature Fusion Module(AF-Module)
class AFModule(nn.Module):
    def __init__(self, N, num_dim=2):
        super(AFModule, self).__init__()
        self.c_in = N
        self.num_dim = num_dim

        # Channel information enhancement module
        self.channel_encoder = nn.Sequential(
            nn.Linear(num_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, N)
        )

        # Adaptive feature fusion module
        self.fusion_net = nn.Sequential(
            nn.Linear(2*N, 4*N),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*N, 4*N),
            nn.LeakyReLU(),
            nn.Linear(4*N, N),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, side_info = x
        batch_size = x.size(0)
        
        # Spatial context features (batch, C)
        context = torch.mean(x, dim=(2, 3))
        
        # Channel feature encoding (batch, num_dim) -> (batch, N)
        side_info = side_info.view(batch_size, self.num_dim)
        channel_feat = self.channel_encoder(side_info)
        
        # Feature fusion (batch, 2N)
        fused_feat = torch.cat([context, channel_feat], dim=1)
        
        # Generate adaptive mask
        mask = self.fusion_net(fused_feat).view(-1, self.c_in, 1, 1)
        
        return mask * x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers  单独的下采样层
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")    # stem layer
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Upsampling module (for decoder)
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=2):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        factor = upsample_factor ** 2
        self.conv = nn.Conv2d(in_channels, out_channels * factor, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upsample_factor)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


# DeepJSCC Encoder Decoder for cifar-10
'''
class DeepJSCCEncoder(nn.Module):
    def __init__(self, N=256, M=16, C=3, depths=[3, 3], dims=[96, 192], drop_path_rate=0.1):
        super().__init__()
        self.C = C  
        self.M = M  


        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        
        stem = nn.Sequential(
            nn.Conv2d(C, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        
        downsample_layer = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        self.downsample_layers.append(downsample_layer)

        # ConvNeXt Blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(2):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    
        self.output_layer = nn.Sequential(
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[-1], M, kernel_size=1)
        )

        
        self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=1) for i in range(2)])
        #self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=3) for i in range(2)])
        #self.af_modules = nn.ModuleList([CBAM_AFModule(N=dims[i], num_dim=2) for i in range(2)])
        self.attn_blocks = nn.ModuleList([AttentionBlock(dims[i]) for i in range(2)])

        self.apply(self._init_weights)

    #def _init_weights(self, m):
    #    if isinstance(m, (nn.Conv2d, nn.Linear)):
    #        trunc_normal_(m.weight, std=.02)
    #        nn.init.constant_(m.bias, 0)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:  
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for i in range(2):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.af_modules[i]((x, snr)) if snr is not None else x
            x = self.attn_blocks[i](x)

        x = self.output_layer(x)
        return x

class DeepJSCCDecoder(nn.Module):
    def __init__(self, N=256, M=16, C=3, depths=[3, 3], dims=[192, 96], drop_path_rate=0.1):
        super().__init__()
        self.C = C  
        self.M = M  

        
        self.input_layer = nn.Sequential(
            nn.Conv2d(M, dims[0], kernel_size=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        
        self.upsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        
        self.upsample_layers.append(UpsampleBlock(dims[0], dims[1], upsample_factor=2))
        
        self.upsample_layers.append(UpsampleBlock(dims[1], dims[1], upsample_factor=2))

        # ConvNeXt Blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(2):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        
        self.output_layer = nn.Conv2d(dims[-1], C, kernel_size=1)

        
        self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=1) for i in range(2)])
        #self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=3) for i in range(2)])
        #self.af_modules = nn.ModuleList([CBAM_AFModule(N=dims[i], num_dim=2) for i in range(2)])
        self.attn_blocks = nn.ModuleList([AttentionBlock(dims[i]) for i in range(2)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        x = self.input_layer(x)
        for i in range(2):
            x = self.stages[i](x)
            x = self.af_modules[i]((x, snr)) if snr is not None else x
            x = self.attn_blocks[i](x)
            x = self.upsample_layers[i](x)

        x = self.output_layer(x)
        return x
'''


# DeepJSCC Encoder Decoder for FFHQ-256
class DeepJSCCEncoder(nn.Module):
    def __init__(self, N=256, C=3, M=32, depths=[2, 2, 6, 2], dims=[96, 192, 384, 768], drop_path_rate=0.1):
        super().__init__()
        self.C = C  
        self.M = M  

        # Downsampling layers and stages
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Stem: first downsampling (256x256 -> 128x128)
        stem = nn.Sequential(
            nn.Conv2d(C, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Subsequent downsampling (128x128 -> 64x64 -> 32x32 -> 16x16)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # ConvNeXt Blocks (reference convnext_tiny depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Output layer: adjust to (batch, 32, 16, 16)
        self.output_layer = nn.Sequential(
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[-1], M, kernel_size=1)
        )

        # Optional AFModule and AttentionBlock
        #self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=1) for i in range(4)])
        self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=3) for i in range(4)])
        self.attn_blocks = nn.ModuleList([AttentionBlock(dims[i]) for i in range(4)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.af_modules[i]((x, snr)) if snr is not None else x
            x = self.attn_blocks[i](x)

        x = self.output_layer(x)
        return x


class DeepJSCCDecoder(nn.Module):
    def __init__(self, N=256, M=32, C=3, depths=[2, 6, 2, 2], dims=[768, 384, 192, 96], drop_path_rate=0.1):
        super().__init__()
        self.C = C  
        self.M = M  

        # Input layer: adjust from M to first stage dimension
        self.input_layer = nn.Sequential(
            nn.Conv2d(M, dims[0], kernel_size=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        # Upsampling layers and stages
        self.upsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Upsampling (16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256)
        for i in range(3):
            upsample_layer = UpsampleBlock(dims[i], dims[i + 1], upsample_factor=2)
            self.upsample_layers.append(upsample_layer)
        self.upsample_layers.append(UpsampleBlock(dims[-1], dims[-1], upsample_factor=2))

        # ConvNeXt Blocks (reference convnext_tiny depths, reverse order)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Output layer: adjust to target channels C
        self.output_layer = nn.Conv2d(dims[-1], C, kernel_size=1)

        # Optional AFModule and AttentionBlock
        #self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=1) for i in range(4)])
        self.af_modules = nn.ModuleList([AFModule(N=dims[i], num_dim=3) for i in range(4)])
        self.attn_blocks = nn.ModuleList([AttentionBlock(dims[i]) for i in range(4)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        x = self.input_layer(x)
        for i in range(4):
            x = self.stages[i](x)
            x = self.af_modules[i]((x, snr)) if snr is not None else x
            x = self.attn_blocks[i](x)
            x = self.upsample_layers[i](x)

        x = self.output_layer(x)
        return x


#encoder = DeepJSCCQ2Encoder()
#x = torch.randn(1, 3, 32, 32)
#csi = torch.ones(1, 1)
#latent = encoder((x, csi))
#print("latent shape:", latent.shape)

#decoder = DeepJSCCQ2Decoder()
#x = torch.randn(32, 16, 8, 8)

