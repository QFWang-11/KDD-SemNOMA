import torch
import torch.nn as nn
from models.layers import *
from models.afmodule import AFModule

class DeepJSCCQ2Encoder(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(      # idx=0
                in_ch=C,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),     # idx=1
            ResidualBlock(                # idx=2
                in_ch=N,
                out_ch=N),
            ResidualBlock(                # idx=3
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),     # idx=4
            AttentionBlock(N),            # idx=5
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=M),
            AFModule(N=M, num_dim=1),
            AttentionBlock(M),            # idx=12
        ]) 
    
    
    def forward(self, x):        # Encoder input is the concatenation of image and SNR
        
        if isinstance(x, tuple): # x includes the concatenation of SNR and image data
            x, snr = x  # x:(batch, 4, H, W)
            #print(x.type())
            #print(snr.type())
        else:
            snr = None

        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, snr))   # Introduce attention mechanism, embed SNR information into image data
            else:
                x = layer(x)          # Other layers input dimension (batch, 3, H, W)

        return x

class DeepJSCCQ2Decoder(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            AFModule(N=C, num_dim=1),
        ])


    def forward(self, x):
        
        if isinstance(x, tuple):
            x, snr = x
            #x = x.cuda()
            #snr = snr.cuda()
        else:
            snr = None
        
        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x

'''
# Test encoder
encoder = DeepJSCCQ2Encoder(N=256, M=16, C=3)
x = torch.randn(32, 3, 32, 32)
csi = torch.ones(32, 1)
for idx, layer in enumerate(encoder.g_a):
    if isinstance(layer, AFModule):
        x = layer((x, csi))
    else:
        x = layer(x)
    print(f"enc_layer_{idx} shape:", x.shape)

# Test decoder
decoder = DeepJSCCQ2Decoder(N=256, M=16, C=3)
x = torch.randn(32, 16, 8, 8)
for idx, layer in enumerate(decoder.g_s):
    if isinstance(layer, AFModule):
        x = layer((x, csi))
    else:
        x = layer(x)
    print(f"dec_layer_{idx} shape:", x.shape)'
'''