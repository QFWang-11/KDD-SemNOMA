import torch
import torch.nn as nn
from models.channel import ComplexAWGNMAC, ComplexRayleighMAC
from models.power_constraint import ComplexAveragePowerConstraint
#from models.autoencoder import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder          # baseline
from models.autoencoder_convnext import DeepJSCCEncoder, DeepJSCCDecoder    # Convnext based method


# Convnext
class SemNOMA_Rayleigh(nn.Module):
    def __init__(self, num_devices, M):
        super().__init__()
        self.num_devices = num_devices
        self.M = M

        self.channel = ComplexRayleighMAC()
        #self.channel = ComplexAWGNMAC()
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        self.encoders = nn.ModuleList([DeepJSCCEncoder(N=256, M=self.M, C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([DeepJSCCDecoder(N=256, M=self.M, C=3) for _ in range(2)])

        self.device_images = nn.Embedding(self.num_devices, embedding_dim=256 * 256)

    def forward(self, image, csi):
        x, snr = image, csi  # x: (B, Devices, C, H, W), snr: (B, 1)

        # Embed device identifiers
        emb = torch.stack(
            [self.device_images(torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i).view(x.size(0), 1, 256, 256)
             for i in range(self.num_devices)],
            dim=1,
        )
        x = torch.cat([x, emb], dim=2)  # (B, Devices, C+1, H, W)

        # Generate Rayleigh fading coefficients for each device
        B = x.shape[0]
        D = self.num_devices
        h_real = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h = torch.complex(h_real, h_imag)  # (B, D)
        h_magnitude = torch.abs(h)  # (B, D)
        h_phase = torch.angle(h)  # (B, D), phase information
        

        # Construct side_info
        snr_expanded = snr.expand(-1, D)  # (B, D), replicate global SNR for each device
        side_info = torch.cat([snr_expanded, h_magnitude, h_phase], dim=1)  # (B, 3*D), [SNR_1, h_mag_1, h_phase_1, ...]

        # Encoding
        transmissions = []
        for i in range(self.num_devices):
            device_side_info = side_info[:, [i, i + D, i + 2*D]] # (B, 3), [SNR, h_magnitude, h_phase] for device i
            #t = self.encoders[0]((x[:, i, ...], side_info))  # All information for adaptive Rayleigh fading channel
            t = self.encoders[0]((x[:, i, ...], device_side_info))   # Multi-encoder test
            transmissions.append(t)

        # Channel processing
        x = torch.stack(transmissions, dim=1)  # (B, D, C, H, W)
        x = self.power_constraint(x)
        noisy_signal = self.channel((x, snr, h))  # (B, C, H, W)

        # Decoding
        receives = []
        for i in range(self.num_devices):
            #r = self.decoders[i]((noisy_signal, side_info))   # All information for adaptive Rayleigh fading
            r = self.decoders[i]((noisy_signal, side_info[:, [i, i + D, i + 2*D]]))  # (B, 3)
            receives.append(r)
        x = torch.stack(receives, dim=1)  # (B, D, C, H, W)

        return x

