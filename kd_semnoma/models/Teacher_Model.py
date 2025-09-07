import torch
import torch.nn as nn
from models.channel import ComplexAWGNMAC, ComplexRayleighOrthogonalChannel    # load channel model
from models.power_constraint import ComplexAveragePowerConstraint
#from models.autoencoder import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder
from models.autoencoder_modify import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder


# Teacher Model for AWGN Channel
class Teacher_Model_AWGN(nn.Module):
    def __init__(self, num_devices, M=16):
        super().__init__()


        self.num_devices = num_devices
        self.M = M

        self.channel = ComplexAWGNMAC()
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        self.encoders = nn.ModuleList([DeepJSCCQ2Encoder(N=256, M=self.M, C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([DeepJSCCQ2Decoder(N=256, M=self.M, C=3) for _ in range(1)])

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=256 * 256
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")

    def forward(self, image, csi):
        """
        Forward pass with Perfect SIC logic, adapted for orthogonal transmission.
        Args:
            image (torch.Tensor): Input images, shape (batch_size, num_devices, 3, H, W)
            csi (torch.Tensor): Channel state information, shape (batch_size, 1)
        Returns:
            torch.Tensor: Reconstructed images, shape (batch_size, num_devices, 3, H, W)
            torch.Tensor: Teacher features, shape (batch_size, num_devices, M, H', W')
        """
        x, csi = image, csi

        emb = torch.stack(
            [
                self.device_images(
                    torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i
                ).view(x.size(0), 1, 256, 256)
                for i in range(self.num_devices)
            ],
            dim=1,
        )
        x = torch.cat([x, emb], dim=2)

        transmissions = []

        # Device-wise encoding and channel transmission (orthogonal transmission)
        for i in range(self.num_devices):
            # Encoding
            t = self.encoders[0]((x[:, i, ...], csi))  # (batch, M, H', W'), e.g., (batch, 16, 8, 8)

            # Power constraint (device-wise adjustment)
            t = self.power_constraint(
                t[:, None, ...],  # Add device dimension (batch, 1, M, H', W')
                mult=torch.sqrt(torch.tensor(1, dtype=t.dtype, device=t.device))
            )  # (batch, 1, M, H', W')
            t = t.squeeze(1)  # Remove device dimension (batch, M, H', W')

            # Channel transmission (independent noise addition)
            if isinstance(self.channel, ComplexAWGNMAC):
                # Complex AWGN noise, multiply by 0.5 for complex channel variance allocation
                awgn = torch.randn_like(t) * torch.sqrt(10.0 ** (-csi[..., None, None] / 10.0)) * torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))
            else:
                # Real AWGN noise
                awgn = torch.randn_like(t) * torch.sqrt(10.0 ** (-csi[..., None, None] / 10.0))
            t = t + awgn  # Each device transmits independently with independent noise

            transmissions.append(t)

        # Decoding (no need for unbinding, directly use transmission results)
        results = []
        feat_teacher = []
        for i in range(self.num_devices):
            # Intermediate features
            x_unbound = transmissions[i]  # (batch, M, H', W')
            feat_teacher.append(x_unbound[:, None, ...])  # (batch, 1, M, H', W'), maintain dimension consistency

            # Decode single device image
            r = self.decoders[0]((x_unbound, csi))  # (batch, 3, H, W)
            results.append(r)

        # Stack all device features
        teacher_feat = torch.cat(feat_teacher, dim=1)  # (batch, num_devices, M, H', W')
        # Stack all device results
        x = torch.stack(results, dim=1)  # (batch, num_devices, 3, H, W)

        return x, teacher_feat


# Teacher Model for Rayleigh Channel
class Teacher_Model_Rayleigh(nn.Module):
    def __init__(self, num_devices, M=16):
        super().__init__()
        self.num_devices = num_devices
        self.M = M

        # Channel and power constraint modules
        self.channel = ComplexRayleighOrthogonalChannel()  # Use independent Rayleigh channel
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        # Encoders and decoders
        self.encoders = nn.ModuleList([DeepJSCCQ2Encoder(N=256, M=self.M, C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([DeepJSCCQ2Decoder(N=256, M=self.M, C=3) for _ in range(1)])

        # Device embedding
        self.device_images = nn.Embedding(self.num_devices, embedding_dim=256 * 256)

    def forward_decoder_with_intermediate(self, decoder, x, side_info):
        """Decompose decoder and return all intermediate features"""
        intermediate_feats = []

        x = decoder.input_layer(x)
        intermediate_feats.append(x)

        for i in range(4):
            x = decoder.stages[i](x)
            intermediate_feats.append(x)

            x = decoder.af_modules[i]((x, side_info)) if side_info is not None else x
            intermediate_feats.append(x)

            x = decoder.attn_blocks[i](x)
            intermediate_feats.append(x)

            x = decoder.upsample_layers[i](x)
            intermediate_feats.append(x)

        x = decoder.output_layer(x)
        intermediate_feats.append(x)

        return intermediate_feats

    def forward(self, image, csi):
        """
        Forward pass with Perfect SIC logic, adapted for independent Rayleigh channels with adaptive fading.
        Args:
            image (torch.Tensor): Input images, shape (batch_size, num_devices, 3, H, W)
            csi (torch.Tensor): Channel state information (SNR), shape (batch_size, 1)
        Returns:
            torch.Tensor: Reconstructed images, shape (batch_size, num_devices, 3, H, W)
            torch.Tensor: Teacher features (after channel), shape (batch_size, num_devices, M, H', W')
            torch.Tensor: Teacher transmissions (before channel), shape (batch_size, num_devices, M, H', W')
            list: Teacher decoder intermediate features for each device, list of tensors
        """
        x, snr = image, csi  # x: (B, Devices, C, H, W), snr: (B, 1)

        # Embed device identifiers
        emb = torch.stack(
            [
                self.device_images(
                    torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i
                ).view(x.size(0), 1, 256, 256)
                for i in range(self.num_devices)
            ],
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
        h_phase = torch.angle(h)  # (B, D)

        # Construct side_info
        snr_expanded = snr.expand(-1, D)  # (B, D)
        side_info = torch.cat([snr_expanded, h_magnitude, h_phase], dim=1)  # (B, 3*D)

        # Encoding and power constraint
        transmissions = []
        for i in range(self.num_devices):
            device_side_info = side_info[:, [i, i + D, i + 2 * D]]  # (B, 3)
            t = self.encoders[0]((x[:, i, ...], device_side_info))  # (B, M, H', W')
            t = self.power_constraint(
                t[:, None, ...],  # (B, 1, M, H', W')
                mult=torch.sqrt(torch.tensor(1, dtype=t.dtype, device=t.device))
            )  # (B, 1, M, H', W')
            t = t.squeeze(1)  # (B, M, H', W')
            transmissions.append(t)

        teacher_transmissions = torch.stack(transmissions, dim=1)  # (B, D, M, H', W')

        # Device-wise independent Rayleigh channel transmission
        x = self.channel((teacher_transmissions, snr, h))  # (B, D, M, H', W')

        # Decoding
        results = []
        feat_teacher = []
        teacher_decoder_feats = []
        for i in range(self.num_devices):
            x_unbound = x[:, i, ...]  # (B, M, H', W')
            feat_teacher.append(x_unbound[:, None, ...])  # (B, 1, M, H', W')
            device_side_info = side_info[:, [i, i + D, i + 2 * D]]  # (B, 3)

            # Get decoder intermediate features
            decoder_feats = self.forward_decoder_with_intermediate(
                self.decoders[0], x_unbound, device_side_info
            )
            teacher_decoder_feats.append(decoder_feats)
            r = decoder_feats[-1]  # (B, 3, H, W)
            results.append(r)

        teacher_feat = torch.cat(feat_teacher, dim=1)  # (B, D, M, H', W')
        x = torch.stack(results, dim=1)  # (B, D, 3, H, W)

        return x, teacher_feat
