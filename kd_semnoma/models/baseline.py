import torch
import torch.nn as nn

from models.channel import ComplexAWGNMAC, ComplexRayleighMAC_for_baseline
from models.power_constraint import ComplexAveragePowerConstraint
from models.autoencoder import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder

class SingleModelNet(nn.Module):
    def __init__(
        self, num_devices, M, ckpt_path=None):
        super().__init__()

        self.M = M
        self.num_devices = num_devices

        #self.channel = ComplexAWGNMAC()
        self.channel = ComplexRayleighMAC_for_baseline()
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        self.encoders = nn.ModuleList([DeepJSCCQ2Encoder(N=256, M=self.M, C=4) for _ in range(1)])  
        self.decoders = nn.ModuleList([DeepJSCCQ2Decoder(N=256, M=self.M, C=6) for _ in range(1)])

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=256 * 256
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)["state_dict"]

            enc_state_dict = {}
            dec_state_dict = {}
            images_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith("net.encoders.0"):
                    enc_state_dict[k.replace("net.encoders.0", "0")] = v
                elif k.startswith("net.decoders.0"):
                    dec_state_dict[k.replace("net.decoders.0", "0")] = v
                elif k.startswith("net.device_images."):
                    images_state_dict[k.replace("net.device_images.", "")] = v

            self.encoders.load_state_dict(enc_state_dict)
            self.decoders.load_state_dict(dec_state_dict)
            self.device_images.load_state_dict(images_state_dict)
            print("checkpoint loaded")

    def forward(self, image, csi):
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
        for i in range(self.num_devices):
            t = self.encoders[0]((x[:, i, ...], csi))
            transmissions.append(t)

        x = torch.stack(transmissions, dim=1)

        x = self.power_constraint(x)
        x = self.channel((x, csi))

        x = self.decoders[0]((x, csi))
        x = x.view(x.size(0), self.num_devices, 3, x.size(2), x.size(3))

        return x