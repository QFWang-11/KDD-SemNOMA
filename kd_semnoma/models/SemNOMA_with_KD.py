import os
import torch
import torch.nn as nn
from models.channel import ComplexAWGNMAC, ComplexRayleighMAC
from models.afmodule import AFModule
from models.power_constraint import ComplexAveragePowerConstraint
#from models.autoencoder import DeepJSCCQ2Encoder, DeepJSCCQ2Decoder
from models.autoencoder_convnext import DeepJSCCEncoder, DeepJSCCDecoder
from models.kd_loss import spatial_similarity, AT, FSP
from models.Teacher_Model import Teacher_Model_AWGN, Teacher_Model_Rayleigh
from models.losses import MSE

# kd loss: sa(mse) + cross kd
class SemNOMA_KD_AWGN(nn.Module):
    def __init__(self, num_devices, M, teacher_checkpoint_path=None, cross_layer_idx=0):
        super().__init__()

        self.num_devices = num_devices
        self.M = M
        self.cross_layer_idx = cross_layer_idx  # Choose which layer of student decoder for CrossKD

        self.spatial_similarity = spatial_similarity
        self.AT = AT
        self.FSP = FSP

        self.mse_loss = MSE()
        self.mae_loss = nn.L1Loss()

        # Instantiate and load pretrained teacher_head
        self.teacher_head = Teacher_Model_AWGN(num_devices=2, M=self.M)   
        if teacher_checkpoint_path and os.path.exists(teacher_checkpoint_path):
            print(f"Loading teacher_head pretrained model from {teacher_checkpoint_path}...")
            checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')
            self.teacher_head.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")

        # Freeze all parameters of teacher_head
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.teacher_head.eval()

        self.channel = ComplexAWGNMAC()
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        self.encoders = nn.ModuleList([DeepJSCCEncoder(N=256, M=self.M, C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([DeepJSCCDecoder(N=256, M=self.M, C=3) for _ in range(self.num_devices)])

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=256 * 256
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")

        self.feat_proj_layers = nn.ModuleList([
            nn.Conv2d(16, 16, kernel_size=1, bias=False),  # Adjust channel numbers based on DeepJSCCQ2Decoder layers
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            # Add more projection layers based on actual layer count and channel numbers
        ])

        # Initialize student model using teacher_head pretrained parameters
        self._initialize_student_from_teacher()

    def _initialize_student_from_teacher(self):
        """Initialize student model using teacher model parameters, including two decoders"""
        teacher_dict = self.teacher_head.state_dict()
        student_dict = self.state_dict()

        # Create mapping table to map teacher model parameters to student model
        mapping = {
            'encoders.0': ['encoders.0'],  # Encoder maps to student model's encoders.0
            'decoders.0': ['decoders.0', 'decoders.1'],  # Teacher decoder maps to student's two decoders
            'device_images': ['device_images'],     # Teacher's device_images maps to student's device_images
        }

        # Iterate through mapping table for parameter initialization
        for t_prefix, s_prefixes in mapping.items():
            for t_key, t_value in teacher_dict.items():
                if t_key.startswith(t_prefix):
                    for s_prefix in s_prefixes:  # Process each target prefix in student model
                        s_key = t_key.replace(t_prefix, s_prefix, 1)
                        if s_key in student_dict and t_value.shape == student_dict[s_key].shape:
                            student_dict[s_key].copy_(t_value)
                            print(f"Initialized {s_key} from {t_key}")
                        elif s_key in student_dict:
                            print(f"Shape mismatch: {t_key} ({t_value.shape}) vs {s_key} ({student_dict[s_key].shape})")
                        else:
                            print(f"Key {s_key} not found in student model")

        # Load updated parameters into student model
        self.load_state_dict(student_dict)

    def inference(self, image, csi):
        """
        Inference method using only student branch
        Args:
            image: Input image tensor (batch_size, num_devices, channels, height, width)
            csi: Channel state information
        Returns:
            reconstructed images from student branch
        """
        x = image

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
        
        # Student branch inference
        transmissions = []
        for i in range(self.num_devices):
            t = self.encoders[0]((x[:, i, ...], csi))
            transmissions.append(t)

        # Stack transmissions and apply power constraint
        x = torch.stack(transmissions, dim=1)
        
        
        # Apply power constraint and channel
        x = self.power_constraint(x)
        x = self.channel((x, csi))
        
        
        # Pass through decoder
        receives = []
        for i in range(self.num_devices):
            r = self.decoders[i]((x, csi))
            receives.append(r)
        
        # Stack the received signals
        x = torch.stack(receives, dim=1)
        
        return x  # Return reconstructed images

    def forward_decoder_with_intermediate(self, decoder, x, csi):
        """Decompose decoder and return all intermediate features"""
        intermediate_feats = []

        # Input layer
        x = decoder.input_layer(x)
        intermediate_feats.append(x)

        # Stages and upsampling
        for i in range(4):
            x = decoder.stages[i](x)
            intermediate_feats.append(x)

            x = decoder.af_modules[i]((x, csi)) if csi is not None else x
            intermediate_feats.append(x)

            x = decoder.attn_blocks[i](x)
            intermediate_feats.append(x)

            x = decoder.upsample_layers[i](x)
            intermediate_feats.append(x)

        # Output layer
        x = decoder.output_layer(x)
        intermediate_feats.append(x)

        return intermediate_feats

    def forward_teacher_decoder_from_idx(self, decoder, x, csi, start_idx):
        """Forward propagation from specified layer of teacher decoder"""
        # Define all layers in order and corresponding processing
        layers = [
            decoder.input_layer,           # 0
            decoder.stages[0],             # 1
            lambda x: decoder.af_modules[0]((x, csi)) if csi is not None else x,  # 2
            decoder.attn_blocks[0],        # 3
            decoder.upsample_layers[0],    # 4
            decoder.stages[1],             # 5
            lambda x: decoder.af_modules[1]((x, csi)) if csi is not None else x,  # 6
            decoder.attn_blocks[1],        # 7
            decoder.upsample_layers[1],    # 8
            decoder.stages[2],             # 9
            lambda x: decoder.af_modules[2]((x, csi)) if csi is not None else x,  # 10
            decoder.attn_blocks[2],        # 11
            decoder.upsample_layers[2],    # 12
            decoder.stages[3],             # 13
            lambda x: decoder.af_modules[3]((x, csi)) if csi is not None else x,  # 14
            decoder.attn_blocks[3],        # 15
            decoder.upsample_layers[3],    # 16
            decoder.output_layer           # 17
        ]

        # Execute subsequent layers starting from start_idx+1
        for layer in layers[start_idx+1:]:
            x = layer(x)
        return x

    def forward(self, image, csi):
        if not self.training:
            return self.inference(image, csi)

        x, csi = image, csi

        # Teacher branch
        teacher_x, teacher_csi = x.clone().detach(), csi.clone().detach()
        teacher_out, teacher_feat = self.teacher_head(teacher_x, teacher_csi)  # (batch, num_devices, 3, 32, 32), (batch, num_devices, 16, 8, 8)

        # Student branch
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

        x = torch.stack(transmissions, dim=1)  # (batch, num_devices, 16, 8, 8)
        x = self.power_constraint(x)
        x = self.channel((x, csi))

        # multiuser features
        X = []
        for i in range(self.num_devices):
            X.append(x)
        x = torch.cat(X, dim=1)        #(batch,num_devices*16,8,8)
        x = x.view(x.size(0), self.num_devices, self.M, x.size(2), x.size(3))   #(batch,2,16,8,8)
        
        student_feat = x

        # Get intermediate features from student decoder
        #student_decoder = self.decoders[0]
        pseudo_teacher_out = []
        for i in range(self.num_devices):
            # Get all intermediate features
            intermediate_feats = self.forward_decoder_with_intermediate(self.decoders[i], student_feat[:,i,...], csi)
            student_out = intermediate_feats[-1]  # Student final output

            # Select intermediate features from specified layer for CrossKD
            fs_i = intermediate_feats[self.cross_layer_idx]  # Features output from decoder layer idx, need to feed to teacher_head layer idx+1 as input
            
            # Project features to match teacher decoder input (adjust based on actual channel numbers)
            #fs_i_proj = self.feat_proj_layers[self.cross_layer_idx](fs_i)
            fs_i_proj = fs_i  # For M=64 case, layer 8 features are the same for teacher and student

            # Input to corresponding subsequent layers of teacher decoder
            teacher_decoder = self.teacher_head.decoders[0]
            pseudo_out = self.forward_teacher_decoder_from_idx(teacher_decoder, fs_i_proj, csi, self.cross_layer_idx)
            pseudo_teacher_out.append(pseudo_out)
        
        pseudo_teacher_out = torch.stack(pseudo_teacher_out, dim=1)  # (batch, num_devices, 3, 32, 32)
        student_out = torch.stack([self.decoders[i]((student_feat[:,i,...], csi)) for i in range(self.num_devices)], dim=1)

        # Distillation loss
        loss_dict = {}

        # 1. Spatial similarity loss (kd_spatial)
        loss_kd_spatial = 0
        for i in range(self.num_devices):
            teacher_sim = self.spatial_similarity(teacher_feat[:, i, ...])  # (batch, 64, 64)
            student_sim = self.spatial_similarity(student_feat[:, i, ...])  # (batch, 64, 64)
            loss_kd_spatial += self.mae_loss(teacher_sim, student_sim) / (256 * 256)    # (batch,32,16,16) after SA calculation becomes 256
        loss_dict['kd_spatial'] = 100 * loss_kd_spatial

        # 2. CrossKD loss (kd_cross)
        loss_kd_cross = 0
        for i in range(self.num_devices):
            loss_kd_cross += self.mae_loss(pseudo_teacher_out[:, i, ...], teacher_out[:, i, ...])
        loss_dict['kd_cross'] = 1 * loss_kd_cross  # Weight adjustable

        return student_out, loss_dict


class SemNOMA_KD_Rayleigh(nn.Module):
    def __init__(self, num_devices, M, teacher_checkpoint_path=None, cross_layer_idx=0):
        super().__init__()

        self.num_devices = num_devices
        self.M = M
        self.cross_layer_idx = cross_layer_idx  # Choose which layer of student decoder for CrossKD

        self.spatial_similarity = spatial_similarity
        self.AT = AT
        self.FSP = FSP

        self.mse_loss = MSE()
        self.mae_loss = nn.L1Loss()

        # Instantiate and load pretrained teacher_head
        self.teacher_head = Teacher_Model_Rayleigh(num_devices=self.num_devices, M=self.M)   
        if teacher_checkpoint_path and os.path.exists(teacher_checkpoint_path):
            print(f"Loading teacher_head pretrained model from {teacher_checkpoint_path}...")
            checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')
            self.teacher_head.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")

        # Freeze all parameters of teacher_head
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        self.teacher_head.eval()

        self.channel = ComplexRayleighMAC()
        self.power_constraint = ComplexAveragePowerConstraint(power=1, num_devices=self.num_devices)

        self.encoders = nn.ModuleList([DeepJSCCEncoder(N=256, M=self.M, C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([DeepJSCCDecoder(N=256, M=self.M, C=3) for _ in range(self.num_devices)])

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=256 * 256
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")


        self.feat_proj_layers = nn.ModuleList([
            nn.Conv2d(16, 16, kernel_size=1, bias=False),  # Adjust channel numbers based on DeepJSCCQ2Decoder layers
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            # Add more projection layers based on actual layer count and channel numbers
        ])

        # Initialize student model using teacher_head pretrained parameters
        self._initialize_student_from_teacher()

    def _initialize_student_from_teacher(self):
        """Initialize student model using teacher model parameters, including two decoders"""
        teacher_dict = self.teacher_head.state_dict()
        student_dict = self.state_dict()

        # Create mapping table to map teacher model parameters to student model
        mapping = {
            'encoders.0': ['encoders.0'],  # Encoder maps to student model's encoders.0
            'decoders.0': ['decoders.0', 'decoders.1'],  # Teacher decoder maps to student's two decoders
            'device_images': ['device_images'],     # Teacher's device_images maps to student's device_images
        }

        # Iterate through mapping table for parameter initialization
        for t_prefix, s_prefixes in mapping.items():
            for t_key, t_value in teacher_dict.items():
                if t_key.startswith(t_prefix):
                    for s_prefix in s_prefixes:  # Process each target prefix in student model
                        s_key = t_key.replace(t_prefix, s_prefix, 1)
                        if s_key in student_dict and t_value.shape == student_dict[s_key].shape:
                            student_dict[s_key].copy_(t_value)
                            print(f"Initialized {s_key} from {t_key}")
                        elif s_key in student_dict:
                            print(f"Shape mismatch: {t_key} ({t_value.shape}) vs {s_key} ({student_dict[s_key].shape})")
                        else:
                            print(f"Key {s_key} not found in student model")

        # Load updated parameters into student model
        self.load_state_dict(student_dict)

    def inference(self, image, csi):
        """
        Inference method using only student branch
        Args:
            image: Input image tensor (batch_size, num_devices, channels, height, width)
            csi: Channel state information
        Returns:
            reconstructed images from student branch
        """
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
            device_side_info = side_info[:, [i, i + D, i + 2*D]] # (B, 2), [SNR, h_magnitude] for device i
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
            r = self.decoders[i]((noisy_signal, side_info[:, [i, i + D, i + 2*D]]))  # (B, 2)
            receives.append(r)
        x = torch.stack(receives, dim=1)  # (B, D, C, H, W)

        return x

    def forward_decoder_with_intermediate(self, decoder, x, csi):
        """Decompose decoder and return all intermediate features"""
        intermediate_feats = []

        # Input layer
        x = decoder.input_layer(x)
        intermediate_feats.append(x)

        # Stages and upsampling
        for i in range(4):
            x = decoder.stages[i](x)
            intermediate_feats.append(x)

            x = decoder.af_modules[i]((x, csi)) if csi is not None else x
            intermediate_feats.append(x)

            x = decoder.attn_blocks[i](x)
            intermediate_feats.append(x)

            x = decoder.upsample_layers[i](x)
            intermediate_feats.append(x)

        # Output layer
        x = decoder.output_layer(x)
        intermediate_feats.append(x)

        return intermediate_feats

    def forward_teacher_decoder_from_idx(self, decoder, x, csi, start_idx):
        """Forward propagation from specified layer of teacher decoder"""
        # Define all layers in order and corresponding processing
        layers = [
            decoder.input_layer,           # 0
            decoder.stages[0],             # 1
            lambda x: decoder.af_modules[0]((x, csi)) if csi is not None else x,  # 2
            decoder.attn_blocks[0],        # 3
            decoder.upsample_layers[0],    # 4
            decoder.stages[1],             # 5
            lambda x: decoder.af_modules[1]((x, csi)) if csi is not None else x,  # 6
            decoder.attn_blocks[1],        # 7
            decoder.upsample_layers[1],    # 8
            decoder.stages[2],             # 9
            lambda x: decoder.af_modules[2]((x, csi)) if csi is not None else x,  # 10
            decoder.attn_blocks[2],        # 11
            decoder.upsample_layers[2],    # 12
            decoder.stages[3],             # 13
            lambda x: decoder.af_modules[3]((x, csi)) if csi is not None else x,  # 14
            decoder.attn_blocks[3],        # 15
            decoder.upsample_layers[3],    # 16
            decoder.output_layer           # 17
        ]

        # Execute subsequent layers starting from start_idx+1
        for layer in layers[start_idx+1:]:
            x = layer(x)
        return x

    def forward(self, image, csi):
        if not self.training:
            return self.inference(image, csi)

        x, csi = image, csi

        # Generate Rayleigh fading coefficients
        B, D = x.shape[0], self.num_devices
        h_real = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(B, D, device=x.device) * (0.5 ** 0.5)
        h = torch.complex(h_real, h_imag)  # (B, D)
        h_magnitude = torch.abs(h)  # (B, D)
        h_phase = torch.angle(h)  # (B, D)

        # Construct side_info
        snr_expanded = csi.expand(-1, D)  # (B, D)
        side_info = torch.cat([snr_expanded, h_magnitude, h_phase], dim=1)  # (B, 3*D)

        # Teacher branch
        teacher_x, teacher_csi = x.clone().detach(), csi.clone().detach()
        teacher_out, teacher_feat = self.teacher_head(teacher_x, teacher_csi)  # (batch, num_devices, 3, 32, 32), (batch, num_devices, 16, 8, 8)

        # Student branch
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
            device_side_info = side_info[:, [i, i + D, i + 2*D]]  # (B, 3)
            t = self.encoders[0]((x[:, i, ...], device_side_info))
            transmissions.append(t)

        x = torch.stack(transmissions, dim=1)  # (batch, num_devices, 16, 8, 8)
        x = self.power_constraint(x)
        x = self.channel((x, csi, h))

        # multiuser features
        X = []
        for i in range(self.num_devices):
            X.append(x)
        x = torch.cat(X, dim=1)        #(batch,num_devices*16,8,8)
        x = x.view(x.size(0), self.num_devices, self.M, x.size(2), x.size(3))   #(batch,2,16,8,8)
        
        student_feat = x

        # Get intermediate features from student decoder
        #student_decoder = self.decoders[0]
        pseudo_teacher_out = []
        for i in range(self.num_devices):
            # Get all intermediate features
            intermediate_feats = self.forward_decoder_with_intermediate(self.decoders[i], student_feat[:,i,...], side_info[:, [i, i + D, i + 2*D]])
            student_out = intermediate_feats[-1]  # Student final output

            # Select intermediate features from specified layer for CrossKD
            fs_i = intermediate_feats[self.cross_layer_idx]
            
            # Project features to match teacher decoder input (adjust based on actual channel numbers)
            #fs_i_proj = self.feat_proj_layers[self.cross_layer_idx](fs_i)
            fs_i_proj = fs_i

            # Input to corresponding subsequent layers of teacher decoder
            teacher_decoder = self.teacher_head.decoders[0]
            pseudo_out = self.forward_teacher_decoder_from_idx(teacher_decoder, fs_i_proj, side_info[:, [i, i + D, i + 2*D]], self.cross_layer_idx)
            pseudo_teacher_out.append(pseudo_out)
        
        pseudo_teacher_out = torch.stack(pseudo_teacher_out, dim=1)  # (batch, num_devices, 3, 32, 32)
        student_out = torch.stack([self.decoders[i]((student_feat[:,i,...], side_info[:, [i, i + D, i + 2*D]])) for i in range(self.num_devices)], dim=1)

        # Distillation loss
        loss_dict = {}

        # 1. Spatial similarity loss (kd_spatial)
        loss_kd_spatial = 0
        for i in range(self.num_devices):
            teacher_sim = self.spatial_similarity(teacher_feat[:, i, ...])  # (batch, 64, 64)
            student_sim = self.spatial_similarity(student_feat[:, i, ...])  # (batch, 64, 64)
            loss_kd_spatial += self.mae_loss(teacher_sim, student_sim) / (256 * 256)
        loss_dict['kd_spatial'] = 100 * loss_kd_spatial

        # 2. CrossKD loss (kd_cross)
        loss_kd_cross = 0
        for i in range(self.num_devices):
            loss_kd_cross += self.mae_loss(pseudo_teacher_out[:, i, ...], teacher_out[:, i, ...])
        loss_dict['kd_cross'] = 1 * loss_kd_cross  # Weight adjustable

        return student_out, loss_dict

