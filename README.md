# Knowledge Distillation Driven Semantic NOMA for Image Transmission with Diffusion Model

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A research project on **Semantic Non-Orthogonal Multiple Access with Knowledge Distillation and Diffusion Model** for efficient multi-user semantic communication systems.

## 📖 Overview

This repository implements a novel semantic communication framework that combines **Knowledge Distillation (KD)** and **Diffusion Model** for Non-Orthogonal Multiple Access (NOMA) transmisstion. 

### Key Innovations
- 🧠 **Knowledge Distillation Framework**: Teacher-student architecture for semantic feature extraction
- 📡 **Semantic NOMA**: Non-orthogonal multiple access with semantic information
- 🔄 **Diffusion Model Integration**: Advanced image reconstruction using diffusion models
- 📊 **Multi-Channel Support**: AWGN and Rayleigh fading channel models

## 🏗️ Architecture Overview

The system architecture consists of:

1. **Semantic Encoder**: Extracts semantic features from input data
2. **Knowledge Distillation Module**: Transfers knowledge from teacher to student networks
3. **NOMA Transmitter**: Encodes multiple users' data using NOMA principles
4. **Channel Layer**: Simulates realistic wireless channel conditions
5. **NOMA Receiver**: Decodes multi-user signals with interference cancellation
6. **Semantic Decoder**: Reconstructs original data from semantic features
7. **Diffusion Model**: Refine the received images

## 📁 Project Structure

```
KDD-SemNOMA/
├── 📄 images/                         # Figures
│   ├── KD-SemNOMA.pdf               
│   └── KDD-SemNOMA.pdf              
├── 🧠 kd_semnoma/                  # Core KD-SemNOMA implementation
│   ├── models/                     # Neural network models
│   │   ├── SemNOMA.py              # Basic SemNOMA model
│   │   ├── SemNOMA_with_KD.py      # KD-enhanced SemNOMA
│   │   ├── Teacher_Model.py        # Teacher network
│   │   ├── autoencoder.py          # Autoencoder architectures
│   │   ├── kd_loss.py              # Knowledge distillation losses
│   │   ├── baseline.py             # Baseline models
│   │   ├── channel.py              # Channel modeling
│   │   ├── layers.py               # Custom neural layers
│   │   ├── losses.py               # Loss functions
│   │   └── power_constraint.py     # Power control utilities
│   ├── datasets/                   # Dataset processing
│   │   ├── cifar10_group_wrapper.py  # CIFAR-10 handler
│   │   └── ffhq256_group_wrapper.py  # FFHQ-256 handler
│   ├── train_baseline.py           # Baseline training
│   ├── train_semnoma.py            # SemNOMA training
│   ├── train_teacher_model.py      # Teacher model training
│   ├── train_student_model.py      # Student model training (KD)
│   └── test.py                     # Model testing
└── 🚀 kdd_semnoma/                 # Advanced KDD-SemNOMA
    ├── kdd_semnoma/                # Enhanced implementation
    │   ├── sampler.py              # Diffusion model sampling
    │   ├── calculate_metrics.py    # Performance metrics
    │   ├── inference_difface.py    # inference
    │   └── basicsr/                # Image quality metrics
    └── semnoma_cgan/               # Conditional GAN implementation
        ├── Acnet_gan.py            # SemNOMA-CGAN architecture
        └── eval_ACGAN.py           # SemNOMA-CGAN evaluation
```

## ✨ Key Features

### 🔬 Model Architecture
- **Knowledge Distillation**: Teacher-student framework with cross-layer feature matching
- **Semantic Communication**: Efficient transmission of semantic information rather than raw data
- **NOMA Technology**: Non-orthogonal multiple access for improved spectral efficiency
- **Diffusion Models**: State-of-the-art image reconstruction capabilities

### 📡 Channel Models
| Channel Type | Model Class | Description |
|--------------|-------------|-------------|
| AWGN | [`SemNOMA_KD_AWGN`](kd_semnoma/models/SemNOMA_with_KD.py) | Additive White Gaussian Noise |
| Rayleigh | [`SemNOMA_KD_Rayleigh`](kd_semnoma/models/SemNOMA_with_KD.py) | Rayleigh Fading Channel |

### 📊 Supported Datasets
| Dataset | Resolution | Wrapper |
|---------|------------|---------|
| CIFAR-10 | 32×32 |[`cifar10_group_wrapper.py`](kd_semnoma/datasets/cifar10_group_wrapper.py) |
| FFHQ-256 | 256×256 | [`ffhq256_group_wrapper.py`](kd_semnoma/datasets/ffhq256_group_wrapper.py) |

## 🚀 Quick Start

### 📋 Prerequisites

```bash
# Minimum requirements
Python >= 3.8
CUDA >= 11.0 (for GPU acceleration)
```

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/KDD-SemNOMA.git
cd KDD-SemNOMA

# Create virtual environment (recommended)
python -m venv kdd_env
source kdd_env/bin/activate  # On Windows: kdd_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 🏃‍♂️ Training Pipeline

#### 1️⃣ Train Teacher Model
```bash
python kd_semnoma/train_teacher_model.py
```

#### 2️⃣ Train Student Model with Knowledge Distillation
```bash
python kd_semnoma/train_student_model.py
```

#### 3️⃣ Train SemNOMA System without Knowledge Distillation (for comparison)
```bash
python kd_semnoma/train_semnoma.py
```

#### 4️⃣ Train Baseline Models (for comparison)
```bash
python kd_semnoma/train_baseline.py
```

### 🧪 Testing and Evaluation

#### 🔍 Basic Model Testing (SemNOMA/KD-SemNOMA)
```bash
python kd_semnoma/test.py
```

#### 📡 SemNOMA-CGAN Evaluation
```bash
python kdd_semnoma/semnoma_cgan/eval_ACGAN.py \
    --ckptdir ./checkpoints \
    --net_G unet_256 \
    --dataset ffhq256
```

#### 📈 KDD-SemNOMA Evaluation
```bash
python kdd_semnoma/kdd_semnoma/inference_difface.py \
    --in_path ./testdata \
    --out_path ./results/diffusion \
    --task restoration \
    --eta 0.5
```

## 📊 Evaluation Metrics

The framework supports comprehensive evaluation using multiple metrics:

### 🖼️ Image Quality Metrics
| Metric | Description | Range | Better |
|--------|-------------|--------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | [0, ∞) dB | Higher |
| **SSIM** | Structural Similarity Index | [0, 1] | Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity | [0, ∞) | Lower |
| **FID** | Fréchet Inception Distance | [0, ∞) | Lower |


## 🔬 Research Background

### 🌐 Semantic NOMA Architecture

![KDD-SemNOMA Architecture](images\SemNOMA.jpg)

Our innovative approach combines semantic communication with NOMA:

- **Semantic Encoding**: Extract task-relevant features instead of raw data
- **Adaptative Channel Feature Encoding**: Adaptative AWGN/Rayleigh Channel Feature Encoding
- **Joint Optimization**: End-to-end learning for optimal performance

### 📚 Knowledge Distillation in Semantic Communication

![KDD-SemNOMA Architecture](images\KD-SemNOMA.jpg)

Knowledge Distillation enables efficient knowledge transfer from complex teacher networks to lightweight student networks:

- **Teacher Network**: High-capacity model trained for optimal feature extraction
- **Student Network**: Efficient model deployed at the transmitter
- **Knowledge Transfer**: Cross-layer feature matching and soft target learning
- **Benefits**: Reduced complexity while maintaining performance

### ✨ Diffusion Model Enhancement

![KDD-SemNOMA Architecture](images\KDD-SemNOMA.jpg)

Diffusion models provide superior image reconstruction:

- **High Quality**: State-of-the-art generative capabilities
- **Iterative Refinement**: Gradual denoising process
- **Conditional Generation**: Task-specific image enhancement
- **Real-time Inference**: Optimized for practical deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- The wireless communication community for foundational NOMA research
- Some codes are brought from [DifFace](https://github.com/zsyOAOA/DifFace/), [CrossKD](https://github.com/jbwang1997/CrossKD). Thanks for their awesome works.

## 📞 Contact

- **Email**: wqfneubit@163.com
---

<div align="center">
  <sub>Built with ❤️ for advancing semantic communication research</sub>
</div>