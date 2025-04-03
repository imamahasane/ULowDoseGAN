# GANs Framework for Low-Dose Medical Image Reconstruction

## Overview
This project implements a Generative Adversarial Network (GAN) framework to reconstruct high-quality medical images from low-dose CT scans. It features an ultra-light UNet-based generator and a custom discriminator, trained on the LoDoPaB-CT dataset, aiming to enhance image quality with computational efficiency for medical imaging applications where reducing radiation dose is critical.

## Features
- **UltraLightUNetGenerator**: A lightweight UNet-based generator optimized for medical image reconstruction.
- **Custom Discriminator**: An efficient patch-based discriminator for adversarial training.
- **Multi-Loss Optimization**: Combines adversarial, perceptual, and pixel-wise losses for superior image quality.
- **Evaluation Metrics**: Uses PSNR, SSIM, and LPIPS to assess reconstruction performance.
- **Dataset Support**: Seamlessly processes the LoDoPaB-CT dataset.
- 
## Dataset
This project utilizes the [LoDoPaB-CT Dataset](https://zenodo.org/records/3384092), which provides paired low-dose and normal-dose CT images. The dataset is licensed under the Creative Commons Attribution 4.0 International License.

- **Citation**: Leuschner, J., Schmidt, M., Baguer, D. O., & Maass, P. (2019). The LoDoPaB-CT Dataset: A Benchmark Dataset for Low-Dose CT Reconstruction Methods. Zenodo. https://doi.org/10.5281/zenodo.3384092

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for training)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/imamahasane/ULowDoseGAN.git
cd ULowDoseGAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the LoDoPaB-CT dataset from Zenodo and update BASE_PATH in src/main.py to point to your dataset directory.

### Usage
Run the main script to start training:
```bash
python src/main.py
```
###
Edit src/main.py to adjust:
- BASE_PATH: Path to the dataset
- batch_size: Training batch size (default: 12)
- epochs: Number of training epochs (default: 5)
- Learning rates and optimizer hyperparameters

### Project Structure
```bash
ULowDoseGAN/
├── src/
│   ├── data/           # Dataset handling
│   │   └── dataset.py
│   ├── models/         # Model architectures
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── losses.py
│   ├── utils/          # Training utilities
│   │   └── training.py
│   └── main.py         # Main execution script
├── requirements.txt    # Dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```

### Requirements
- torch
- torchvision
- numpy
- h5py
- piqa (for SSIM and LPIPS metrics)
- torchsummary

### License
This project is open-source and available under the MIT License.

### Contact
For questions or issues, open a GitHub issue or email: emamahasane@gmail.com.
