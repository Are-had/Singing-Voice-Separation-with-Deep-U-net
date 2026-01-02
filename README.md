# Singing Voice Separation with Deep U-Net

Implementation of the U-Net architecture for singing voice separation, based on the paper ["Singing Voice Separation with Deep U-Net Convolutional Networks"](https://openaccess.city.ac.uk/id/eprint/19289/) by Jansson et al. (2017).

## Overview

This project implements a deep learning model to separate vocals from mixed audio tracks using the U-Net architecture with skip connections. The model predicts a soft mask that is applied to the mixture spectrogram to isolate the vocal component.

## Requirements
```bash
pip install torch numpy librosa scipy musdb museval soundfile matplotlib tqdm
```

## Project Structure
```
project/
├── src/
│   ├── preprocessing.py    # Audio to spectrogram conversion
│   ├── data_loader.py      # PyTorch Dataset and DataLoader
│   └── model.py            # U-Net architecture
├── scripts/
│   ├── preprocess.py       # Run preprocessing
│   ├── train.py            # Training script
│   ├── inference.py        # Separate vocals from audio
│   └── evaluate.py         # Compute SDR/SIR/SAR metrics
├── data/
│   ├── musdb7s/            # MUSDB dataset
│   └── spectrograms/       # Preprocessed spectrograms
└── checkpoints/            # Trained models
```

## Usage

### 1. Preprocessing

Convert audio to spectrograms:
```bash
python scripts/preprocess.py
```

### 2. Training

Train the U-Net model:
```bash
python scripts/train.py --epochs 50 --batch_size 32
```

### 3. Inference

Separate vocals from a mixed audio file:
```bash
python scripts/inference.py --model_path checkpoints/model_epoch_20.pth --input song.wav --output vocal.wav
```

### 4. Evaluation

Compute separation metrics (SDR, SIR, SAR):
```bash
python scripts/evaluate.py
```

## Model Architecture

- **Input**: Mixture spectrogram (512 × 128)
- **Encoder**: 6 convolutional layers with downsampling
- **Decoder**: 6 deconvolutional layers with skip connections
- **Output**: Soft mask (0-1) applied to mixture
- **Loss**: L1 (MAE)
- **Optimizer**: Adam (lr=1e-3)

## Dataset

- **MUSDB18-7s**: 144 tracks (7 seconds each)
- Split: 80% train, 20% validation
- Preprocessing: Downsample to 8192 Hz, STFT (n_fft=1024, hop=768)

## Results

| Metric | Value |
|--------|-------|
| SDR Vocal | X.XX dB |
| SIR Vocal | X.XX dB |
| SAR Vocal | X.XX dB |

Comparison with original paper (iKala dataset):
- Article: SDR 11.09 dB
- This implementation: X.XX dB

## Implementation Details

- **Normalization**: Per-patch normalization to [0, 1]
- **Phase reconstruction**: Uses original mix phase for ISTFT
- **Padding**: Zero-padding for spectrograms < 128 frames
- **Device**: Auto-detection (CUDA if available, else CPU)

## Notes

- Trained on MUSDB18-7s (limited data) → overfitting observed
- Better results expected with full MUSDB18-HQ dataset
- Early stopping around epoch 15-20 recommended

## References

Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep U-Net convolutional networks. *ISMIR 2017*.

## Author

Arezki HADDOUCHE - M2 Automatique Robotique, Sorbonne Université