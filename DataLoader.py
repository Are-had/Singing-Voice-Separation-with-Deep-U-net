import torch
import torch.utils.data as Data
import numpy as np
import os


class DataLoader (Data.Dataset):
    def __init__(self, path):
        """
        Dataset for loading full spectrograms (7 seconds version with padding)
        
        Args:
            path: root folder containing 'mixture' and 'vocal' subfolders
        """
        self.path = path
        self.files = sorted(os.listdir(os.path.join(path, 'mixture')))
        self.files = [name for name in self.files if 'spec' in name]
    
    def __len__(self):
        """Return total number of files"""
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load one spectrogram, pad to 128 frames, and normalize
        
        Args:
            idx: index of the file to load
            
        Returns:
            mix: (1, 512, 128) mixture spectrogram tensor
            voc: (1, 512, 128) vocal spectrogram tensor
        """
        # Load spectrograms (NOT normalized)
        mix = np.load(os.path.join(self.path, 'mixture', self.files[idx]))
        voc = np.load(os.path.join(self.path, 'vocal', self.files[idx]))
        
        # Remove first frequency bin (513 → 512)
        mix = mix[1:, :]
        voc = voc[1:, :]
        
        # STEP 1: Pad FIRST to 128 frames
        if mix.shape[-1] < 128:
            padding = 128 - mix.shape[-1]
            mix = np.pad(mix, ((0, 0), (0, padding)), mode='constant')
            voc = np.pad(voc, ((0, 0), (0, padding)), mode='constant')
        
        # STEP 2: Normalize AFTER padding
        mix = self._normalize(mix)
        voc = self._normalize(voc)
        
        # Add channel dimension (512, 128) → (512, 128, 1)
        mix = mix[:, :, np.newaxis].astype(np.float32)
        voc = voc[:, :, np.newaxis].astype(np.float32)
        
        # Convert to PyTorch tensors and permute (512, 128, 1) → (1, 512, 128)
        mix = torch.from_numpy(mix).permute(2, 0, 1)
        voc = torch.from_numpy(voc).permute(2, 0, 1)
        
        return mix, voc
    
    def _normalize(self, spectrogram):
        """
        Normalize spectrogram to [0, 1]
        
        Args:
            spectrogram: numpy array
            
        Returns:
            normalized spectrogram
        """
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max - spec_min > 0:
            normalized = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            normalized = spectrogram
        
        return normalized


def create_dataloader(data_path, batch_size=8, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader
    
    Args:
        data_path: path to spectrograms folder (containing 'mixture' and 'vocal' subfolders)
        batch_size: number of samples per batch (default: 8)
        shuffle: whether to shuffle data (default: True)
        num_workers: number of parallel workers for data loading (default: 0)
        
    Returns:
        DataLoader object
    """
    dataset = DataLoader (data_path)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return loader
