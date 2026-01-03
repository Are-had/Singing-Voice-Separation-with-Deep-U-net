from scipy import signal
import numpy as np
import librosa
from tqdm import tqdm
import os
import torch






def preprocess_musdb_and_save(tracks, output_folder):

    """
    Process all MUSDB tracks and save FULL spectrograms WITHOUT normalization
    
    Args:
        mus: musdb.DB object
        output_folder: path to save spectrograms
    """
      
    mix_folder = os.path.join(output_folder, 'mixture')
    vocal_folder = os.path.join(output_folder, 'vocal')
    phase_folder = os.path.join(output_folder, 'phase')

    os.makedirs(mix_folder, exist_ok=True)
    os.makedirs(vocal_folder, exist_ok=True)
    os.makedirs(phase_folder, exist_ok=True)
    
    for track_idx, track in enumerate(tqdm(tracks, desc="Processing tracks")):
        try:
            track_name = track.name.replace(' ', '_').replace('/', '_')
            
            # Get audio
            mix_audio = track.audio
            vocal_audio = track.targets['vocals'].audio
            
            # Convert to full spectrograms (NO normalization, NO chunking)
            mix_spec, mix_phase = __audio_to_spectrogram(mix_audio)
            vocal_spec, _ = __audio_to_spectrogram(vocal_audio)
            
            #  NORMALISER PAR CHANSON 
            norm = mix_spec.max()  # Max of mixture
            mix_spec = mix_spec / norm
            vocal_spec = vocal_spec / norm  
            
            # Save
            mix_path = os.path.join(mix_folder, f'{track_name}_spec.npy')
            vocal_path = os.path.join(vocal_folder, f'{track_name}_spec.npy')
            phase_path = os.path.join(phase_folder, f'{track_name}_phase.npy')
            
            np.save(mix_path, mix_spec)
            np.save(vocal_path, vocal_spec)
            np.save(phase_path, mix_phase)
            
        except Exception as e:
            print(f"\n[Warning] Error processing track '{track.name}': {e}")
            continue
    
    print(f"\nPreprocessing done! Spectrograms saved in: {output_folder}")




def separate_vocals(model, audio, sr=44100, patch_size=128, hop_size=64):
    """
    Separate vocals from audio: audio → spectrogram → inference → vocal audio
    
    Args:
        model: Trained U-Net model
        audio: Audio signal (mono or stereo)
        sr: Sample rate (default: 44100)
        patch_size: Size of patches for inference (default: 128)
        hop_size: Hop size for overlapping patches (default: 64)
    
    Returns:
        vocal_audio: Separated vocal audio as numpy array
    """
    
    # Step 1: Audio → Spectrogram 
    mix_spec, mix_phase = __audio_to_spectrogram(
        audio=audio,
        sr_original=sr,
        target_sr=8192,
        n_fft=1024,
        hop_length=768
    )
    
    # Step 2: Normalization 
    norm = mix_spec.max()
    mix_normalized = mix_spec / norm
    
    # Step 3: Inference with overlapping patches
    freq_bins, total_frames = mix_spec.shape
    
    vocal_spec_sum = np.zeros_like(mix_spec)
    weight_sum = np.zeros(total_frames)
    
    start = 0
    
    model.eval()
    with torch.no_grad():
        while start < total_frames:
            end = min(start + patch_size, total_frames)
            
            # Extract patch
            patch = mix_normalized[:, start:end]
            original_patch_size = patch.shape[1]
            
            # Pad if necessary
            if patch.shape[1] < patch_size:
                padding = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, 0), (0, padding)), mode='constant')
            
            # Remove first frequency bin (513 → 512)
            patch_512 = patch[1:, :]
            
            # Convert to tensor
            patch_tensor = torch.from_numpy(patch_512[np.newaxis, np.newaxis, :, :]).float()
            
            # Predict mask
            mask = model.forward(patch_tensor)
            
            # Convert back to numpy
            mask_np = mask.cpu().numpy()[0, 0, :, :]
            
            # Add first frequency bin back
            mask_full = np.zeros((513, patch_size))
            mask_full[1:, :] = mask_np
            
            # Apply mask
            vocal_patch = mask_full * patch
            
            # Crop to actual size
            vocal_patch = vocal_patch[:, :original_patch_size]
            
            # Accumulate
            vocal_spec_sum[:, start:end] += vocal_patch
            weight_sum[start:end] += 1
            
            start += hop_size
    
    # Average overlapping regions
    vocal_spec_normalized = vocal_spec_sum / weight_sum[np.newaxis, :]
    
    # Step 4: Denormalization
    vocal_spec_full = vocal_spec_normalized * norm
    
    # Step 5: Spectrogram → Audio 
    vocal_audio = spectrogram_to_audio(
        magnitude=vocal_spec_full,
        phase=mix_phase,
        target_sr=8192,
        hop_length=768,
        n_fft=1024,
        original_sr=sr
    )
    
    return vocal_audio


def __audio_to_spectrogram(audio, sr_original=44100, target_sr=8192, 
                          n_fft=1024, hop_length=768):
    """
    Convert audio to full spectrogram WITHOUT normalization
    
    Args:
        audio: (n_samples, 2) stereo audio
        sr_original: original sampling rate (default: 44100)
        target_sr: target sampling rate (default: 8192)
        n_fft: FFT window size (default: 1024)
        hop_length: hop length (default: 768)
    
    Returns:
        magnitude spectrogram: (freq_bins, total_frames) - NOT normalized
    """
    # Step 1: Stereo to Mono
    mono = np.mean(audio, axis=1)
    
    # Step 2: Resample
    duration = len(mono) / sr_original
    target_length = int(duration * target_sr)
    resampled = signal.resample(mono, target_length)
    
    # Step 3: STFT on entire audio
    stft = librosa.stft(resampled, n_fft=n_fft, hop_length=hop_length)
    
    # Step 4: Magnitude 
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    return magnitude , phase






def calculate_val_loss(model, val_loader):
    """Calculate validation loss"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for mix, voc in val_loader:
            mix = mix.to(model.device)
            voc = voc.to(model.device)
            
            mask = model.forward(mix)
            predicted_vocal = mask * mix
            loss = model.criterion(predicted_vocal, voc)
            
            val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses)



def spectrogram_to_audio(magnitude, phase, target_sr=8192, hop_length=768, 
                         n_fft=1024, original_sr=44100):
    """
    Convert spectrogram back to audio using ISTFT
    
    Args:
        magnitude: (freq_bins, n_frames) - magnitude spectrogram
        phase: (freq_bins, n_frames) - phase spectrogram
        target_sr: sampling rate used for spectrogram (default: 8192)
        hop_length: hop length used for STFT (default: 768)
        n_fft: FFT window size (default: 1024)
        original_sr: final output sampling rate (default: 44100)
    
    Returns:
        audio: 1D numpy array at original_sr sampling rate
    """
    # Reconstruct complex spectrogram from magnitude and phase
    stft_matrix = magnitude * np.exp(1j * phase)
    
    # Inverse STFT
    audio = librosa.istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)
    
    # Resample back to original sampling rate if needed
    if target_sr != original_sr:
        chunk_size = target_sr * 30  # 30 secondes
        
        if len(audio) > chunk_size:
            # Process par chunks
            resampled_chunks = []
            
            for start in range(0, len(audio), chunk_size):
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end]
                
                # Resample ce chunk
                duration = len(chunk) / target_sr
                target_length = int(duration * original_sr)
                resampled_chunk = signal.resample(chunk, target_length)
                
                resampled_chunks.append(resampled_chunk)
            
            # Concatenate tous les chunks
            audio = np.concatenate(resampled_chunks)
        else:
            # Audio court, resample normalement
            duration = len(audio) / target_sr
            target_length = int(duration * original_sr)
            audio = signal.resample(audio, target_length)
    
    return audio