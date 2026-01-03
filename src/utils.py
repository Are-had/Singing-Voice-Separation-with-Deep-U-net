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



def prepare_spectrogram_for_inference(mix_spec, model_device, target_frames=128):
    """
    Prepare a spectrogram for inference by normalizing and formatting it for the model
    
    Args:
        mix_spec: (freq_bins, n_frames) - magnitude spectrogram
        model_device: device to put the tensor on (e.g., 'cpu', 'cuda')
        target_frames: target number of frames (default: 128)
    
    Returns:
        mix_tensor: (1, 1, 512, target_frames) - tensor ready for model inference
        spec_min: minimum value used for normalization
        spec_max: maximum value used for normalization
        n_frames: original number of frames (before padding)
    """
    
    # Get original dimensions
    freq_bins, n_frames = mix_spec.shape
    
    # Pad if needed
    norm = mix_spec.max()
    mix_normalized = mix_spec / norm
    
    # Pad if needed
    if n_frames < target_frames:
        padding = target_frames - n_frames
        mix_padded = np.pad(mix_normalized, ((0, 0), (0, padding)), mode='constant')
    else:
        mix_padded = mix_normalized[:, :target_frames]
    
    # Remove first frequency bin (513 → 512)
    mix_normalized_512_bins = mix_normalized[1:, :]  
    
    # Convert to tensor (1, 1, 512, target_frames)
    mix_tensor = torch.from_numpy(mix_normalized_512_bins[np.newaxis, np.newaxis, :, :]).float()
    mix_tensor = mix_tensor.to(model_device)
    
    return mix_tensor, n_frames,norm , mix_normalized  




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
        duration = len(audio) / target_sr
        target_length = int(duration * original_sr)
        audio = signal.resample(audio, target_length)
    
    return audio

