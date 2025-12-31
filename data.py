from scipy import signal
import numpy as np
import librosa
from tqdm import tqdm
import os


def preprocess_musdb_and_save(mus, output_folder):
    """
    Process all MUSDB tracks and save FULL spectrograms WITHOUT normalization
    
    Args:
        mus: musdb.DB object
        output_folder: path to save spectrograms
    """
    # Create folders
    mix_folder = os.path.join(output_folder, 'mixture')
    vocal_folder = os.path.join(output_folder, 'vocal')
    os.makedirs(mix_folder, exist_ok=True)
    os.makedirs(vocal_folder, exist_ok=True)
    
    # Process all tracks
    for track_idx, track in enumerate(tqdm(mus.tracks, desc="Processing tracks")):
        try:
            track_name = track.name.replace(' ', '_').replace('/', '_')
            
            # Get audio
            mix_audio = track.audio
            vocal_audio = track.targets['vocals'].audio
            
            # Convert to full spectrograms (NO normalization, NO chunking)
            mix_spec = __audio_to_spectrogram(mix_audio)
            vocal_spec = __audio_to_spectrogram(vocal_audio)
            
            # Save
            mix_path = os.path.join(mix_folder, f'{track_name}_spec.npy')
            vocal_path = os.path.join(vocal_folder, f'{track_name}_spec.npy')
            
            np.save(mix_path, mix_spec)
            np.save(vocal_path, vocal_spec)
            
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
    
    # Step 4: Magnitude (NO normalization)
    magnitude = np.abs(stft)
    
    return magnitude