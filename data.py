from scipy import signal
import numpy as np
import librosa
from tqdm import tqdm
import os
import numpy as np



def preprocess_musdb_and_save(mus, output_folder, chunk_duration=12.0, overlap=0.5):
    """
    Process all MUSDB tracks and save spectrograms
    
    Args:
        mus: musdb.DB object
        output_folder: path to save spectrograms
        chunk_duration: duration of each chunk in seconds (default: 12.0)
        overlap: overlap ratio between chunks (default: 0.5)
    """

    
    # Create output folder if doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all tracks
    for track_idx, track in enumerate(tqdm(mus.tracks, desc="Processing tracks")):
        try:
            track_name = track.name
            
            # Get audio
            mix_audio = track.audio
            vocal_audio = track.targets['vocals'].audio
            
            # Split into chunks
            mix_chunks = __split_audio_into_chunks(mix_audio, sr=44100, 
                                                 chunk_duration=chunk_duration, 
                                                 overlap=overlap)
            vocal_chunks = __split_audio_into_chunks(vocal_audio, sr=44100, 
                                                   chunk_duration=chunk_duration, 
                                                   overlap=overlap)
            
            # Process each chunk
            for chunk_idx, (mix_chunk, vocal_chunk) in enumerate(zip(mix_chunks, vocal_chunks)):
                try:
                    # Convert to spectrogram
                    mix_spec = __audio_to_spectrogram(mix_chunk)
                    vocal_spec = __audio_to_spectrogram(vocal_chunk)
                    
                    # Save
                    mix_path = os.path.join(output_folder, 
                                          f'{track_name}_mixture_patch{chunk_idx:03d}.npy')
                    vocal_path = os.path.join(output_folder, 
                                            f'{track_name}_vocal_patch{chunk_idx:03d}.npy')
                    
                    np.save(mix_path, mix_spec)
                    np.save(vocal_path, vocal_spec)
                    
                except Exception as e:
                    print(f"\n[Warning] Error processing chunk {chunk_idx} of track '{track_name}': {e}")
                    continue
            
        except Exception as e:
            print(f"\n[Warning] Error processing track '{track.name}': {e}")
            continue
    
    print(f"\nPreprocessing done! Spectrograms saved in: {output_folder}")




def __audio_to_spectrogram(audio, sr_original=44100, target_sr=8192, 
                         n_fft=1024, hop_length=768, target_frames=128):
    """
    Convert audio to spectrogram following the paper's parameters
    
    Args:
        audio: (n_samples, 2) stereo audio
        sr_original: original sampling rate (default: 44100)
        target_sr: target sampling rate (default: 8192)
        n_fft: FFT window size (default: 1024)
        hop_length: hop length (default: 768)
        target_frames: target number of frames (default: 128)
    
    Returns:
        normalized spectrogram: (freq_bins, target_frames)
    """
    # Step 1: Stereo to Mono
    mono = np.mean(audio, axis=1)
    
    # Step 2: Resample 
    duration = len(mono) / sr_original
    target_length = int(duration * target_sr)
    resampled = signal.resample(mono, target_length)

    #split into chunks



    
    # Step 3: STFT
    stft = librosa.stft(resampled, n_fft=n_fft, hop_length=hop_length)
    
    # Step 4: Magnitude
    magnitude = np.abs(stft)
    
    # Step 5: Handle frames 
    magnitude = __handle_frames(magnitude, target_frames)
    
    # Step 6: Normalize 
    normalized = __normalize_spectrogram(magnitude)
    
    return normalized




def __split_audio_into_chunks(audio,  chunk_duration=12.0, overlap=0.5):
    """
    Split audio into overlapping chunks
    
    Args:
        audio: (n_samples, 2) stereo audio
        sr: sampling rate (default: 44100)
        chunk_duration: duration of each chunk in seconds (default: 12.0)
        overlap: overlap ratio between chunks (default: 0.5 = 50%)
    
    Returns:
        list of audio chunks, each of shape (chunk_samples, 2)
    """
    n_samples = len(audio)
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    chunks = []
    
    for start in range(0, n_samples, hop_samples):
        end = start + chunk_samples
        
        if end <= n_samples:
            # Full chunk
            chunk = audio[start:end]
        else:
            # Last chunk - pad with zeros
            chunk = audio[start:]
            padding = chunk_samples - len(chunk)
            chunk = np.pad(chunk, ((0, padding), (0, 0)), mode='constant')
        
        chunks.append(chunk)
        
        # Stop if we've covered all audio
        if end >= n_samples:
            break
    
    return chunks






def __handle_frames(spectrogram, target_frames=128):
    """
    Handle the number of frames
    - If < 128: pad with zeros
    - If > 128: extract a patch of 128 frames
    """
    freq_bins, time_frames = spectrogram.shape
    
    if time_frames < target_frames:
        # Padding
        padding = target_frames - time_frames
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
    elif time_frames > target_frames:
        # Extract a random patch of 128 frames
        start_frame = np.random.randint(0, time_frames - target_frames + 1)
        spectrogram = spectrogram[:, start_frame:start_frame + target_frames]
    
    return spectrogram


def __normalize_spectrogram(spectrogram):
    """
    Normalize spectrogram to [0, 1]
    """
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    
    if spec_max - spec_min > 0:
        normalized = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        normalized = spectrogram
    
    return normalized




