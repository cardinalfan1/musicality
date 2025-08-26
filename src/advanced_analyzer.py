import numpy as np
import librosa
import json
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class AdvancedAnalyzer:
    def __init__(self, config_manager):
        self.config = config_manager
        self.cache_dir = Path(self.config.get('analysis_cache_dir', './analysis_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        # Advanced parameters
        self.r = 1.0  # radius/resolution of diagonal cut
        self.w = 1.0  # checkerboard window size in seconds
        self.w_p_ratio = 4  # window/peak ratio
        self.peak_window = 0.13
        self.period_threshold = 20
        
    def get_cache_path(self, audio_path):
        audio_name = Path(audio_path).stem
        return self.cache_dir / f"{audio_name}_advanced_analysis.json"
    
    def extract_features(self, y, sr, hop_length=512):
        """Extract multiple feature types for better phrase detection."""
        
        features_list = []
        
        # Harmonic-percussive separation
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
        except:
            y_harmonic = y
            y_percussive = y
        
        # 1. Chroma features (for harmonic content)
        try:
            chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=hop_length, n_fft=2048)
            features_list.append(chroma)
        except Exception as e:
            print(f"Chroma extraction failed: {e}")
        
        # 2. MFCCs (for timbral changes) - avoid mel spectrogram issues
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=2048)
            features_list.append(mfcc)
        except Exception as e:
            print(f"MFCC extraction failed: {e}")
        
        # 3. Zero crossing rate (simple but effective)
        try:
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
            features_list.append(zcr)
        except Exception as e:
            print(f"ZCR extraction failed: {e}")
        
        # If all features failed, use a simple spectrogram
        if not features_list:
            print("All features failed, using simple spectrogram")
            stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
            features_list.append(stft[:100, :])  # Use first 100 frequency bins
        
        # Ensure all features have the same number of frames
        if features_list:
            min_frames = min(f.shape[1] for f in features_list)
            features_list = [f[:, :min_frames] for f in features_list]
            features = np.vstack(features_list)
        else:
            # Absolute fallback
            n_frames = len(y) // hop_length
            features = np.random.randn(20, n_frames)
        
        return features, y_harmonic, y_percussive
    
    def compute_ssm(self, features):
        """Compute Self-Similarity Matrix with improved distance metric."""
        
        # Normalize features
        features_norm = librosa.util.normalize(features, axis=1)
        
        # Compute recurrence matrix
        rec = librosa.segment.recurrence_matrix(
            features_norm, 
            k=None,  # Use full similarity
            width=9,  # Diagonal smoothing
            metric='cosine',
            mode='affinity'
        )
        
        # Apply path enhancement
        rec_smooth = librosa.segment.path_enhance(rec, 15)
        
        return rec_smooth
    
    def compute_novelty(self, ssm, w_frames):
        """Compute novelty curve using checkerboard kernel correlation."""
        
        # Create checkerboard kernel
        kernel_size = min(w_frames, ssm.shape[0] // 4)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:kernel_size//2, :kernel_size//2] = 1
        kernel[kernel_size//2:, kernel_size//2:] = 1
        kernel[:kernel_size//2, kernel_size//2:] = -1
        kernel[kernel_size//2:, :kernel_size//2] = -1
        
        # Compute novelty by correlation
        novelty = []
        pad = kernel_size // 2
        ssm_padded = np.pad(ssm, pad, mode='edge')
        
        for i in range(ssm.shape[0]):
            region = ssm_padded[i:i+kernel_size, i:i+kernel_size]
            novelty.append(np.sum(region * kernel))
        
        novelty = np.array(novelty)
        
        # Debug the novelty array before filtering
        print(f"Novelty shape: {novelty.shape}, dtype: {novelty.dtype}")
        print(f"Novelty has NaN: {np.any(np.isnan(novelty))}, has Inf: {np.any(np.isinf(novelty))}")
        
        # Smooth and normalize
        try:
            # Ensure novelty is finite before filtering
            if np.any(~np.isfinite(novelty)):
                print("Warning: Novelty contains non-finite values, replacing with zeros")
                novelty = np.nan_to_num(novelty, nan=0.0, posinf=0.0, neginf=0.0)
            
            novelty = gaussian_filter1d(novelty, sigma=4)
        except Exception as e:
            print(f"Gaussian filter failed: {e}")
            print(f"Novelty details: shape={novelty.shape}, dtype={novelty.dtype}")
            import traceback
            traceback.print_exc()
            # Skip smoothing if it fails
        novelty = np.maximum(novelty, 0)
        if np.max(novelty) > 0:
            novelty = novelty / np.max(novelty)
        else:
            novelty = novelty
        
        return novelty
    
    def filter_by_period(self, peaks, frames_per_beat, tolerance=20):
        """Filter peaks to align with musical periods (4, 8, 16 bars)."""
        
        if len(peaks) < 2 or frames_per_beat <= 0:
            return peaks
        
        # Common phrase lengths in beats
        phrase_beats = [16, 32, 64]  # 4, 8, 16 bars (assuming 4/4 time)
        
        # Convert to frames
        phrase_frames = [int(beats * frames_per_beat) for beats in phrase_beats]
        
        # Find the most common interval between peaks
        intervals = np.diff(peaks)
        
        # Keep peaks that align with musical periods
        filtered_peaks = [peaks[0]]
        
        for i in range(1, len(peaks)):
            interval = peaks[i] - filtered_peaks[-1]
            
            # Check if interval matches any common phrase length
            for pf in phrase_frames:
                if abs(interval - pf) < tolerance * frames_per_beat:
                    filtered_peaks.append(peaks[i])
                    break
        
        return np.array(filtered_peaks)
    
    def analyze_song(self, audio_path, force_reanalyze=False):
        """Analyze song using advanced phrase detection."""
        
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists() and not force_reanalyze:
            print(f"Loading cached advanced analysis for {Path(audio_path).name}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
                # Check for manual boundaries
                if 'manual_boundaries' in data and len(data['manual_boundaries']) > 0:
                    data['boundaries'] = data['manual_boundaries']
                    print(f"Using {len(data['boundaries'])} manual phrase boundaries")
                return data
        
        print(f"Performing advanced analysis on {Path(audio_path).name}...")
        
        # Load audio
        print("Loading audio file...")
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=None)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Audio loaded: {duration:.1f} seconds")
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Fallback to basic analysis
            return {
                'boundaries': [30, 60, 90, 120],
                'duration': 180,
                'audio_path': audio_path,
                'tempo': 120.0,
                'beat_times': [],
                'manual_boundaries': [],
                'novelty_curve': []
            }
        
        hop_length = 512
        
        # Extract features
        print("Extracting audio features...")
        try:
            features, y_harmonic, y_percussive = self.extract_features(y, sr, hop_length)
            print("Features extracted successfully")
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            # Use simple features
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Get tempo and beats
        print("Detecting tempo and beats...")
        try:
            if 'y_percussive' in locals():
                tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=hop_length)
            else:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            
            # Convert tempo to float if it's an array
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
            
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            print(f"Tempo detected: {tempo:.1f} BPM")
        except Exception as e:
            print(f"Beat detection failed: {str(e)}")
            tempo = 120.0
            beat_times = np.array([])
        
        # Frames per beat for period filtering
        if tempo > 0:
            frames_per_beat = int((60.0 / tempo) * sr / hop_length)
        else:
            frames_per_beat = int(0.5 * sr / hop_length)  # Default to 120 BPM
        
        # Compute self-similarity matrix
        print("Computing self-similarity matrix...")
        try:
            ssm = self.compute_ssm(features)
            print("SSM computed")
        except Exception as e:
            print(f"SSM computation failed: {e}, using simple method")
            # Simple alternative
            ssm = librosa.segment.recurrence_matrix(features, mode='affinity')
        
        # Compute novelty curve
        print("Computing novelty curve...")
        try:
            w_frames = int(self.w * sr / hop_length)
            novelty = self.compute_novelty(ssm, w_frames)
            print("Novelty curve computed")
        except Exception as e:
            print(f"Novelty computation failed: {e}, using simple method")
            # Simple novelty from SSM diagonal
            novelty = np.sum(np.abs(np.diff(ssm, axis=1)), axis=0)
        
        # Peak picking with musical constraints
        w_p = w_frames // self.w_p_ratio
        
        # Find peaks with adaptive threshold
        threshold = np.percentile(novelty, 80)
        distance = max(w_p, frames_per_beat * 4)  # At least 1 bar apart
        
        peaks, properties = find_peaks(
            novelty,
            height=threshold,
            distance=distance,
            prominence=threshold/2
        )
        
        # Filter by musical period
        peaks = self.filter_by_period(peaks, frames_per_beat)
        
        # Convert to time
        boundaries_time = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        
        # Filter boundaries to have enough space
        boundaries_time = boundaries_time[boundaries_time > 5]
        boundaries_time = boundaries_time[boundaries_time < duration - 10]
        
        # If no boundaries found, fall back to regular intervals
        if len(boundaries_time) == 0:
            print("No boundaries detected, creating 8-bar intervals")
            if tempo > 0:
                bars_per_phrase = 8
                seconds_per_bar = (60.0 / tempo) * 4
                phrase_duration = seconds_per_bar * bars_per_phrase
                boundaries_time = np.arange(phrase_duration, duration - 10, phrase_duration)
            else:
                boundaries_time = np.arange(32, duration - 10, 32)
        
        # Convert to list if it's a numpy array
        if isinstance(boundaries_time, np.ndarray):
            boundaries_list = boundaries_time.tolist()
        else:
            boundaries_list = list(boundaries_time)
        
        if isinstance(beat_times, np.ndarray):
            beat_times_list = beat_times.tolist()
        else:
            beat_times_list = list(beat_times) if beat_times else []
        
        if isinstance(novelty, np.ndarray):
            novelty_list = novelty.tolist()
        else:
            novelty_list = list(novelty) if hasattr(novelty, '__iter__') else []
        
        result = {
            'boundaries': boundaries_list,
            'duration': duration,
            'audio_path': audio_path,
            'tempo': float(tempo),
            'beat_times': beat_times_list,
            'manual_boundaries': [],
            'novelty_curve': novelty_list
        }
        
        with open(cache_path, 'w') as f:
            json.dump(result, f)
        
        print(f"Advanced analysis found {len(boundaries_time)} phrase boundaries at tempo {tempo:.1f} BPM")
        return result
    
    def save_manual_boundaries(self, audio_path, boundaries):
        """Save user-defined phrase boundaries."""
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'boundaries': boundaries,
                'duration': 300,
                'audio_path': audio_path,
                'tempo': 120,
                'beat_times': [],
                'novelty_curve': []
            }
        
        data['manual_boundaries'] = boundaries
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {len(boundaries)} manual boundaries")
        return data