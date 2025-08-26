import numpy as np
import librosa
import json
from pathlib import Path
import random

class AudioAnalyzer:
    def __init__(self, config_manager):
        self.config = config_manager
        self.cache_dir = Path(self.config.get('analysis_cache_dir', './analysis_cache'))
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, audio_path):
        audio_name = Path(audio_path).stem
        return self.cache_dir / f"{audio_name}_analysis.json"
    
    def analyze_song(self, audio_path, force_reanalyze=False, detection_mode='auto'):
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists() and not force_reanalyze:
            print(f"Loading cached analysis for {Path(audio_path).name}")
            with open(cache_path, 'r') as f:
                data = json.load(f)
                # Check if we have manual boundaries
                if 'manual_boundaries' in data and len(data['manual_boundaries']) > 0:
                    data['boundaries'] = data['manual_boundaries']
                    print(f"Using {len(data['boundaries'])} manual phrase boundaries")
                return data
        
        print(f"Analyzing {Path(audio_path).name} for phrase boundaries and tempo...")
        
        try:
            y, sr = librosa.load(audio_path, sr=22050)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            # Try with default sample rate
            try:
                y, sr = librosa.load(audio_path, sr=None)
                print(f"Loaded with native sample rate: {sr}")
            except Exception as e2:
                print(f"Failed to load audio completely: {str(e2)}")
                raise e2
        
        hop_length = 512
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Try to get tempo and beats, but don't fail if it doesn't work
        tempo = 120.0  # Default tempo
        beat_times = []
        
        try:
            # Try the simplest approach first
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            try:
                # Try new API first (librosa >= 0.10)
                tempo_array = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            except (AttributeError, ImportError):
                # Fall back to old API
                tempo_array = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
            tempo = float(tempo_array[0]) if len(tempo_array) > 0 else 120.0
            
            # Generate approximate beat times based on tempo
            beat_duration = 60.0 / tempo
            beat_times = np.arange(0, duration, beat_duration).tolist()
            print(f"Detected tempo: {tempo:.1f} BPM")
        except Exception as e:
            print(f"Beat detection failed, using default tempo: {e}")
            # Generate beat times for default 120 BPM
            beat_duration = 0.5  # 60/120
            beat_times = np.arange(0, duration, beat_duration).tolist()
        
        # Extract features for boundary detection
        try:
            # Use simpler STFT-based features if CQT fails
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
            except:
                # Fallback to even simpler approach
                stft = np.abs(librosa.stft(y, hop_length=hop_length))
                chroma = np.ones((12, stft.shape[1]))  # Dummy chroma
            
            try:
                # Use direct MFCC computation without mel spectrogram
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=2048)
            except Exception as e:
                print(f"MFCC failed ({e}), using simple alternative")
                # Fallback to dummy MFCCs
                n_frames = len(y) // hop_length
                mfcc = np.ones((13, n_frames))
            
            # Ensure same number of frames
            min_frames = min(chroma.shape[1], mfcc.shape[1])
            chroma = chroma[:, :min_frames]
            mfcc = mfcc[:, :min_frames]
            
            features = np.vstack([chroma, mfcc])
            
            # Detect boundaries using self-similarity
            rec = librosa.segment.recurrence_matrix(features, k=5, mode='affinity', 
                                                    metric='cosine', sparse=True)
            
            R = librosa.segment.path_enhance(rec, 15)
            
            # Compute novelty curve
            novelty = np.sum(np.abs(np.diff(R.toarray(), axis=1)), axis=0)
            novelty = np.pad(novelty, (1, 0), mode='constant')
            
            # Find peaks in novelty curve
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(novelty, 
                                          height=np.percentile(novelty, 75),
                                          distance=sr*2//hop_length)
            
            boundaries_frames = peaks
            boundaries_time = librosa.frames_to_time(boundaries_frames, sr=sr, hop_length=hop_length)
            
            # Align boundaries to musical grid (8 or 16 bars)
            if tempo > 0 and len(beat_times) > 0:
                boundaries_time = self.align_to_musical_grid(boundaries_time, tempo, duration)
            
        except Exception as e:
            import traceback
            print(f"\n!!! Feature extraction failed !!!")
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            print("Traceback:")
            traceback.print_exc()
            print("\nFalling back to beat-grid segmentation (less accurate)")
            print("Recommendation: Use Training Mode to mark phrases manually\n")
            
            # Fallback: Create boundaries at regular 8-bar intervals
            if tempo > 0:
                bars_per_phrase = 8  # Most dance music uses 8-bar phrases
                seconds_per_bar = (60.0 / tempo) * 4  # 4 beats per bar
                phrase_duration = seconds_per_bar * bars_per_phrase
                boundaries_time = np.arange(phrase_duration, duration - 10, phrase_duration)
                print(f"Creating boundaries every {bars_per_phrase} bars ({phrase_duration:.1f} seconds)")
            else:
                boundaries_time = np.arange(32, duration - 10, 32)  # Default 32 seconds
                print("Using default 32-second intervals")
        
        # Filter boundaries to have enough space before and after
        boundaries_time = boundaries_time[boundaries_time > 10]
        boundaries_time = boundaries_time[boundaries_time < duration - 10]
        
        # If no boundaries found, create at 8-bar intervals
        if len(boundaries_time) == 0:
            print("No boundaries detected, creating 8-bar intervals")
            if tempo > 0:
                bars_per_phrase = 8
                seconds_per_bar = (60.0 / tempo) * 4
                phrase_duration = seconds_per_bar * bars_per_phrase
                boundaries_time = np.arange(phrase_duration, duration - 10, phrase_duration)
            else:
                boundaries_time = np.arange(32, duration - 10, 32)
        
        result = {
            'boundaries': boundaries_time.tolist(),
            'duration': duration,
            'audio_path': audio_path,
            'tempo': float(tempo),
            'beat_times': beat_times,
            'manual_boundaries': []  # Will store user-defined boundaries
        }
        
        with open(cache_path, 'w') as f:
            json.dump(result, f)
        
        print(f"Found {len(boundaries_time)} phrase boundaries, tempo: {tempo:.1f} BPM")
        return result
    
    def get_random_boundary(self, analysis_result):
        boundaries = analysis_result['boundaries']
        if not boundaries:
            return None
        return random.choice(boundaries)
    
    def get_measure_offset(self, boundary_time, beat_times, tempo, num_measures):
        """Calculate the start time that is num_measures before the boundary."""
        beats_per_measure = 4
        beats_per_second = tempo / 60
        seconds_per_measure = beats_per_measure / beats_per_second
        
        # Simple calculation: just go back by the number of measures
        offset_seconds = num_measures * seconds_per_measure
        start_time = max(0, boundary_time - offset_seconds)
        
        # If we have beat times, try to snap to the nearest beat
        if beat_times and len(beat_times) > 0:
            beat_times_array = np.array(beat_times)
            # Find the beat closest to our calculated start time
            closest_beat_idx = np.argmin(np.abs(beat_times_array - start_time))
            if closest_beat_idx < len(beat_times):
                start_time = beat_times[closest_beat_idx]
        
        return start_time
    
    def align_to_musical_grid(self, boundaries, tempo, duration):
        """Align detected boundaries to nearest 8 or 16 bar grid points."""
        if tempo <= 0:
            return boundaries
        
        # Calculate phrase grid (8-bar intervals)
        bars_per_phrase = 8
        seconds_per_bar = (60.0 / tempo) * 4  # 4 beats per bar
        phrase_duration = seconds_per_bar * bars_per_phrase
        
        # Create grid points every 8 bars
        grid_points = np.arange(0, duration, phrase_duration)
        
        # For 16-bar phrases (common in EDM), also add those
        grid_points_16 = np.arange(0, duration, phrase_duration * 2)
        all_grid_points = np.unique(np.concatenate([grid_points, grid_points_16]))
        
        # Snap each boundary to nearest grid point
        aligned_boundaries = []
        for boundary in boundaries:
            if len(all_grid_points) > 0:
                nearest_idx = np.argmin(np.abs(all_grid_points - boundary))
                nearest_grid = all_grid_points[nearest_idx]
                
                # Only snap if within 2 seconds of a grid point
                if abs(nearest_grid - boundary) < 2.0:
                    aligned_boundaries.append(nearest_grid)
                else:
                    # Keep original if too far from grid
                    aligned_boundaries.append(boundary)
            else:
                aligned_boundaries.append(boundary)
        
        # Remove duplicates and sort
        aligned_boundaries = sorted(list(set(aligned_boundaries)))
        
        print(f"Aligned {len(boundaries)} boundaries to {len(aligned_boundaries)} grid points")
        return np.array(aligned_boundaries)
    
    def save_manual_boundaries(self, audio_path, boundaries):
        """Save user-defined phrase boundaries."""
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            # Create minimal data if cache doesn't exist
            data = {
                'boundaries': boundaries,
                'duration': 300,  # Default
                'audio_path': audio_path,
                'tempo': 120,
                'beat_times': []
            }
        
        data['manual_boundaries'] = boundaries
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {len(boundaries)} manual boundaries")
        return data