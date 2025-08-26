import numpy as np
import librosa
import json
from pathlib import Path

class SimpleAnalyzer:
    def __init__(self, config_manager):
        self.config = config_manager
        self.cache_dir = Path(self.config.get('analysis_cache_dir', './analysis_cache'))
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, audio_path):
        audio_name = Path(audio_path).stem
        return self.cache_dir / f"{audio_name}_analysis.json"
    
    def analyze_song(self, audio_path, force_reanalyze=False):
        """Load song metadata and training data only."""
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists() and not force_reanalyze:
            print(f"Loading analysis for {Path(audio_path).name}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        print(f"Analyzing {Path(audio_path).name}...")
        
        # Just get basic info - duration and tempo
        y, sr = librosa.load(audio_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Get tempo and beat positions for rhythm display
        try:
            print("Detecting beats for rhythm display...")
            
            # Try multiple methods and pick the best one
            hop_length = 512
            
            # Method 1: Standard beat tracking
            print("Trying standard beat tracking...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempo1, beats1 = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Method 2: Percussive separation
            print("Trying percussive separation...")
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=2.0)
            tempo2, beats2 = librosa.beat.beat_track(
                y=y_percussive, 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Method 3: Use onset detection with peak picking
            print("Trying onset-based beat tracking...")
            onset_env_strong = librosa.onset.onset_strength(
                y=y, 
                sr=sr,
                hop_length=hop_length,
                aggregate=np.median
            )
            tempo3 = librosa.beat.tempo(onset_envelope=onset_env_strong, sr=sr)[0]
            
            # Choose the method with the most reasonable beat count
            beat_counts = [
                (len(beats1), tempo1, beats1, "standard"),
                (len(beats2), tempo2, beats2, "percussive"),
            ]
            
            # Pick the tempo that seems most reasonable (usually between 60-180 BPM for dance music)
            tempos = [t for c, t, b, m in beat_counts if 60 <= float(t if not hasattr(t, '__len__') else t[0]) <= 180]
            if tempos:
                tempo = float(np.median(tempos))
            else:
                tempo = float(tempo1 if not hasattr(tempo1, '__len__') else tempo1[0])
            
            # Expected beat count
            expected = (duration / 60.0) * tempo
            
            # Choose method closest to expected
            best_method = min(beat_counts, key=lambda x: abs(x[0] - expected))
            beats = best_method[2]
            method_name = best_method[3]
            print(f"Using {method_name} method")
            
            # Convert to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512).tolist()
            
            # Filter out beats that are too close together (likely errors)
            if len(beat_times) > 1:
                filtered_beats = [beat_times[0]]
                min_beat_gap = 60.0 / (tempo * 2)  # Minimum gap is half a beat
                for beat in beat_times[1:]:
                    if beat - filtered_beats[-1] > min_beat_gap:
                        filtered_beats.append(beat)
                beat_times = filtered_beats
            
            # Analyze pairs of beats to determine boom-tick pattern
            downbeats = []
            
            if len(beat_times) >= 2:
                # Sample first 8 beats to determine pattern
                sample_beats = beat_times[:min(8, len(beat_times))]
                beat_centroids = []
                
                for beat_time in sample_beats:
                    # Get audio segment around beat (50ms window)
                    start_sample = int(max(0, (beat_time - 0.025) * sr))
                    end_sample = int(min(len(y), (beat_time + 0.025) * sr))
                    beat_segment = y[start_sample:end_sample]
                    
                    if len(beat_segment) > 0:
                        try:
                            # Calculate spectral centroid (average frequency)
                            fft = np.abs(np.fft.rfft(beat_segment))
                            freqs = np.fft.rfftfreq(len(beat_segment), 1/sr)
                            if np.sum(fft) > 0:
                                centroid = np.sum(freqs * fft) / np.sum(fft)
                                beat_centroids.append(centroid)
                            else:
                                beat_centroids.append(1000)  # Default to mid frequency
                        except:
                            beat_centroids.append(1000)
                    else:
                        beat_centroids.append(1000)
                
                # Determine which beats in the pattern are booms (lower frequency)
                # Compare alternating beats
                if len(beat_centroids) >= 2:
                    # Check if odd beats (0, 2, 4...) or even beats (1, 3, 5...) are lower
                    odd_avg = np.mean([beat_centroids[i] for i in range(0, len(beat_centroids), 2)])
                    even_avg = np.mean([beat_centroids[i] for i in range(1, len(beat_centroids), 2)])
                    
                    # The lower frequency group are the booms
                    if odd_avg < even_avg:
                        # Beats 0, 2, 4... are booms
                        boom_indices = list(range(0, len(beat_times), 2))
                    else:
                        # Beats 1, 3, 5... are booms
                        boom_indices = list(range(1, len(beat_times), 2))
                    
                    # Create list of boom beat times
                    downbeats = [beat_times[i] for i in boom_indices if i < len(beat_times)]
                else:
                    # Default to beats 0, 2, 4... as booms
                    downbeats = [beat_times[i] for i in range(0, len(beat_times), 2)]
            else:
                downbeats = []
            
            print(f"Detected {len(beat_times)} beats at {tempo:.1f} BPM")
            print(f"  - {len(downbeats)} boom beats, {len(beat_times) - len(downbeats)} tick beats")
            print(f"  - Song duration: {duration:.1f}s")
            print(f"  - Average beat interval: {60.0/tempo:.3f}s")
            
            # Sanity check - if we have way too few or too many beats, try alternate method
            expected_beats = (duration / 60.0) * tempo
            print(f"  - Expected ~{expected_beats:.0f} beats based on tempo")
            if len(beat_times) < expected_beats * 0.5 or len(beat_times) > expected_beats * 2:
                print(f"Beat count seems off (expected ~{expected_beats:.0f}, got {len(beat_times)})")
                print("Trying percussive separation method...")
                
                # Try with percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                tempo_p, beats_p = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=512)
                beat_times_p = librosa.frames_to_time(beats_p, sr=sr, hop_length=512).tolist()
                
                if abs(len(beat_times_p) - expected_beats) < abs(len(beat_times) - expected_beats):
                    print(f"Using percussive method: {len(beat_times_p)} beats")
                    beat_times = beat_times_p
                    tempo = float(tempo_p) if not hasattr(tempo_p, '__len__') else float(tempo_p[0])
                    
                    # Redo boom/tick analysis with new beats
                    if len(beat_times) >= 2:
                        downbeats = [beat_times[i] for i in range(0, len(beat_times), 2)]
                    else:
                        downbeats = []
                        
        except Exception as e:
            print(f"Beat detection failed: {e}")
            tempo = 120.0
            beat_times = []
            downbeats = []
        
        result = {
            'boundaries': [],  # No auto-detected boundaries
            'manual_boundaries': [],  # User will add these in Training Mode
            'duration': duration,
            'audio_path': audio_path,
            'tempo': float(tempo),
            'beat_times': beat_times,  # All beats
            'downbeats': downbeats  # Strong beats (1st beat of each measure)
        }
        
        with open(cache_path, 'w') as f:
            json.dump(result, f)
        
        print(f"Song loaded: {duration:.1f}s, {tempo:.1f} BPM")
        print("Use Training Mode to mark phrase boundaries")
        return result
    
    def save_training_data(self, audio_path, training_data):
        """Save all training data including beats and tempo sections."""
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'boundaries': [],
                'duration': 300,
                'audio_path': audio_path,
                'tempo': 120
            }
        
        # Update with all training data
        data['manual_boundaries'] = training_data.get('phrase_marks', [])
        data['boundaries'] = training_data.get('phrase_marks', [])
        data['beat_times'] = training_data.get('beat_times', [])
        data['downbeats'] = training_data.get('downbeats', [])
        data['tempo_sections'] = training_data.get('tempo_sections', [])
        if training_data.get('manual_tempo'):
            data['tempo'] = training_data['manual_tempo']
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved complete training data with {len(data['manual_boundaries'])} phrases and {len(data.get('tempo_sections', []))} tempo sections")
        return data
    
    def save_manual_boundaries(self, audio_path, boundaries):
        """Save user-defined phrase boundaries."""
        cache_path = self.get_cache_path(audio_path)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'boundaries': [],
                'duration': 300,
                'audio_path': audio_path,
                'tempo': 120
            }
        
        data['manual_boundaries'] = boundaries
        data['boundaries'] = boundaries  # Use manual as primary
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved {len(boundaries)} phrase boundaries")
        return data
    
    def get_measure_offset(self, boundary_time, beat_times, tempo, measures_before):
        """Calculate the start time that is X measures before the boundary."""
        if not beat_times:
            # Fallback to simple time calculation
            if tempo and tempo > 0:
                seconds_per_measure = (60.0 / tempo) * 4  # 4 beats per measure
                return max(0, boundary_time - (measures_before * seconds_per_measure))
            else:
                # Default fallback
                return max(0, boundary_time - (measures_before * 8))  # Assume ~120 BPM
        
        # Find the beat closest to the boundary
        import numpy as np
        beat_array = np.array(beat_times)
        distances = np.abs(beat_array - boundary_time)
        boundary_beat_idx = np.argmin(distances)
        
        # Calculate how many beats to go back (4 beats per measure)
        beats_before = measures_before * 4
        start_beat_idx = max(0, boundary_beat_idx - beats_before)
        
        # Return the time of that beat
        if start_beat_idx < len(beat_times):
            return beat_times[start_beat_idx]
        else:
            # Fallback if we don't have enough beats
            # Use the actual beat spacing near the boundary for better accuracy
            if boundary_beat_idx > 0:
                # Calculate average beat interval near the boundary
                nearby_beats = beat_times[max(0, boundary_beat_idx-4):boundary_beat_idx+1]
                if len(nearby_beats) > 1:
                    intervals = [nearby_beats[i+1] - nearby_beats[i] for i in range(len(nearby_beats)-1)]
                    avg_beat_interval = sum(intervals) / len(intervals)
                    measure_duration = avg_beat_interval * 4
                    return max(0, boundary_time - (measures_before * measure_duration))
            
            # Final fallback
            if tempo and tempo > 0:
                seconds_per_measure = (60.0 / tempo) * 4
                return max(0, boundary_time - (measures_before * seconds_per_measure))
            else:
                return max(0, boundary_time - (measures_before * 8))