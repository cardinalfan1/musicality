import pygame
import random
import threading
import time

class PlaybackEngine:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.current_song = None
        self.current_position = 0
        self.is_playing = False
        self.volume = 0.7
        pygame.mixer.music.set_volume(self.volume)
        
        self.playback_start_time = 0
        self.playback_offset = 0
        
        self.callbacks = {
            'on_complete': None,
            'on_position_update': None
        }
    
    def load_song(self, audio_path):
        self.current_song = audio_path
        pygame.mixer.music.load(audio_path)
        self.current_position = 0
    
    def play_segment(self, audio_path, start_time, end_time):
        self.load_song(audio_path)
        
        self.playback_offset = start_time
        pygame.mixer.music.play(start=start_time)
        self.playback_start_time = time.time()
        self.is_playing = True
        
        def stop_at_end():
            while self.is_playing:
                current = self.get_position()
                if current >= end_time:
                    self.stop()
                    if self.callbacks['on_complete']:
                        self.callbacks['on_complete']()
                    break
                
                if self.callbacks['on_position_update']:
                    self.callbacks['on_position_update'](current, end_time)
                
                time.sleep(0.1)
        
        threading.Thread(target=stop_at_end, daemon=True).start()
    
    def play_phrase_segment(self, audio_path, boundary_time, start_time, after_seconds=10):
        end_time = boundary_time + after_seconds
        
        print(f"Playing from {start_time:.1f}s to {end_time:.1f}s (boundary at {boundary_time:.1f}s)")
        self.play_segment(audio_path, start_time, end_time)
        
        return start_time, end_time, boundary_time
    
    def get_position(self):
        if not self.is_playing:
            return self.current_position
        
        elapsed = time.time() - self.playback_start_time
        return self.playback_offset + elapsed
    
    def pause(self):
        if self.is_playing:
            self.current_position = self.get_position()
            pygame.mixer.music.pause()
            self.is_playing = False
    
    def resume(self):
        if not self.is_playing and self.current_song:
            pygame.mixer.music.unpause()
            self.playback_start_time = time.time() - (self.current_position - self.playback_offset)
            self.is_playing = True
    
    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_position = 0
    
    def set_volume(self, volume):
        self.volume = max(0, min(1, volume))
        pygame.mixer.music.set_volume(self.volume)
    
    def set_callback(self, event, callback):
        if event in self.callbacks:
            self.callbacks[event] = callback