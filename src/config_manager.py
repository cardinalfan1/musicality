import json
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config_file = Path.home() / '.musicality_config.json'
        self.config = self.load_config()
    
    def load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {
            'youtube_playlist_url': '',
            'youtube_username': '',
            'youtube_password': '',
            'browser_cookies_path': '',
            'audio_cache_dir': './audio_cache',
            'analysis_cache_dir': './analysis_cache',
            'measures_before': 4
        }
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        self.save_config()
    
    def update(self, updates):
        self.config.update(updates)
        self.save_config()