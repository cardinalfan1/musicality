import os
import random
from pathlib import Path
import yt_dlp

class YouTubeManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.playlist_url = self.config.get('youtube_playlist_url')
        self.cache_dir = Path(self.config.get('audio_cache_dir', './audio_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.cache_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        cookies_path = self.config.get('browser_cookies_path')
        if cookies_path and os.path.exists(cookies_path):
            self.ydl_opts['cookiefile'] = cookies_path
        
        username = self.config.get('youtube_username')
        password = self.config.get('youtube_password')
        if username and password:
            self.ydl_opts['username'] = username
            self.ydl_opts['password'] = password
    
    def get_playlist_info(self):
        """Get playlist metadata only (no actual video downloads)."""
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            # extract_flat=True means it only gets metadata, not actual videos
            playlist_info = ydl.extract_info(self.playlist_url, download=False)
            return playlist_info.get('entries', [])
    
    def get_random_song(self):
        songs = self.get_playlist_info()
        if not songs:
            raise ValueError("No songs found in playlist")
        return random.choice(songs)
    
    def download_song(self, song_info):
        video_id = song_info.get('id') or song_info.get('url')
        title = song_info.get('title', 'Unknown')
        
        cached_path = self.cache_dir / f"{title}.mp3"
        if cached_path.exists():
            print(f"Using cached: {title}")
            return str(cached_path)
        
        print(f"Downloading: {title}")
        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else song_info.get('url')
        
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            mp3_path = filename.rsplit('.', 1)[0] + '.mp3'
            return mp3_path
    
    def download_random_song(self):
        song = self.get_random_song()
        return self.download_song(song), song
    
    def get_playlist_videos(self):
        """Get all videos from the playlist."""
        return self.get_playlist_info()
    
    def download_specific_song(self, video_info):
        """Download a specific song from the playlist."""
        return self.download_song(video_info), video_info