import sys
import random
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, QProgressBar,
                            QTextEdit, QGroupBox, QDialog, QLineEdit, QFormLayout,
                            QDialogButtonBox, QComboBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor

from src.youtube_manager import YouTubeManager
from src.simple_analyzer import SimpleAnalyzer
from src.playback_engine import PlaybackEngine
from src.config_manager import ConfigManager
from src.timeline_widget import TimelineWidget
from src.training_mode import TrainingModeDialog

class SettingsDialog(QDialog):
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QFormLayout()
        
        self.playlist_input = QLineEdit(self.config.get('youtube_playlist_url', ''))
        self.playlist_input.setPlaceholderText("https://www.youtube.com/playlist?list=...")
        layout.addRow("YouTube Playlist URL:", self.playlist_input)
        
        self.username_input = QLineEdit(self.config.get('youtube_username', ''))
        self.username_input.setPlaceholderText("Optional: for private playlists")
        layout.addRow("YouTube Username:", self.username_input)
        
        self.password_input = QLineEdit(self.config.get('youtube_password', ''))
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Optional: for private playlists")
        layout.addRow("YouTube Password:", self.password_input)
        
        cookies_layout = QHBoxLayout()
        self.cookies_input = QLineEdit(self.config.get('browser_cookies_path', ''))
        self.cookies_input.setPlaceholderText("Optional: browser cookies file path")
        cookies_layout.addWidget(self.cookies_input)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_cookies)
        cookies_layout.addWidget(browse_button)
        
        layout.addRow("Browser Cookies:", cookies_layout)
        
        self.measures_combo = QComboBox()
        self.measures_combo.addItems(["Random (2-8 measures)", "2 measures", "4 measures", "6 measures", "8 measures"])
        current_measures = self.config.get('measures_before', 'random')
        if current_measures == 'random':
            index = 0
        elif current_measures in [2, 4, 6, 8]:
            index = [2, 4, 6, 8].index(current_measures) + 1
        else:
            index = 0
        self.measures_combo.setCurrentIndex(index)
        layout.addRow("Play before phrase change:", self.measures_combo)
        
        cache_layout = QHBoxLayout()
        self.cache_input = QLineEdit(self.config.get('audio_cache_dir', './audio_cache'))
        cache_layout.addWidget(self.cache_input)
        
        cache_browse = QPushButton("Browse...")
        cache_browse.clicked.connect(self.browse_cache)
        cache_layout.addWidget(cache_browse)
        
        layout.addRow("Audio Cache Directory:", cache_layout)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(buttons)
        
        self.setLayout(main_layout)
    
    def browse_cookies(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Browser Cookies File", "", "All Files (*)"
        )
        if file_path:
            self.cookies_input.setText(file_path)
    
    def browse_cache(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Cache Directory"
        )
        if dir_path:
            self.cache_input.setText(dir_path)
    
    def save_settings(self):
        measures_text = self.measures_combo.currentText()
        if "Random" in measures_text:
            measures = 'random'
        else:
            measures = int(measures_text.split()[0])
        
        self.config.update({
            'youtube_playlist_url': self.playlist_input.text(),
            'youtube_username': self.username_input.text(),
            'youtube_password': self.password_input.text(),
            'browser_cookies_path': self.cookies_input.text(),
            'audio_cache_dir': self.cache_input.text(),
            'analysis_cache_dir': self.cache_input.text().replace('audio_cache', 'analysis_cache'),
            'measures_before': measures
        })
        
        self.accept()

class AnalysisThread(QThread):
    finished = pyqtSignal(dict, dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    
    def __init__(self, youtube_manager, analyzer, specific_video=None):
        super().__init__()
        self.youtube_manager = youtube_manager
        self.analyzer = analyzer
        self.specific_video = specific_video
    
    def run(self):
        try:
            if self.specific_video:
                self.status.emit(f"Downloading: {self.specific_video.get('title', 'Unknown')}")
                audio_path, song_info = self.youtube_manager.download_specific_song(self.specific_video)
            else:
                self.status.emit("Fetching random song from playlist...")
                audio_path, song_info = self.youtube_manager.download_random_song()
            
            self.status.emit(f"Analyzing: {song_info.get('title', 'Unknown')}")
            # Try to load from advanced analyzer cache first
            analysis = self.analyzer.analyze_song(audio_path)
            
            self.finished.emit(analysis, song_info)
        except Exception as e:
            self.error.emit(str(e))

class MusicilityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.youtube_manager = YouTubeManager(self.config_manager)
        self.analyzer = SimpleAnalyzer(self.config_manager)
        self.playback_engine = PlaybackEngine()
        
        self.current_analysis = None
        self.current_song_info = None
        self.current_segment = None
        self.current_boundary_idx = 0
        self.phrase_flash_timer = QTimer()
        self.phrase_flash_timer.timeout.connect(self.reset_phrase_indicator)
        
        # Playlist cache
        self.playlist_cache = None
        self.playlist_cache_time = None
        
        self.init_ui()
        self.setup_callbacks()
        
        if not self.config_manager.get('youtube_playlist_url'):
            self.show_settings()
    
    def init_ui(self):
        self.setWindowTitle("Musicality Training App")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)
        layout.addWidget(settings_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        info_group = QGroupBox("Current Song")
        info_layout = QVBoxLayout()
        
        self.song_label = QLabel("No song loaded")
        self.song_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_layout.addWidget(self.song_label)
        
        self.segment_label = QLabel("")
        info_layout.addWidget(self.segment_label)
        
        self.boundaries_label = QLabel("")
        info_layout.addWidget(self.boundaries_label)
        
        self.tempo_label = QLabel("")
        info_layout.addWidget(self.tempo_label)
        
        self.beat_label = QLabel("")
        self.beat_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.beat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.beat_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Phrase change indicator
        self.phrase_indicator = QWidget()
        self.phrase_indicator.setFixedHeight(80)
        self.phrase_indicator.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 10px;
                border: 2px solid #444;
            }
        """)
        
        indicator_layout = QVBoxLayout(self.phrase_indicator)
        
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.countdown_label.setStyleSheet("color: white;")
        indicator_layout.addWidget(self.countdown_label)
        
        self.phrase_status_label = QLabel("Waiting...")
        self.phrase_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phrase_status_label.setFont(QFont("Arial", 14))
        self.phrase_status_label.setStyleSheet("color: #888;")
        indicator_layout.addWidget(self.phrase_status_label)
        
        layout.addWidget(self.phrase_indicator)
        
        # Timeline widget
        self.timeline = TimelineWidget()
        layout.addWidget(self.timeline)
        
        progress_layout = QHBoxLayout()
        self.time_label = QLabel("0:00")
        progress_layout.addWidget(self.time_label)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.duration_label = QLabel("0:00")
        progress_layout.addWidget(self.duration_label)
        
        layout.addLayout(progress_layout)
        
        controls_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_phrase)
        controls_layout.addWidget(self.prev_button)
        
        self.play_button = QPushButton("Play/Pause")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_phrase)
        controls_layout.addWidget(self.next_button)
        
        self.repeat_button = QPushButton("Repeat")
        self.repeat_button.clicked.connect(self.repeat_segment)
        controls_layout.addWidget(self.repeat_button)
        
        self.repeat_random_button = QPushButton("Repeat (Random Measures)")
        self.repeat_random_button.clicked.connect(self.repeat_with_random_measures)
        controls_layout.addWidget(self.repeat_random_button)
        
        layout.addLayout(controls_layout)
        
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("70%")
        volume_layout.addWidget(self.volume_label)
        
        layout.addLayout(volume_layout)
        
        action_layout = QHBoxLayout()
        
        self.new_song_button = QPushButton("Load New Random Song")
        self.new_song_button.clicked.connect(self.load_new_song)
        action_layout.addWidget(self.new_song_button)
        
        self.select_song_button = QPushButton("Select Song from Playlist")
        self.select_song_button.clicked.connect(self.select_from_playlist)
        action_layout.addWidget(self.select_song_button)
        
        self.new_phrase_button = QPushButton("Random Phrase (Same Song)")
        self.new_phrase_button.clicked.connect(self.random_phrase_same_song)
        self.new_phrase_button.setEnabled(False)
        action_layout.addWidget(self.new_phrase_button)
        
        layout.addLayout(action_layout)
        
        # Training mode button
        training_layout = QHBoxLayout()
        
        self.training_mode_button = QPushButton("🎯 Training Mode - Mark Phrases")
        self.training_mode_button.clicked.connect(self.enter_training_mode)
        self.training_mode_button.setEnabled(False)
        self.training_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b00;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #ff8c00;
            }
        """)
        training_layout.addWidget(self.training_mode_button)
        
        self.clear_marks_button = QPushButton("Clear All Marks")
        self.clear_marks_button.clicked.connect(self.clear_manual_marks)
        self.clear_marks_button.setEnabled(False)
        training_layout.addWidget(self.clear_marks_button)
        
        layout.addLayout(training_layout)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_progress)
        self.update_timer.start(100)
    
    def setup_callbacks(self):
        self.playback_engine.set_callback('on_complete', self.on_segment_complete)
    
    def show_settings(self):
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            self.youtube_manager = YouTubeManager(self.config_manager)
            self.analyzer = AudioAnalyzer(self.config_manager)
            self.status_text.append("Settings updated successfully")
    
    def load_new_song(self):
        if not self.config_manager.get('youtube_playlist_url'):
            self.status_text.append("Please configure YouTube playlist URL in Settings")
            self.show_settings()
            return
        
        self.new_song_button.setEnabled(False)
        self.status_text.append("Loading new song...")
        
        self.analysis_thread = AnalysisThread(self.youtube_manager, self.analyzer)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.status.connect(lambda msg: self.status_text.append(msg))
        self.analysis_thread.start()
    
    def select_from_playlist(self):
        """Let user select a specific song from the playlist."""
        if not self.config_manager.get('youtube_playlist_url'):
            self.status_text.append("Please configure YouTube playlist URL in Settings")
            self.show_settings()
            return
        
        # Check if we need to refresh the cache (10 minutes = 600 seconds)
        import time
        current_time = time.time()
        cache_expired = (self.playlist_cache is None or 
                        self.playlist_cache_time is None or 
                        current_time - self.playlist_cache_time > 600)
        
        # Get playlist info (from cache or fresh)
        if cache_expired:
            self.status_text.append("Fetching playlist...")
            try:
                playlist_videos = self.youtube_manager.get_playlist_videos()
                if not playlist_videos:
                    self.status_text.append("No videos found in playlist")
                    return
                # Cache the results
                self.playlist_cache = playlist_videos
                self.playlist_cache_time = current_time
                self.status_text.append(f"Loaded {len(playlist_videos)} songs from playlist")
            except Exception as e:
                self.status_text.append(f"Error fetching playlist: {str(e)}")
                return
        else:
            playlist_videos = self.playlist_cache
            self.status_text.append(f"Using cached playlist ({len(playlist_videos)} songs)")
        
        try:
            
            # Create a dialog with autocomplete search
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListWidget, QDialogButtonBox, QListWidgetItem, QLabel
            from PyQt6.QtCore import Qt
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Song from Playlist")
            dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout()
            
            # Add search field with label
            search_label = QLabel("Search for a song:")
            layout.addWidget(search_label)
            
            search_field = QLineEdit()
            search_field.setPlaceholderText("Type to search...")
            layout.addWidget(search_field)
            
            # List widget for filtered results
            list_widget = QListWidget()
            
            # Store video data for filtering (not QListWidgetItem objects)
            all_videos = []
            for video in playlist_videos:
                title = video.get('title', 'Unknown')
                all_videos.append((title, video))
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, video)
                list_widget.addItem(item)
            
            layout.addWidget(list_widget)
            
            # Filter function
            def filter_songs():
                search_text = search_field.text().lower()
                list_widget.clear()
                
                for title, video in all_videos:
                    if search_text in title.lower():
                        # Create new item each time
                        new_item = QListWidgetItem(title)
                        new_item.setData(Qt.ItemDataRole.UserRole, video)
                        list_widget.addItem(new_item)
                
                # Auto-select first item if there's only one match
                if list_widget.count() == 1:
                    list_widget.setCurrentRow(0)
            
            # Connect search field to filter
            search_field.textChanged.connect(filter_songs)
            
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                      QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            dialog.setLayout(layout)
            
            # Focus on search field
            search_field.setFocus()
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_item = list_widget.currentItem()
                if selected_item:
                    video_info = selected_item.data(Qt.ItemDataRole.UserRole)
                    self.load_specific_song(video_info)
        except Exception as e:
            self.status_text.append(f"Error fetching playlist: {str(e)}")
    
    def load_specific_song(self, video_info):
        """Load a specific song from the playlist."""
        self.new_song_button.setEnabled(False)
        self.select_song_button.setEnabled(False)
        self.status_text.append(f"Loading: {video_info.get('title', 'Unknown')}...")
        
        # Create a custom thread that loads this specific song
        self.analysis_thread = AnalysisThread(self.youtube_manager, self.analyzer, specific_video=video_info)
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.status.connect(lambda msg: self.status_text.append(msg))
        self.analysis_thread.start()
    
    def on_analysis_complete(self, analysis, song_info):
        self.current_analysis = analysis
        self.current_song_info = song_info
        
        title = song_info.get('title', 'Unknown')
        self.song_label.setText(f"Song: {title}")
        
        # Check for manual boundaries
        manual_boundaries = analysis.get('manual_boundaries', [])
        if manual_boundaries:
            boundaries = manual_boundaries
            self.boundaries_label.setText(f"✓ {len(boundaries)} phrase boundaries trained")
        else:
            boundaries = []
            self.boundaries_label.setText("⚠️ No phrases marked - Click Training Mode to start!")
        
        tempo = analysis.get('tempo', 0)
        self.tempo_label.setText(f"Tempo: {tempo:.1f} BPM")
        
        # Always enable training mode
        self.training_mode_button.setEnabled(True)
        self.clear_marks_button.setEnabled(bool(manual_boundaries) and len(manual_boundaries) > 0)
        
        if manual_boundaries:
            # If we have training data, start practice
            self.play_random_phrase()
            self.new_phrase_button.setEnabled(True)
            self.repeat_random_button.setEnabled(True)
            self.status_text.append(f"✓ Ready to practice with {len(manual_boundaries)} phrases!")
        else:
            # No training data yet
            self.new_phrase_button.setEnabled(False)
            self.repeat_random_button.setEnabled(False)
            self.status_text.append("Welcome! To get started:")
            self.status_text.append("1. Click '🎯 Training Mode'")
            self.status_text.append("2. Play the song and press SPACEBAR at each phrase change")
            self.status_text.append("3. Save your marks to start practicing!")
        
        self.new_song_button.setEnabled(True)
        self.select_song_button.setEnabled(True)
    
    def on_analysis_error(self, error_msg):
        self.status_text.append(f"Error: {error_msg}")
        self.new_song_button.setEnabled(True)
        self.select_song_button.setEnabled(True)
    
    def play_random_phrase(self):
        if not self.current_analysis or not self.current_analysis['boundaries']:
            return
        
        boundaries = self.current_analysis['boundaries']
        self.current_boundary_idx = random.randint(0, len(boundaries) - 1)
        boundary = boundaries[self.current_boundary_idx]
        
        audio_path = self.current_analysis['audio_path']
        tempo = self.current_analysis.get('tempo', 120)
        beat_times = self.current_analysis.get('beat_times', [])
        
        measures_setting = self.config_manager.get('measures_before', 'random')
        if measures_setting == 'random':
            measures_before = random.choice([2, 4, 6, 8])
        else:
            measures_before = measures_setting
        
        start_time = self.analyzer.get_measure_offset(boundary, beat_times, tempo, measures_before)
        
        start, end, boundary_time = self.playback_engine.play_phrase_segment(
            audio_path, boundary, start_time
        )
        
        self.current_segment = (start, end, boundary_time)
        self.segment_label.setText(
            f"Playing: {measures_before} measures before phrase change at {boundary_time:.1f}s"
        )
        
        # Update timeline
        self.timeline.set_segment(start, end, boundary_time)
        self.timeline.set_playing(True)
        self.status_text.append(f"Playing {measures_before} measures before boundary at {boundary_time:.1f}s")
        
        # Update timeline
        self.timeline.set_segment(start, end, boundary_time)
        self.timeline.set_playing(True)
    
    def random_phrase_same_song(self):
        if self.current_analysis:
            self.playback_engine.stop()
            self.play_random_phrase()
    
    def previous_phrase(self):
        if not self.current_analysis or not self.current_analysis['boundaries']:
            return
        
        boundaries = self.current_analysis['boundaries']
        self.current_boundary_idx = (self.current_boundary_idx - 1) % len(boundaries)
        boundary = boundaries[self.current_boundary_idx]
        
        audio_path = self.current_analysis['audio_path']
        tempo = self.current_analysis.get('tempo', 120)
        beat_times = self.current_analysis.get('beat_times', [])
        
        measures_setting = self.config_manager.get('measures_before', 'random')
        if measures_setting == 'random':
            measures_before = random.choice([2, 4, 6, 8])
        else:
            measures_before = measures_setting
        
        start_time = self.analyzer.get_measure_offset(boundary, beat_times, tempo, measures_before)
        
        start, end, boundary_time = self.playback_engine.play_phrase_segment(
            audio_path, boundary, start_time
        )
        
        self.current_segment = (start, end, boundary_time)
        self.segment_label.setText(
            f"Playing: {measures_before} measures before phrase change at {boundary_time:.1f}s"
        )
        
        # Update timeline
        self.timeline.set_segment(start, end, boundary_time)
        self.timeline.set_playing(True)
    
    def next_phrase(self):
        if not self.current_analysis or not self.current_analysis['boundaries']:
            return
        
        boundaries = self.current_analysis['boundaries']
        self.current_boundary_idx = (self.current_boundary_idx + 1) % len(boundaries)
        boundary = boundaries[self.current_boundary_idx]
        
        audio_path = self.current_analysis['audio_path']
        tempo = self.current_analysis.get('tempo', 120)
        beat_times = self.current_analysis.get('beat_times', [])
        
        measures_setting = self.config_manager.get('measures_before', 'random')
        if measures_setting == 'random':
            measures_before = random.choice([2, 4, 6, 8])
        else:
            measures_before = measures_setting
        
        start_time = self.analyzer.get_measure_offset(boundary, beat_times, tempo, measures_before)
        
        start, end, boundary_time = self.playback_engine.play_phrase_segment(
            audio_path, boundary, start_time
        )
        
        self.current_segment = (start, end, boundary_time)
        self.segment_label.setText(
            f"Playing: {measures_before} measures before phrase change at {boundary_time:.1f}s"
        )
        
        # Update timeline
        self.timeline.set_segment(start, end, boundary_time)
        self.timeline.set_playing(True)
    
    def repeat_segment(self):
        if self.current_segment and self.current_analysis:
            start, end, boundary = self.current_segment
            audio_path = self.current_analysis['audio_path']
            self.playback_engine.play_segment(audio_path, start, end)
            self.status_text.append("Repeating current segment")
            self.timeline.set_playing(True)
    
    def toggle_playback(self):
        if self.playback_engine.is_playing:
            self.playback_engine.pause()
            self.status_text.append("Paused")
            self.timeline.set_playing(False)
        else:
            self.playback_engine.resume()
            self.status_text.append("Resumed")
            self.timeline.set_playing(True)
    
    def change_volume(self, value):
        volume = value / 100
        self.playback_engine.set_volume(volume)
        self.volume_label.setText(f"{value}%")
    
    def update_progress(self):
        if self.playback_engine.is_playing and self.current_segment:
            position = self.playback_engine.get_position()
            start, end, boundary = self.current_segment
            
            progress = int(((position - start) / (end - start)) * 100)
            self.progress_bar.setValue(min(100, max(0, progress)))
            
            # Update beat display
            if self.current_analysis:
                beat_times = self.current_analysis.get('beat_times', [])
                downbeats = self.current_analysis.get('downbeats', [])
                
                if beat_times:
                    # Find closest beat
                    closest_beat_idx = min(range(len(beat_times)), 
                                         key=lambda i: abs(beat_times[i] - position),
                                         default=-1)
                    
                    if closest_beat_idx >= 0:
                        beat_time = beat_times[closest_beat_idx]
                        # Check if we're close to this beat (within 0.1 seconds)
                        if abs(beat_time - position) < 0.1:
                            # Check if it's a downbeat
                            if beat_time in downbeats:
                                self.beat_label.setText("● DOWNBEAT")
                                self.beat_label.setStyleSheet("color: #ff4444;")
                            else:
                                beat_number = (closest_beat_idx % 4) + 1
                                self.beat_label.setText(f"○ Beat {beat_number}")
                                self.beat_label.setStyleSheet("color: #44ff44;")
                        else:
                            self.beat_label.setText("")
                            self.beat_label.setStyleSheet("color: #888;")
            
            self.time_label.setText(f"{int(position//60)}:{int(position%60):02d}")
            self.duration_label.setText(f"{int(end//60)}:{int(end%60):02d}")
            
            # Update timeline
            self.timeline.set_position(position)
            
            # Update phrase change indicator
            time_to_boundary = boundary - position
            
            if time_to_boundary > 0 and time_to_boundary <= 5:
                # Show countdown
                self.countdown_label.setText(f"{time_to_boundary:.1f}")
                self.phrase_status_label.setText("Phrase change in...")
                self.phrase_status_label.setStyleSheet("color: #ffaa00;")
                
                # Make indicator more prominent as we get closer
                if time_to_boundary <= 1:
                    self.phrase_indicator.setStyleSheet("""
                        QWidget {
                            background-color: #553300;
                            border-radius: 10px;
                            border: 3px solid #ff8800;
                        }
                    """)
            elif time_to_boundary <= 0 and time_to_boundary > -0.5:
                # Flash at phrase change
                self.countdown_label.setText("NOW!")
                self.phrase_status_label.setText("PHRASE CHANGE!")
                self.phrase_status_label.setStyleSheet("color: #00ff00;")
                self.phrase_indicator.setStyleSheet("""
                    QWidget {
                        background-color: #003300;
                        border-radius: 10px;
                        border: 3px solid #00ff00;
                    }
                """)
                if not self.phrase_flash_timer.isActive():
                    self.phrase_flash_timer.start(500)
            elif time_to_boundary < -0.5:
                # After phrase change
                self.countdown_label.setText("")
                self.phrase_status_label.setText("Playing after phrase change")
                self.phrase_status_label.setStyleSheet("color: #888;")
                self.reset_phrase_indicator()
            else:
                # Before countdown starts
                self.countdown_label.setText("")
                self.phrase_status_label.setText("Approaching phrase change...")
                self.phrase_status_label.setStyleSheet("color: #888;")
                self.reset_phrase_indicator()
    
    def reset_phrase_indicator(self):
        self.phrase_indicator.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 10px;
                border: 2px solid #444;
            }
        """)
        self.phrase_flash_timer.stop()
    
    def on_segment_complete(self):
        self.status_text.append("Segment complete")
        self.progress_bar.setValue(100)
    
    def repeat_with_random_measures(self):
        """Repeat the current phrase boundary but with a new random measure offset."""
        if not self.current_segment or not self.current_analysis:
            return
        
        # Get the current boundary time
        _, _, boundary = self.current_segment
        audio_path = self.current_analysis['audio_path']
        tempo = self.current_analysis.get('tempo', 120)
        beat_times = self.current_analysis.get('beat_times', [])
        
        # Force random selection of measures
        measures_before = random.choice([2, 4, 6, 8])
        start_time = self.analyzer.get_measure_offset(boundary, beat_times, tempo, measures_before)
        
        start, end, boundary_time = self.playback_engine.play_phrase_segment(
            audio_path, boundary, start_time
        )
        
        self.current_segment = (start, end, boundary_time)
        self.segment_label.setText(
            f"Playing: {measures_before} measures before phrase change at {boundary_time:.1f}s"
        )
        self.status_text.append(f"Repeating with {measures_before} measures before boundary")
        
        # Update timeline
        self.timeline.set_segment(start, end, boundary_time)
        self.timeline.set_playing(True)
    
    def enter_training_mode(self):
        """Enter training mode to mark phrase boundaries."""
        if not self.current_analysis:
            self.status_text.append("Load a song first!")
            return
        
        # Stop current playback
        self.playback_engine.stop()
        
        audio_path = self.current_analysis['audio_path']
        title = self.current_song_info.get('title', 'Unknown') if self.current_song_info else 'Unknown'
        duration = self.current_analysis.get('duration', 300)
        beat_times = self.current_analysis.get('beat_times', [])
        downbeats = self.current_analysis.get('downbeats', [])
        
        # Open training dialog with beat times for snapping
        dialog = TrainingModeDialog(audio_path, title, duration, beat_times, downbeats, self)
        dialog.phrases_marked.connect(self.on_training_complete)
        
        # Load existing marks if any
        existing_marks = self.current_analysis.get('manual_boundaries', [])
        if existing_marks:
            dialog.phrase_marks = existing_marks.copy()
            dialog.update_marks_list()
            dialog.marks_group.setTitle(f"Marked Phrases ({len(existing_marks)})")
        
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.setFocus()
        dialog.exec()
    
    def on_training_complete(self, phrase_marks):
        """Handle completion of training mode."""
        if not phrase_marks:
            return
        
        # Save the manual boundaries
        audio_path = self.current_analysis['audio_path']
        self.current_analysis = self.analyzer.save_manual_boundaries(audio_path, phrase_marks)
        self.current_analysis['boundaries'] = phrase_marks
        self.current_analysis['manual_boundaries'] = phrase_marks
        
        # Update UI
        self.boundaries_label.setText(f"Using {len(phrase_marks)} trained phrase boundaries")
        self.status_text.append(f"Training complete! Saved {len(phrase_marks)} phrase boundaries")
        self.status_text.append("Ready to practice!")
        
        # Enable practice buttons
        self.new_phrase_button.setEnabled(True)
        self.clear_marks_button.setEnabled(True)
        
        # Start practice with first phrase
        if phrase_marks:
            self.play_random_phrase()
    
    def clear_manual_marks(self):
        """Clear all manual phrase marks."""
        if not self.current_analysis:
            return
        
        reply = QMessageBox.question(self, "Clear Training Data",
                                    "Remove all trained phrase marks for this song?",
                                    QMessageBox.StandardButton.Yes |
                                    QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            audio_path = self.current_analysis['audio_path']
            self.current_analysis = self.analyzer.save_manual_boundaries(audio_path, [])
            self.current_analysis['manual_boundaries'] = []
            
            # Update UI
            self.boundaries_label.setText("No phrases marked - use Training Mode!")
            self.status_text.append("Cleared all training data")
            self.clear_marks_button.setEnabled(False)
    

def main():
    app = QApplication(sys.argv)
    window = MusicilityApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()