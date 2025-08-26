from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QListWidget, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
import pygame

class TrainingModeDialog(QDialog):
    phrases_marked = pyqtSignal(list)
    training_complete = pyqtSignal(dict)  # Emit all training data
    
    def __init__(self, audio_path, song_title, duration, beat_times=None, downbeats=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.song_title = song_title
        self.duration = duration
        self.beat_times = beat_times or []  # All beat positions for snapping
        self.downbeats = downbeats or []  # Boom beats (low frequency percussion)
        self.phrase_marks = []
        self.is_playing = False
        self.start_time = 0
        self.pause_position = 0  # Track pause position
        self.beat_taps = []  # For manual beat calibration
        self.manual_tempo = None
        self.tempo_sections = []  # List of (start_time, end_time, tempo, beat_times) for different tempo sections
        self.has_tapped_before = False  # Track if user has ever tapped beats
        
        self.setWindowTitle(f"Training Mode - {song_title}")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        self.init_ui()
        self.init_audio()
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        
        # Timer for updating position
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_position)
        self.update_timer.start(100)
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "TRAINING MODE\n\n"
            "1. Click 'Play Full Song' to start\n"
            "2. Press SPACEBAR at each phrase change\n"
            "3. Use ← → arrow keys to skip 5 seconds\n"
            "4. Press B to tap the beat (recalibrate beat detection)\n"
            "5. Click 'Save & Exit' when done\n\n"
            "Tips:\n"
            "• Marks automatically snap to the nearest beat\n"
            "• Marks within 10 seconds replace old ones for fine-tuning\n"
            "• If beats seem off, tap B key 8+ times to the rhythm"
        )
        instructions.setFont(QFont("Arial", 11))
        instructions.setStyleSheet("background-color: #333; color: white; padding: 15px; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Current position display
        position_group = QGroupBox("Current Position")
        position_layout = QVBoxLayout()
        
        self.position_label = QLabel("0:00 / 0:00")
        self.position_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        position_layout.addWidget(self.position_label)
        
        self.last_mark_label = QLabel("Last mark: None")
        self.last_mark_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        position_layout.addWidget(self.last_mark_label)
        
        # Manual beat calibration status
        self.beat_calibration_label = QLabel("Press B to tap beat for calibration")
        self.beat_calibration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.beat_calibration_label.setStyleSheet("color: #ffaa44; font-weight: bold;")
        position_layout.addWidget(self.beat_calibration_label)
        
        # Beat indicator
        self.beat_indicator = QLabel("")
        self.beat_indicator.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.beat_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.beat_indicator.setStyleSheet("color: #888;")
        self.beat_indicator.setFixedHeight(40)  # Fixed height to prevent jumping
        position_layout.addWidget(self.beat_indicator)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play Full Song")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.mark_button = QPushButton("Mark Phrase (SPACE)")
        self.mark_button.clicked.connect(self.mark_phrase)
        controls_layout.addWidget(self.mark_button)
        
        self.undo_button = QPushButton("Undo Last Mark")
        self.undo_button.clicked.connect(self.undo_last_mark)
        controls_layout.addWidget(self.undo_button)
        
        layout.addLayout(controls_layout)
        
        # Marked phrases list
        marks_group = QGroupBox(f"Marked Phrases ({len(self.phrase_marks)})")
        marks_layout = QVBoxLayout()
        
        self.marks_list = QListWidget()
        marks_layout.addWidget(self.marks_list)
        
        marks_group.setLayout(marks_layout)
        self.marks_group = marks_group
        layout.addWidget(marks_group)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all_marks)
        bottom_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("Save & Exit")
        self.save_button.clicked.connect(self.save_and_exit)
        self.save_button.setStyleSheet("background-color: #4CAF50; color: white;")
        bottom_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_dialog)
        bottom_layout.addWidget(self.cancel_button)
        
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)
    
    def init_audio(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_path)
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            print("Space key pressed")  # Debug
            self.mark_phrase()
            event.accept()
        elif event.key() == Qt.Key.Key_Right:
            print("Right arrow pressed")  # Debug
            self.skip_forward()
            event.accept()
        elif event.key() == Qt.Key.Key_Left:
            print("Left arrow pressed")  # Debug
            self.skip_backward()
            event.accept()
        elif event.key() == Qt.Key.Key_B:
            print("Beat tap")  # Debug
            self.tap_beat()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def toggle_playback(self):
        if not self.is_playing:
            if self.play_button.text() == "Resume":
                # Resume from where we paused
                pygame.mixer.music.unpause()
                self.is_playing = True
                self.play_button.setText("Pause")
                import time
                # Adjust playback_start to account for pause time
                current_pos = pygame.mixer.music.get_pos() / 1000.0  # Get position in seconds
                self.playback_start = time.time() - self.pause_position
            else:
                # Start from beginning
                pygame.mixer.music.play()
                self.is_playing = True
                self.play_button.setText("Pause")
                import time
                self.playback_start = time.time()
        else:
            # Pause and remember position
            self.pause_position = self.get_current_position()
            pygame.mixer.music.pause()
            self.is_playing = False
            self.play_button.setText("Resume")
    
    def mark_phrase(self):
        if not self.is_playing:
            QMessageBox.information(self, "Not Playing", "Start playing the song first!")
            return
        
        current_pos = self.get_current_position()
        
        # Snap to nearest beat if beat times are available
        snapped_pos = self.snap_to_beat(current_pos)
        
        # Check if within 10 seconds of existing mark - if so, replace it
        replaced = False
        new_marks = []
        for mark in self.phrase_marks:
            if abs(mark - snapped_pos) < 10.0:
                # Replace the old mark with the new one
                if not replaced:
                    new_marks.append(snapped_pos)
                    replaced = True
                    if abs(current_pos - snapped_pos) > 0.05:
                        self.last_mark_label.setText(f"Replaced {mark:.1f}s → {snapped_pos:.1f}s (snapped from {current_pos:.1f}s)")
                    else:
                        self.last_mark_label.setText(f"Replaced mark at {mark:.1f}s with {snapped_pos:.1f}s")
                # Skip the old mark (don't add it back)
            else:
                new_marks.append(mark)
        
        if not replaced:
            # Add as new mark if not replacing
            new_marks.append(snapped_pos)
            if abs(current_pos - snapped_pos) > 0.05:
                self.last_mark_label.setText(f"Marked: {snapped_pos:.1f}s (snapped from {current_pos:.1f}s)")
            else:
                self.last_mark_label.setText(f"Marked: {snapped_pos:.1f}s")
        
        self.phrase_marks = sorted(new_marks)
        
        # Update UI
        self.update_marks_list()
        self.marks_group.setTitle(f"Marked Phrases ({len(self.phrase_marks)})")
    
    def skip_forward(self):
        """Skip forward 5 seconds."""
        if self.play_button.text() == "Play Full Song":
            # Not started yet
            return
            
        if self.is_playing:
            current_pos = self.get_current_position()
        else:
            # If paused, use the pause position
            current_pos = self.pause_position
            
        new_pos = min(current_pos + 5, self.duration)
        
        # Show which tempo section we're in
        current_tempo = self.get_current_tempo(new_pos)
        tempo_str = f" ({current_tempo:.0f} BPM)" if current_tempo else ""
        
        if self.is_playing:
            # Playing - restart from new position
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=new_pos)
            import time
            self.playback_start = time.time() - new_pos
        else:
            # Paused - just update pause position
            self.pause_position = new_pos
            
        self.last_mark_label.setText(f"Skipped to {new_pos:.1f}s{tempo_str}")
        # Force immediate position update
        self.update_position()
    
    def skip_backward(self):
        """Skip backward 5 seconds."""
        if self.play_button.text() == "Play Full Song":
            # Not started yet
            return
            
        if self.is_playing:
            current_pos = self.get_current_position()
        else:
            # If paused, use the pause position
            current_pos = self.pause_position
            
        new_pos = max(current_pos - 5, 0)
        
        # Show which tempo section we're in
        current_tempo = self.get_current_tempo(new_pos)
        tempo_str = f" ({current_tempo:.0f} BPM)" if current_tempo else ""
        
        if self.is_playing:
            # Playing - restart from new position
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=new_pos)
            import time
            self.playback_start = time.time() - new_pos
        else:
            # Paused - just update pause position
            self.pause_position = new_pos
            
        self.last_mark_label.setText(f"Skipped to {new_pos:.1f}s{tempo_str}")
        # Force immediate position update
        self.update_position()
    
    def undo_last_mark(self):
        if self.phrase_marks:
            removed = self.phrase_marks.pop()
            self.update_marks_list()
            self.last_mark_label.setText(f"Removed mark at {removed:.1f}s")
            self.marks_group.setTitle(f"Marked Phrases ({len(self.phrase_marks)})")
    
    def clear_all_marks(self):
        reply = QMessageBox.question(self, "Clear All", 
                                    "Remove all phrase marks?",
                                    QMessageBox.StandardButton.Yes | 
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.phrase_marks = []
            self.update_marks_list()
            self.last_mark_label.setText("Last mark: None")
            self.marks_group.setTitle(f"Marked Phrases ({len(self.phrase_marks)})")
    
    def update_marks_list(self):
        self.marks_list.clear()
        for i, mark in enumerate(self.phrase_marks):
            time_str = f"{int(mark//60)}:{int(mark%60):02d}"
            self.marks_list.addItem(f"Phrase {i+1}: {time_str} ({mark:.1f}s)")
    
    def get_current_position(self):
        if not self.is_playing:
            return 0
        import time
        return time.time() - self.playback_start
    
    def tap_beat(self):
        """Record beat tap for manual tempo detection."""
        if not self.is_playing:
            QMessageBox.information(self, "Not Playing", "Start playing the song first to tap beats!")
            return
        
        current_time = self.get_current_position()
        
        # Check if this is a tempo change (user has tapped before and there's a recent phrase mark)
        is_tempo_change = False
        section_start = 0
        current_section_start = 0
        
        # Find which phrase section we're currently in
        if self.phrase_marks:
            for mark in sorted(self.phrase_marks, reverse=True):
                if mark <= current_time:
                    current_section_start = mark
                    break
        
        # Check if we need to start tempo detection for a new section
        if self.has_tapped_before:
            # Check if we have any taps
            if self.beat_taps:
                # Find which section our last tap was in
                last_tap_time = self.beat_taps[-1]
                last_tap_section = 0
                for mark in sorted(self.phrase_marks, reverse=True):
                    if mark <= last_tap_time:
                        last_tap_section = mark
                        break
                
                # If we're in a different section, start new tempo detection
                if current_section_start != last_tap_section:
                    section_start = current_section_start
                    is_tempo_change = True
                    self.beat_taps = []  # Clear for new section
                    self.beat_calibration_label.setText(f"Detecting tempo change for section starting at {section_start:.1f}s...")
                    self.beat_calibration_label.setStyleSheet("color: #ffff00; font-weight: bold;")
                    print(f"Starting new tempo section at phrase mark {section_start:.1f}s")
            else:
                # No previous taps in this session, check if we're after a phrase mark
                if current_section_start > 0:
                    section_start = current_section_start
                    is_tempo_change = True
        
        self.beat_taps.append(current_time)
        self.has_tapped_before = True
        print(f"Tap at {current_time:.2f}s, total taps: {len(self.beat_taps)}, is_tempo_change: {is_tempo_change}")
        
        # Visual feedback for tap
        self.beat_indicator.setText("◉ TAP!")
        self.beat_indicator.setStyleSheet("color: #ffff00; background-color: #444400; padding: 5px; border-radius: 5px;")
        QTimer.singleShot(100, lambda: self.beat_indicator.setText(""))
        
        # Keep only recent taps (last 16 beats)
        if len(self.beat_taps) > 16:
            self.beat_taps = self.beat_taps[-16:]
        
        # Calculate tempo from taps if we have enough
        if len(self.beat_taps) >= 4:
            intervals = [self.beat_taps[i+1] - self.beat_taps[i] for i in range(len(self.beat_taps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            if avg_interval > 0:
                new_tempo = 60.0 / avg_interval
                
                if is_tempo_change:
                    # Create a new tempo section starting from the last phrase mark
                    self.add_tempo_section(section_start, new_tempo, avg_interval)
                    self.beat_calibration_label.setText(f"✓ Tempo changed: {new_tempo:.1f} BPM from phrase at {section_start:.1f}s")
                    self.beat_calibration_label.setStyleSheet("color: #44ff44; font-weight: bold;")
                    print(f"Tempo change detected: {new_tempo:.1f} BPM starting at {section_start:.1f}s")
                else:
                    # First tempo calibration or regular calibration
                    self.manual_tempo = new_tempo
                    
                    # Regenerate beat grid based on manual tempo
                    first_beat = self.beat_taps[-1]
                    self.beat_times = []
                    
                    # Generate beats backward from current position
                    beat_time = first_beat
                    while beat_time > 0:
                        self.beat_times.insert(0, beat_time)
                        beat_time -= avg_interval
                    
                    # Generate beats forward from current position
                    beat_time = first_beat + avg_interval
                    while beat_time < self.duration:
                        self.beat_times.append(beat_time)
                        beat_time += avg_interval
                    
                    # Update boom/tick pattern (alternating)
                    self.downbeats = [self.beat_times[i] for i in range(0, len(self.beat_times), 2)]
                    
                    self.beat_calibration_label.setText(f"✓ Beat calibrated: {new_tempo:.1f} BPM")
                    self.beat_calibration_label.setStyleSheet("color: #44ff44; font-weight: bold;")
                    print(f"Beat grid regenerated at {new_tempo:.1f} BPM from {len(self.beat_taps)} taps")
        else:
            remaining = 4 - len(self.beat_taps)
            if is_tempo_change:
                self.beat_calibration_label.setText(f"Tap {remaining} more for new tempo")
            else:
                self.beat_calibration_label.setText(f"Tap {remaining} more times to calibrate")
            self.beat_calibration_label.setStyleSheet("color: #ffaa44; font-weight: bold;")
    
    def add_tempo_section(self, start_time, tempo, beat_interval):
        """Add a new tempo section starting from a specific time."""
        # Remove any existing tempo section that starts at the same time
        self.tempo_sections = [s for s in self.tempo_sections if s['start'] != start_time]
        
        # Find where this section ends (next phrase mark or end of song)
        end_time = self.duration
        for mark in sorted(self.phrase_marks):
            if mark > start_time:
                end_time = mark
                break
        
        # Check if we need to split an existing section
        for i, section in enumerate(self.tempo_sections):
            if section['start'] < start_time < section['end']:
                # Split the existing section
                old_end = section['end']
                section['end'] = start_time
                # Recalculate beats for the shortened section
                section['beats'] = []
                beat_time = section['start']
                while beat_time < section['end']:
                    section['beats'].append(beat_time)
                    beat_time += section['beat_interval']
        
        # Generate beats for this new section
        section_beats = []
        beat_time = start_time
        while beat_time < end_time:
            section_beats.append(beat_time)
            beat_time += beat_interval
        
        # Add the new tempo section
        self.tempo_sections.append({
            'start': start_time,
            'end': end_time,
            'tempo': tempo,
            'beat_interval': beat_interval,
            'beats': section_beats
        })
        
        # Rebuild the complete beat grid from all sections
        self.rebuild_beat_grid()
    
    def rebuild_beat_grid(self):
        """Rebuild the complete beat grid from tempo sections and base beats."""
        all_beats = []
        
        # Sort tempo sections by start time
        self.tempo_sections.sort(key=lambda x: x['start'])
        
        # If we have tempo sections, build the beat grid section by section
        if self.tempo_sections:
            current_pos = 0
            
            for i, section in enumerate(self.tempo_sections):
                # Add original beats before this section
                if i == 0:
                    # For first section, keep original beats before it
                    for beat in self.beat_times:
                        if beat < section['start']:
                            all_beats.append(beat)
                
                # Add beats from this tempo section
                all_beats.extend(section['beats'])
                current_pos = section['end']
            
            # Add any remaining original beats after the last section
            if current_pos < self.duration:
                for beat in self.beat_times:
                    if beat >= current_pos:
                        all_beats.append(beat)
        else:
            # No tempo sections, use original beats
            all_beats = self.beat_times.copy()
            return
        
        # Sort and remove duplicates
        self.beat_times = sorted(list(set(all_beats)))
        
        # Rebuild downbeats (alternating within each section for consistency)
        self.downbeats = [self.beat_times[i] for i in range(0, len(self.beat_times), 2)]
        
        print(f"Rebuilt beat grid with {len(self.beat_times)} beats across {len(self.tempo_sections)} tempo sections")
    
    def get_current_tempo(self, position):
        """Get the tempo at a specific position in the song."""
        # Check tempo sections
        for section in self.tempo_sections:
            if section['start'] <= position < section['end']:
                return section['tempo']
        
        # Return manual tempo or None
        return self.manual_tempo
    
    def snap_to_beat(self, position):
        """Snap a position to the nearest beat."""
        if not self.beat_times:
            # No beat information, return original position
            return position
        
        # Find the nearest beat
        import numpy as np
        beat_array = np.array(self.beat_times)
        distances = np.abs(beat_array - position)
        nearest_idx = np.argmin(distances)
        nearest_beat = self.beat_times[nearest_idx]
        
        # Only snap if within 0.5 seconds of a beat
        if distances[nearest_idx] < 0.5:
            return nearest_beat
        else:
            # Too far from any beat, keep original position
            return position
    
    def update_position(self):
        if self.is_playing:
            current = self.get_current_position()
        elif self.play_button.text() == "Resume":
            # Show paused position
            current = self.pause_position
        else:
            current = 0
            
        current_str = f"{int(current//60)}:{int(current%60):02d}"
        duration_str = f"{int(self.duration//60)}:{int(self.duration%60):02d}"
        self.position_label.setText(f"{current_str} / {duration_str}")
            
        # Update beat indicator (works even when paused)
        if self.beat_times and current > 0:
            # Find closest beat
            import numpy as np
            beat_array = np.array(self.beat_times)
            distances = np.abs(beat_array - current)
            closest_idx = np.argmin(distances)
            
            if closest_idx < len(self.beat_times):
                beat_time = self.beat_times[closest_idx]
                # Check if we're close to this beat (within 0.15 seconds)
                if distances[closest_idx] < 0.15:
                    # Find the most recent phrase mark and count beats from there
                    beats_since_phrase = 0
                    
                    if self.phrase_marks:
                        # Find the most recent phrase mark
                        most_recent_phrase = None
                        for mark in sorted(self.phrase_marks, reverse=True):
                            if mark <= current:
                                most_recent_phrase = mark
                                break
                        
                        if most_recent_phrase is not None:
                            # Count beats from the phrase mark to current position
                            # This accounts for tempo changes by counting actual beats
                            beats_since_phrase = 0
                            for i, bt in enumerate(self.beat_times):
                                if bt >= most_recent_phrase and bt <= beat_time:
                                    beats_since_phrase += 1
                            beats_since_phrase = max(0, beats_since_phrase - 1)  # Adjust for 0-indexing
                        else:
                            # No phrase before current position, count from start
                            beats_since_phrase = closest_idx
                    else:
                        # No phrases marked yet
                        beats_since_phrase = closest_idx
                    
                    # Calculate which 8-count we're in and the beat within it
                    eight_count_number = (beats_since_phrase // 8) + 1  # Which set of 8
                    beat_num = (beats_since_phrase % 8) + 1  # Beat within the 8-count
                    
                    # Check if it's a boom beat (low frequency)
                    is_boom = False
                    if self.downbeats:  # downbeats now contains boom beats
                        # Check if this beat time is in boom beats
                        for boom_time in self.downbeats:
                            if abs(beat_time - boom_time) < 0.01:  # Within 10ms
                                is_boom = True
                                break
                    
                    # Get current tempo if available
                    current_tempo = self.get_current_tempo(current)
                    tempo_str = f" @ {current_tempo:.0f}bpm" if current_tempo else ""
                    
                    # Show both the beat number and which 8-count set
                    if is_boom:
                        self.beat_indicator.setText(f"● Beat {beat_num}/8 [Set {eight_count_number}] (BOOM){tempo_str}")
                        self.beat_indicator.setStyleSheet("color: #ff4444; background-color: #441111; padding: 5px; border-radius: 5px;")
                    else:
                        self.beat_indicator.setText(f"○ Beat {beat_num}/8 [Set {eight_count_number}] (tick){tempo_str}")
                        self.beat_indicator.setStyleSheet("color: #44ff44; background-color: #114411; padding: 5px; border-radius: 5px;")
                else:
                    self.beat_indicator.setText("")
                    self.beat_indicator.setStyleSheet("color: #888;")
            
        # Stop at end
        if self.is_playing and current >= self.duration:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.play_button.setText("Play Full Song")
            QMessageBox.information(self, "Song Complete", 
                                  f"Song finished!\nMarked {len(self.phrase_marks)} phrase changes.")
    
    def save_and_exit(self):
        if len(self.phrase_marks) == 0:
            reply = QMessageBox.question(self, "No Marks", 
                                        "No phrase changes marked. Exit anyway?",
                                        QMessageBox.StandardButton.Yes | 
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        
        pygame.mixer.music.stop()
        
        # Emit all training data
        training_data = {
            'phrase_marks': self.phrase_marks,
            'beat_times': self.beat_times,
            'downbeats': self.downbeats,
            'tempo_sections': self.tempo_sections,
            'manual_tempo': self.manual_tempo
        }
        
        self.training_complete.emit(training_data)
        self.phrases_marked.emit(self.phrase_marks)  # Keep for backward compatibility
        self.accept()
    
    def cancel_dialog(self):
        """Cancel dialog and stop music."""
        pygame.mixer.music.stop()
        self.reject()
    
    def closeEvent(self, event):
        pygame.mixer.music.stop()
        super().closeEvent(event)