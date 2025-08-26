from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

class TimelineWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)
        self.setMinimumWidth(400)
        
        self.start_time = 0
        self.end_time = 10
        self.current_position = 0
        self.boundary_time = 5
        self.is_playing = False
        
    def set_segment(self, start, end, boundary):
        self.start_time = start
        self.end_time = end
        self.boundary_time = boundary
        self.update()
    
    def set_position(self, position):
        self.current_position = position
        self.update()
    
    def set_playing(self, playing):
        self.is_playing = playing
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(40, 40, 40))
        
        # Timeline track
        track_height = 20
        track_y = (height - track_height) // 2
        
        # Draw the full segment
        painter.fillRect(0, track_y, width, track_height, QColor(60, 60, 60))
        
        if self.end_time > self.start_time:
            # Calculate positions
            duration = self.end_time - self.start_time
            
            # Draw progress
            progress_width = ((self.current_position - self.start_time) / duration) * width
            painter.fillRect(0, track_y, int(progress_width), track_height, QColor(100, 150, 200))
            
            # Draw boundary marker
            boundary_x = ((self.boundary_time - self.start_time) / duration) * width
            
            # Draw boundary line
            painter.setPen(QPen(QColor(255, 200, 0), 3))
            painter.drawLine(int(boundary_x), 0, int(boundary_x), height)
            
            # Draw boundary label
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(int(boundary_x) - 30, 12, "PHRASE")
            
            # Draw current position marker
            current_x = ((self.current_position - self.start_time) / duration) * width
            
            # Playhead
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(int(current_x), 0, int(current_x), height)
            
            # Draw countdown if close to boundary
            time_to_boundary = self.boundary_time - self.current_position
            if 0 < time_to_boundary <= 3:
                painter.setPen(QPen(QColor(255, 200, 100), 1))
                painter.drawText(int(current_x) + 5, height - 5, f"{time_to_boundary:.1f}s")