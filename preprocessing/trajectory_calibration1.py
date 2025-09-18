# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:25:31 2025

@author: su-weisss
"""

import pandas as pd
import os

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import numpy as np

class TrajectoryCalibrationWindow(QMainWindow):
    def __init__(self, x_trajectory, y_trajectory, video_path):
        super().__init__()
        self.x_traj = x_trajectory
        self.y_traj = y_trajectory
        self.video_path = video_path
        self.selected_point = None
        self.px_to_cm_x = None
        self.px_to_cm_y = None
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Trajectory Calibration')
        self.setGeometry(100, 100, 800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Plot trajectory
        self.ax.plot(self.x_traj, self.y_traj, 'b-', label='Trajectory')
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        self.ax.grid(True)
        
        # Create input fields
        input_layout = QHBoxLayout()
        
        # Width input
        width_label = QLabel('Width (cm):')
        self.width_input = QLineEdit()
        input_layout.addWidget(width_label)
        input_layout.addWidget(self.width_input)
        
        # Height input
        height_label = QLabel('Height (cm):')
        self.height_input = QLineEdit()
        input_layout.addWidget(height_label)
        input_layout.addWidget(self.height_input)
        
        # Calculate button
        self.calc_button = QPushButton('Calculate')
        self.calc_button.clicked.connect(self.calculate)
        input_layout.addWidget(self.calc_button)
        
        layout.addLayout(input_layout)
        
        # Connect mouse click event
        self.canvas.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        if event.inaxes == self.ax:
            self.selected_point = (event.xdata, event.ydata)
            # Clear previous points if any
            for artist in self.ax.get_children():
                if isinstance(artist, plt.Line2D) and artist.get_label() == 'Selected Point':
                    artist.remove()
            # Plot new point
            self.ax.plot(event.xdata, event.ydata, 'ro', markersize=10, label='Selected Point')
            self.canvas.draw()
    
    def calculate(self):
        try:
            width_cm = float(self.width_input.text())
            height_cm = float(self.height_input.text())
            
            # Calculate pixel to cm conversion
            x_range = np.ptp(self.x_traj)
            y_range = np.ptp(self.y_traj)
            
            self.px_to_cm_x = width_cm / x_range
            self.px_to_cm_y = height_cm / y_range
            
            # Show video frame with trajectory
            self.show_video_frame()
            
            self.close()
            
        except ValueError:
            print("Please enter valid numbers for width and height")
    
    def show_video_frame(self):
        # Read first frame of video
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert trajectory to image coordinates
            y_traj_img = self.y_traj.copy()  # Flip Y coordinates for image
            
            # Draw trajectory on frame
            pts = np.array([list(zip(self.x_traj, y_traj_img))], dtype=np.int32)
            cv2.polylines(frame, pts, False, (0, 255, 0), 2)
            
            # Draw selected point if exists
            if self.selected_point is not None:
                cv2.circle(frame, (int(self.selected_point[0]), 
                                 int(self.selected_point[1])), 
                          5, (0, 0, 255), -1)
            
            # Show frame
            cv2.imshow('Trajectory on Video', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def calibrate_trajectory(x_trajectory, y_trajectory, video_path):
    """
    Main function to calibrate trajectory and get pixel-to-cm conversion.
    
    Parameters:
    -----------
    x_trajectory : array-like
        X coordinates of the trajectory in pixels
    y_trajectory : array-like
        Y coordinates of the trajectory in pixels
    video_path : str
        Path to the video file
        
    Returns:
    --------
    tuple: (px_to_cm_x, px_to_cm_y, selected_point, distances)
        pixel-to-cm constants for x and y
        selected point coordinates
        distances to selected point at each trajectory point
    """
    app = QApplication(sys.argv)
    window = TrajectoryCalibrationWindow(x_trajectory, y_trajectory, video_path)
    window.show()
    app.exec_()
    
    # Calculate distances to selected point if a point was selected
    distances = None
    if window.selected_point is not None:
        distances = np.sqrt(
            ((x_trajectory - window.selected_point[0]) * window.px_to_cm_x) ** 2 +
            ((y_trajectory - window.selected_point[1]) * window.px_to_cm_y) ** 2
        )
    
    return (window.px_to_cm_x, window.px_to_cm_y, 
            window.selected_point, distances)

# Example usage:
if __name__ == "__main__":
    # Sample data
    t = np.linspace(0, 10, 100)
    x = 100 * np.cos(t)
    y = 100 * np.sin(t)
    video_path = "path_to_your_video.mp4"
    
    # Get calibration results
    px_to_cm_x, px_to_cm_y, selected_point, distances = calibrate_trajectory(x, y, video_path)
    
    if selected_point:
        print(f"Pixel-to-cm conversion: X: {px_to_cm_x:.4f}, Y: {px_to_cm_y:.4f}")
        print(f"Selected point: {selected_point}")
        print(f"Distances to point: min={np.min(distances):.2f}cm, max={np.max(distances):.2f}cm")