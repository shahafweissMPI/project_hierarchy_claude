import pandas as pd
import os
import IPython
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
            
            x_range = np.nanmax(self.x_traj) - np.nanmin(self.x_traj)#np.ptp(self.x_traj)
            y_range = np.nanmax(self.y_traj) - np.nanmin(self.y_traj)#np.ptp(self.y_traj)
            
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
            # Create a figure with specific size
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Display the frame
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw trajectory
            ax.plot(self.x_traj, self.y_traj, 'g-', linewidth=2, label='Trajectory')
            
            # Draw selected point if exists
            if self.selected_point is not None:
                ax.plot(self.selected_point[0], self.selected_point[1], 'ro', 
                       markersize=10, label='Selected Point')
           
            # Convert pixel axes to cm
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            
            # Create new axes with cm units
            ax2 = ax.secondary_xaxis('top', functions=(
                lambda x: x * self.px_to_cm_x,
                lambda x: x / self.px_to_cm_x))
            ax3 = ax.secondary_yaxis('right', functions=(
                lambda x: x * self.px_to_cm_y,
                lambda x: x / self.px_to_cm_y))
            
            # Label the axes
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax2.set_xlabel('X (cm)')
            ax3.set_ylabel('Y (cm)')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend()
            
            # Set window size to 640x480
            mng = plt.get_current_fig_manager()
            mng.window.setGeometry(100, 100, 640, 480)
            
            plt.show()
            
def calibrate_trajectory(x_trajectory, y_trajectory, video_path, csv_path='X.csv', row_number=None):
    """
    Main function to calibrate trajectory and get pixel-to-cm conversion.
    Also writes results to a CSV file.
    
    Parameters:
    -----------
    x_trajectory : array-like
        X coordinates of the trajectory in pixels
    y_trajectory : array-like
        Y coordinates of the trajectory in pixels
    video_path : str
        Path to the video file
    csv_path : str
        Path to the CSV file (default: 'X.csv')
    row_number : int
        Row number to write to (if None, appends new row)
    """
    # Run the calibration GUI
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = TrajectoryCalibrationWindow(x_trajectory, y_trajectory, video_path)
    window.show()
    app.exec_()
    
    # Calculate distances to selected point if a point was selected
    distances = None
    if window.selected_point is not None:
        #checK iF pX2cM needs to be reMoVed
        distances = np.sqrt(
            ((x_trajectory - window.selected_point[0]) * window.px_to_cm_x) ** 2 +
            ((y_trajectory - window.selected_point[1]) * window.px_to_cm_y) ** 2
        )
    
    # Prepare data for CSV
    data = {
        'px_to_cm_x': window.px_to_cm_x,
        'px_to_cm_y': window.px_to_cm_y,
        'selected_point_x': window.selected_point[0] if window.selected_point else None,
        'selected_point_y': window.selected_point[1] if window.selected_point else None,
        'min_distance_cm': np.min(distances) if distances is not None else None,
        'max_distance_cm': np.max(distances) if distances is not None else None
    }
    
    # # Write to CSV
    # try:
    #     # Check if file exists
    #     if os.path.exists(csv_path):
    #         df = pd.read_csv(csv_path)
    #     else:
    #         df = pd.DataFrame(columns=data.keys())
        
    #     # If row_number is specified, update that row
    #     if row_number is not None:
    #         # Extend DataFrame if row_number is beyond current size
    #         if row_number >= len(df):
    #             df = df.reindex(range(row_number + 1))
    #         df.loc[row_number] = data
    #     else:
    #         # Append new row
    #         df = df.append(data, ignore_index=True)
        
    #     # Write to CSV
    #     df.to_csv(csv_path, index=False)
    #     print(f"Data written to {csv_path}")
        
    # except Exception as e:
    #     print(f"Error writing to CSV: {str(e)}")
    
    return (window.px_to_cm_x, window.px_to_cm_y, 
            window.selected_point, distances)

# Example usage:
if __name__ == "__main__":
    # Sample data
    t = np.linspace(0, 10, 100)
    x = 100 * np.cos(t)
    y = 100 * np.sin(t)
    video_path = "path_to_your_video.mp4"
    
    # Get calibration results and write to CSV
    # To write to a specific row:
    row_N = 5  # specify the row number you want to write to
    results = calibrate_trajectory(x, y, video_path, row_number=row_N)
    
    # Or to append a new row:
    # results = calibrate_trajectory(x, y, video_path)