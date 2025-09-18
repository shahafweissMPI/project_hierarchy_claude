# -*- coding: utf-8 -*-
"""
📃 ./99_test_movie_files_vlc.py

🕰️ created on 2025-09-11

🤡 author: Dylan Festa

This is to test the video player on different, pre-saved video files, with hard-coded paths.

"""
#%%


import sys, os
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QComboBox, QFrame
)
import vlc

# ---------- Setup paths ----------
if os.name == 'nt':
    print("Windows OS detected!")
    path_gpfs_base = Path("\\\\gpfs.corp.brain.mpg.de\\STEM")
else:
    print("Linux or MacOS detected!")
    path_gpfs_base = Path("/gpfs/stem")

if not path_gpfs_base.exists():
    raise FileNotFoundError(f"Path {path_gpfs_base} does not exist on this system.")

path_project = path_gpfs_base / "data" / "project_hierarchy"
if not path_project.exists():
    raise FileNotFoundError(f"Path {path_project} does not exist on this system.")

available_files_dict = {
    "no movie": None,
    "original": path_project / "data" / "afm16924" / "240523" / "trial0" / "behavior" / "afm16924_240523_1.mp4",
    "compressed": path_project / "data" / "analysis" / "DylanTempPlots" / "VideosTest" / "test24_compressed1.mp4",
    "high compatibility": path_project / "data" / "analysis" / "DylanTempPlots" / "VideosTest" / "test24_highcompat.wmv",
}

for key, path_file in available_files_dict.items():
    if path_file is None:
        continue
    if not path_file.exists():
        raise FileNotFoundError(f"File for key '{key}' does not exist: {path_file}")

# ---------- Main player class ----------
class VideoWithPlot(QWidget):
    def __init__(self, fps):
        super().__init__()
        self.setWindowTitle("Test video player with VLC backend")
        self.fps = fps

        # VLC instance and media player
        self.vlc_instance = vlc.Instance()
        self.media_player = self.vlc_instance.media_player_new()

        # Video widget
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")

        # Top bar
        self.sourceCombo = QComboBox()
        self.sourceCombo.addItems(list(available_files_dict.keys()))
        self.sourceCombo.setCurrentText("no movie")
        self.loadBtn = QPushButton("Load")
        self.loadBtn.clicked.connect(self.load_selected)
        self.resetBtn = QPushButton("Reset")
        self.resetBtn.clicked.connect(self.reset_video)

        topbar = QHBoxLayout()
        topbar.addWidget(self.sourceCombo)
        topbar.addWidget(self.loadBtn)
        topbar.addWidget(self.resetBtn)

        # Controls
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.set_position)

        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(topbar)
        layout.addWidget(self.video_frame, stretch=5)
        layout.addLayout(controls)
        self.setLayout(layout)

        # Timer to sync slider with video
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

        # No movie on start
        self.open_file(None)

    def open_file(self, path):
        self.media_player.stop()
        if not path:
            self.playBtn.setText("▶")
            return
        p = Path(path)
        if not p.exists():
            print(f"File does not exist: {p}")
            return

        media = self.vlc_instance.media_new(str(p))
        self.media_player.set_media(media)

        # Set video output to the QWidget
        if sys.platform.startswith("linux"):
            # Ensure a valid native window id and set it for VLC (X11/XWayland)
            wid = int(self.video_frame.winId())
            self.media_player.set_xwindow(wid)
        elif sys.platform == "win32":
            self.media_player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(int(self.video_frame.winId()))

        self.media_player.play()
        self.playBtn.setText("⏸")

    def load_selected(self):
        key = self.sourceCombo.currentText()
        path = available_files_dict.get(key)
        self.open_file(path)

    def reset_video(self):
        self.sourceCombo.setCurrentText("no movie")
        self.open_file(None)

    def toggle_play(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.playBtn.setText("▶")
        else:
            self.media_player.play()
            self.playBtn.setText("⏸")

    def update_ui(self):
        if not self.media_player:
            return
        length = self.media_player.get_length()
        if length > 0:
            pos = self.media_player.get_time()
            self.slider.blockSignals(True)
            self.slider.setValue(int(1000 * pos / length))
            self.slider.blockSignals(False)

    def set_position(self, position):
        if self.media_player:
            length = self.media_player.get_length()
            seek_time = int(length * position / 1000)
            self.media_player.set_time(seek_time)


# ---------- Entry point ----------
if __name__ == "__main__":
    print("Now running: video_plot_sync_vlc.py")
    app = QApplication(sys.argv)
    fps = 30
    win = VideoWithPlot(fps)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
