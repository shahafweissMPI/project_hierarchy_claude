# -*- coding: utf-8 -*-
"""
📃 ./99_test_movie_files_pqt5.py

🕰️ created on 2025-09-11

🤡 author: Dylan Festa

This is to test the video player on different, pre-saved video files, with hard-coded paths.

This is not sufficiently cross-platform compatible, the VLC version works much better,
and it will be the new standard going forward.
"""
#%%
from __future__ import annotations


import sys, os, numpy as np
from pathlib import Path

# if window, the path base is differet
if os.name == 'nt':
    print("Windows OS detected !")
    path_gpfs_base = Path("\\\\gpfs.corp.brain.mpg.de\\STEM")
else:
    print("Linux or MacOS detected !")
    path_gpfs_base = Path("/gpfs/stem")

if not path_gpfs_base.exists():
    raise FileNotFoundError(f"Path {path_gpfs_base} does not exist on this system.")

path_project = path_gpfs_base / "data" / "project_hierarchy" 
if not path_project.exists():
    raise FileNotFoundError(f"Path {path_project} does not exist on this system.")

#%%
available_files_dict = {
    "no movie" : None,
    "original": path_project / "data" / "afm16924" / "240523" / "trial0" / "behavior" / "afm16924_240523_1.mp4",
    "compressed": path_project /"data" /"analysis" / "DylanTempPlots" / "VideosTest" / "test24_compressed1.mp4",
    "high compatibility": path_project /"data" /"analysis" / "DylanTempPlots" / "VideosTest" / "test24_highcompat.wmv",
    #"compressed format 2": path_project
}

# check that all files exist (skip the "no movie" entry)
for key, path_file in available_files_dict.items():
    if path_file is None:
        continue
    if not path_file.exists():
        raise FileNotFoundError(f"File for key '{key}' does not exist: {path_file}")


#%%

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSlider, QFileDialog, QComboBox)


class VideoWithPlot(QWidget):
    def __init__(self, fps):
        super().__init__()
        self.setWindowTitle("Test video player with different movie files")
        # ---------- Qt Multimedia ----------
        self.player = QMediaPlayer(self)
        self.player.setNotifyInterval(100)          # 100 ms updates
        video = QVideoWidget()
        self.player.setVideoOutput(video)

        # ---------- Source selector + actions (top bar) ----------
        topbar = QHBoxLayout()
        self.sourceCombo = QComboBox()
        self.sourceCombo.addItems(list(available_files_dict.keys()))
        self.sourceCombo.setCurrentText("no movie")
        self.loadBtn = QPushButton("Load")
        self.loadBtn.clicked.connect(self.load_selected)
        self.resetBtn = QPushButton("Reset")
        self.resetBtn.clicked.connect(self.reset_video)
        topbar.addWidget(self.sourceCombo)
        topbar.addWidget(self.loadBtn)
        topbar.addWidget(self.resetBtn)

        # ---------- Transport controls ----------
        self.playBtn = QPushButton("▶")
        self.playBtn.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self.update_from_video)
        self.player.durationChanged.connect(self.slider.setMaximum)

        # ---------- Layout ----------
        controls = QHBoxLayout()
        controls.addWidget(self.playBtn)
        controls.addWidget(self.slider)
        main = QVBoxLayout(self)
        main.addLayout(topbar)
        main.addWidget(video, stretch=5)
        main.addLayout(controls)

        # ---------- Start with no movie ----------
        self.open_file(None)

    # ---- helpers ----
    def open_file(self, path):
        # Handle "no movie" / reset state
        if not path:
            self.player.stop()
            self.player.setMedia(QMediaContent())  # clear media
            block = self.slider.blockSignals(True)
            self.slider.setMaximum(0)
            self.slider.setValue(0)
            self.slider.blockSignals(block)
            self.playBtn.setText("▶")
            return
        p = Path(path)
        if not p.exists():
            print(f"File does not exist: {p}")
            return
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(p))))
        # Auto-play after loading
        self.player.play()
        self.playBtn.setText("⏸")
        # Reset slider to start
        block = self.slider.blockSignals(True)
        self.slider.setValue(0)
        self.slider.blockSignals(block)

    def load_selected(self):
        key = self.sourceCombo.currentText()
        path = available_files_dict.get(key)
        self.open_file(path)

    def reset_video(self):
        self.sourceCombo.setCurrentText("no movie")
        self.open_file(None)

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.playBtn.setText("▶")
        else:
            self.player.play()
            self.playBtn.setText("⏸")

    def update_from_video(self, ms):
        # keep slider in sync without feedback loops
        block = self.slider.blockSignals(True)
        self.slider.setValue(ms)
        self.slider.blockSignals(block)


if __name__ == "__main__":
    print("Now running: video_plot_sync.py")
    app = QApplication(sys.argv)
    # dummy signal: 10 minutes of 100 Hz samples
    fps = 30                                  # real video fps if known
    duration_s = 600
    win = VideoWithPlot(fps)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())